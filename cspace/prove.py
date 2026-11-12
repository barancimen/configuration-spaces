"""Prove or refute claims about a mechanism's configuration space with z3.

Every answer is either established by the solver or reported as unknown:
  proved   the claim holds (for every admissible parameter value)
  refuted  it fails; the counterexample is a real configuration
  unknown  neither could be established within the limits

Large mechanisms are handled by working on a small part:
  * bodies connected only through the fixed world move independently, so a claim only needs
    the connected groups of the bodies it mentions (exact);
  * for `always` claims, the prover first keeps only the bodies within 0, 1, 2, ... steps of
    the mentioned ones. Dropping constraints only adds configurations, so a proof there is a
    proof for the whole mechanism. A counterexample there may not be real, so it widens instead.
"""
import time
from dataclasses import dataclass, field

import z3
from sympy import And, Not, true

from .claims import ClaimError, Evaluator, mentioned, parse
from .compile import compile_spec
from .graph import WORLD, Graph
from .logic import Z3, ForAll, quantified
from .spec import SpecError


@dataclass
class Result:
    name: str
    kind: str
    text: str
    status: str                  # proved | refuted | unknown | error
    detail: str = ""
    bodies: int = 0
    group: int = 0
    seconds: float = 0.0
    model: dict = field(default_factory=dict)


def read_claims(entries):
    out = []
    for i, e in enumerate(entries, 1):
        if not isinstance(e, dict):
            raise SpecError(f"claim {i} must be a mapping with 'always:' or 'possible:'")
        kinds = [k for k in ("always", "possible") if k in e]
        if len(kinds) != 1:
            raise SpecError(f"claim {i} needs exactly one of 'always:' or 'possible:'")
        out.append((str(e.get("name", f"claim{i}")), kinds[0], str(e[kinds[0]])))
    return out


def _solve(formula, quant, timeout):
    reasons = []
    for label in (("z3", "nlqsat") if quant else ("z3",)):
        s = z3.Solver() if label == "z3" else z3.Tactic("nlqsat").solver()
        s.set("timeout", int(timeout * 1000))
        s.add(formula)
        try:
            r = s.check()
        except z3.Z3Exception as err:
            reasons.append(f"{label}: {err}")
            continue
        if r == z3.unknown:
            reasons.append(f"{label}: {s.reason_unknown()}")
            continue
        return r, (s.model() if r == z3.sat else None), label
    return z3.unknown, None, "; ".join(reasons)


def _groups(query):
    """Split a conjunction into groups that share no free variables."""
    parts = list(query.args) if isinstance(query, And) else [query]
    parent = list(range(len(parts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for i, f in enumerate(parts):
        for v in f.free_symbols:
            if v in owner:
                parent[find(i)] = find(owner[v])
            else:
                owner[v] = i
    groups = {}
    for i, f in enumerate(parts):
        groups.setdefault(find(i), []).append(f)
    return [And(*g) for g in groups.values()]


def _solve_split(query, timeout):
    """Solve each independent group; the conjunction is satisfiable iff every group is.
    Returns (result, model values {name: z3 value}, how, None)."""
    values, hows, unknown = {}, set(), None
    for g in sorted(_groups(query), key=lambda g: len(g.free_symbols)):
        T = Z3()
        fz = T.formula(g)
        r, model, how = _solve(z3.And(fz, *T.defs), quantified(g), timeout)
        if r == z3.unsat:
            return r, None, how, None
        if r == z3.unknown:
            unknown = how
            continue
        hows.add(how)
        values.update({n: model.eval(v, model_completion=True) for n, v in T.vars.items()})
    if unknown is not None:
        return z3.unknown, None, unknown, None
    return z3.sat, values, "+".join(sorted(hows)) or "z3", None


def _value(v):
    if z3.is_rational_value(v):
        return str(v.as_fraction()) if v.denominator_as_long() != 1 else str(v.numerator_as_long())
    if z3.is_algebraic_value(v):
        return v.as_decimal(8).rstrip("?") + "..."
    return str(v)


class Prover:
    def __init__(self, spec, timeout=30.0, max_bodies=12, log=None, budget=120.0):
        self.spec, self.timeout, self.max_bodies, self.budget = spec, timeout, max_bodies, budget
        self.log = log or (lambda msg: None)
        self.graph = Graph(spec)

    def prove(self, name, kind, text):
        t0 = time.time()
        try:
            res = self._prove(name, kind, text)
        except SpecError as err:
            res = Result(name, kind, text, "error", str(err))
        res.seconds = time.time() - t0
        return res

    def _prove(self, name, kind, text):
        tree = parse(text)
        seeds = []
        for n in mentioned(tree):
            if n in self.spec.bodies and n != WORLD:
                new = [n]
            else:
                new = sorted(self.graph.joint_bodies.get(n, set()) - {WORLD})
            seeds += [b for b in new if b not in seeds]
        self.root = seeds[0] if seeds else None
        self.seeds = seeds
        group = self.graph.group(seeds)
        base = Result(name, kind, text, "unknown", group=len(group))

        if len(group) <= self.max_bodies:
            return self._attempt(base, tree, self.graph.subspec(group, first=self.root), "exact")
        radius, last, tried, t0 = 0, None, 0, time.time()
        while True:
            part = self.graph.ball(seeds, radius)
            if len(part) > self.max_bodies or part == last:
                why = f"the next step exceeds --max-bodies {self.max_bodies}"
                break
            if time.time() - t0 > self.budget:
                why = f"the time budget of {self.budget:.0f} s per claim ran out"
                break
            last, radius = part, radius + 1
            # Mentioned bodies split over separate pieces, one of them floating: their relative
            # position is unconstrained in the relaxation, so it would only find meaningless
            # answers. Skipping an attempt cannot make an answer wrong.
            holding = [(p, att) for p, att in self.graph.pieces(part) if p & set(seeds)]
            floating = len(holding) > 1 and not all(att for _, att in holding)
            attempts = [] if floating else [(self.graph.subspec(part, first=self.root), "relaxed")]
            restricted = self.graph.restricted(part, group)
            if restricted is not None:
                attempts.append((restricted, "restricted"))
            for sub, mode in attempts:
                tried += 1
                res = self._attempt(base, tree, sub, mode, len(part))
                if res.status in ("proved", "refuted", "unknown"):
                    return res
        base.detail = (f"no conclusion with up to {len(last or ())} of the {len(group)} connected bodies "
                       f"({tried} attempt(s)), and {why}")
        return base

    def _attempt(self, base, tree, sub, mode, part_size=None):
        """mode: 'exact' (the whole connected group), 'relaxed' (constraints to the rest dropped:
        more configurations) or 'restricted' (joints outside the part held at zero: fewer)."""
        n = part_size if part_size is not None else len(sub.bodies) - (WORLD in sub.bodies)
        res = Result(base.name, base.kind, base.text, "unknown", group=base.group, bodies=n)
        sub = self.graph.prune(sub, self.seeds)
        system = compile_spec(sub, shapes=False)
        phi = Evaluator(sub, system).formula(tree)
        C = system.formula()
        A = And(*self.spec.assumptions) if self.spec.assumptions else true
        params = set(self.spec.params) & (C.free_symbols | phi.free_symbols | A.free_symbols)
        if base.kind == "always":
            query, sat_means = And(A, C, Not(phi)), "refuted"
        elif params:
            body = Not(And(C, phi))
            query = And(A, ForAll(system.variables, body) if system.variables else body)
            sat_means = "refuted"
        else:
            query, sat_means = And(A, C, phi), "proved"
        t0 = time.time()
        r, model, how, _ = _solve_split(query, self.timeout)
        self.log(f"{base.name}: {n} bodies ({mode}), {len(system.variables)} variables -> {r} "
                 f"in {time.time() - t0:.2f} s")
        if r == z3.unknown:
            res.detail = f"solver gave up ({how}) on {n} bodies ({mode})"
            return res
        status = sat_means if r == z3.sat else ("refuted" if sat_means == "proved" else "proved")
        # which conclusions each kind of system supports
        valid = {"exact": True,
                 "relaxed": (base.kind, status) in (("always", "proved"), ("possible", "refuted")),
                 "restricted": (base.kind, status) in (("always", "refuted"), ("possible", "proved"))}
        if not valid[mode]:
            res.status = "inconclusive"
            return res
        if r == z3.sat and model is not None:
            names = [p.name for p in params]          # a refuted 'possible': only parameters matter
            if not (base.kind == "possible" and params):
                names = [v.name for v in system.variables] + names
            res.model = {k: _value(model[k]) for k in names if k in model}
        res.status = status
        res.detail = {"exact": "exact",
                      "relaxed": f"{n} nearest of {base.group} connected bodies, others dropped",
                      "restricted": f"{n} nearest of {base.group} connected bodies free, "
                                    "joints of the others held at zero"}[mode] + f", {how}"
        return res

    def prove_all(self, entries):
        return list(self.iter_prove(entries))

    def iter_prove(self, entries):
        for c in read_claims(entries):
            yield self.prove(*c)


def report(results):
    lines = []
    for r in results:
        lines.append(f"[{r.status.upper():7s}] {r.name}: {r.kind} {r.text}")
        info = f"{r.detail}; " if r.detail else ""
        lines.append(f"          {info}{r.seconds:.2f} s")
        if r.model:
            if r.kind == "always":
                kind = "counterexample"
            elif r.status == "proved":
                kind = "witness"
            else:
                kind = "parameters with no such configuration"
            shown = list(r.model.items())[:16]
            lines.append(f"          {kind}: " + ", ".join(f"{k} = {v}" for k, v in shown)
                         + (" ..." if len(r.model) > 16 else ""))
    return "\n".join(lines)
