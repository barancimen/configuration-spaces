"""Turn a Spec into a System: variables, polynomial equations and inequalities."""
from dataclasses import dataclass, field

from sympy import Add, And, Eq, Ge, Symbol, cos, eye, expand, sin, true

from .joints import KINDS, coordinate, coupling, limits
from .se3 import inv, split
from .spec import SpecError, truthy

OTHER_KINDS = {"couple", "gear", "no_overlap"}
ALL_KINDS = set(KINDS) | OTHER_KINDS
ANGLE_ROLES = (("c", "s"), ("c1", "s1"), ("c2", "s2"))


@dataclass
class Joint:
    name: str
    kind: str
    a: str
    Fa: object
    b: str
    Fb: object
    opts: dict
    tree: bool = False
    parent: str = None                           # tree joints: the end that was placed first
    values: dict = field(default_factory=dict)   # role -> expression in the variables
    theta: dict = field(default_factory=dict)    # angle role -> angle symbol (--trig only)
    R: object = None
    p: object = None


@dataclass
class System:
    name: str
    params: list
    assumptions: list
    variables: list
    equations: list          # each means  expr == 0
    inequalities: list       # each means  expr >= 0
    formulas: list           # further conditions as SymPy booleans (no_overlap)
    poses: dict              # body -> 4x4 world pose
    shapes: dict             # body -> predicate in body coordinates x, y, z
    joints: list
    notes: list
    naive_dof: int
    coords: str
    trig: bool

    def formula(self):
        parts = [Eq(e, 0) for e in self.equations] + self.constraints_list()
        return And(*parts) if parts else true

    def constraints_list(self):
        return [Ge(g, 0) for g in self.inequalities] + list(self.formulas)

    def constraints(self):
        c = self.constraints_list()
        return And(*c) if c else true


class Reducer:
    """Normal form modulo c^2 + s^2 = 1 and |q|^2 = 1 (replace s^2 and q0^2)."""

    def __init__(self):
        self.rules = []

    def add(self, var, square):
        self.rules.append((var, square))

    def __call__(self, e):
        e = expand(e)
        for var, square in self.rules:
            if not e.has(var):
                continue
            out = []
            for t in Add.make_args(e):
                coeff, k = t.as_coeff_exponent(var)
                k = int(k)
                out.append(coeff * var**(k % 2) * square**(k // 2) if k >= 2 else t)
            e = expand(Add(*out))
        return e

    def matrix(self, M):
        return M.applyfunc(self)

    def canon(self, e):
        """Normal form, rational content removed, sign fixed: one representative per equation."""
        e = self(e)
        if e == 0:
            return e
        _, e = e.as_content_primitive()
        e = expand(e)
        return -e if e.could_extract_minus_sign() else e


def _cap(name):
    return name[0].upper() + name[1:]


def _pair_joint(spec, c):
    if not isinstance(c.args, list) or len(c.args) != 2:
        raise SpecError(f"{c.kind} {c.name!r}: expected [frame, frame], got {c.args!r}")
    a, Fa = spec.ref(c.args[0])
    b, Fb = spec.ref(c.args[1])
    if a == b:
        raise SpecError(f"{c.kind} {c.name!r} joins body {a!r} to itself")
    opts = dict(c.opts)
    if c.kind == "distance":
        if "length" not in opts:
            raise SpecError(f"distance {c.name!r} needs a length")
        opts["length"] = spec.num(opts["length"])
    return Joint(c.name, c.kind, a, Fa, b, Fb, opts)


def compile_spec(spec, coords="reduced", trig=False):
    if coords not in ("reduced", "maximal"):
        raise SpecError(f"unknown coordinate choice {coords!r}")
    joints = [_pair_joint(spec, c) for c in spec.constraints if c.kind in KINDS]
    others = [c for c in spec.constraints if c.kind in OTHER_KINDS]
    names = [c.name for c in spec.constraints]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SpecError(f"constraint names used twice: {sorted(dup)}")
    notes = []

    # ---- spanning tree, in file order ----
    placed, tree = {"world"}, []

    def grow():
        while True:
            for j in joints:
                if j.tree or KINDS[j.kind].roles is None or truthy(j.opts.get("closing", "false")):
                    continue
                if (j.a in placed) != (j.b in placed):
                    j.tree, j.parent = True, (j.a if j.a in placed else j.b)
                    placed.add(j.b if j.parent == j.a else j.a)
                    tree.append(j)
                    break
            else:
                return

    if coords == "reduced":
        grow()
    for body in spec.bodies:
        if body in placed or body == "world":
            continue
        jname = body if body not in names else "free" + _cap(body)
        j = Joint(jname, "free", "world", eye(4), body, eye(4), {}, tree=True, parent="world")
        joints.append(j)
        tree.append(j)
        placed.add(body)
        if coords == "reduced":
            notes.append(f"{body}: no tree joint reaches it from world, so it gets a free joint "
                         f"({jname}: 3 positions + unit quaternion)")
            grow()

    # ---- variables ----
    red = Reducer()
    variables, equations, taken = [], [], set(spec.env)
    relations = []           # circle and quaternion relations, kept out of the normal form
    angle_subs = {}

    def new_symbol(name):
        if name in taken:
            raise SpecError(f"generated variable {name!r} clashes with another name; rename a joint or parameter")
        taken.add(name)
        return Symbol(name, real=True)

    for j in tree:
        roles = KINDS[j.kind].roles
        v = {r: new_symbol(r + _cap(j.name)) for r in roles}
        j.values = v
        for cr, sr in ANGLE_ROLES:
            if cr in v:
                red.add(v[sr], 1 - v[cr]**2)
                relations.append((v[cr]**2 + v[sr]**2 - 1, "circle"))
                if trig:
                    th = new_symbol("th" + cr[1:] + _cap(j.name))
                    j.theta[cr] = th
                    angle_subs[v[cr]], angle_subs[v[sr]] = cos(th), sin(th)
        if "q0" in v:
            q = [v[f"q{i}"] for i in range(4)]
            red.add(q[0], 1 - q[1]**2 - q[2]**2 - q[3]**2)
            relations.append((sum(x**2 for x in q) - 1, "quaternion"))
        variables += list(v.values())

    # ---- poses ----
    poses = {"world": eye(4)}
    for j in tree:
        J = KINDS[j.kind].T(j.values)
        if j.parent == j.a:
            steps, start, child = (j.Fa, J, inv(j.Fb)), poses[j.a], j.b
        else:
            steps, start, child = (j.Fb, inv(J), inv(j.Fa)), poses[j.b], j.a
        M = start
        for S in steps:
            M = red.matrix(M * S)
        poses[child] = M
        j.R, j.p = split(J)

    # ---- closing constraints ----
    for j in joints:
        if j.tree:
            continue
        Trel = red.matrix(inv(red.matrix(poses[j.a] * j.Fa)) * red.matrix(poses[j.b] * j.Fb))
        j.R, j.p = split(Trel)
        equations += KINDS[j.kind].implicit(j.R, j.p, j.opts)
        j.values = {k: red(e) for k, e in KINDS[j.kind].readout(j.R, j.p).items()}

    # ---- limits ----
    inequalities = []
    for j in joints:
        inequalities += [red(g) for g in limits(j, spec, trig)]

    # ---- couplings and other n-ary constraints ----
    by_name = {j.name: j for j in joints}

    def joint_coord(key):
        jname, _, cname = str(key).partition(".")
        if jname not in by_name:
            raise SpecError(f"coupling refers to unknown joint {jname!r}")
        return coordinate(by_name[jname], cname or None)

    formulas, n_couplings = [], 0
    for c in others:
        if c.kind == "couple":
            if not isinstance(c.args, dict) or not c.args:
                raise SpecError(f"couple {c.name!r}: expected a mapping joint -> coefficient")
            terms = [(spec.num(k), joint_coord(key)) for key, k in c.args.items()]
            equations += coupling(terms, spec.num(c.opts.get("equals", 0)), trig)
            n_couplings += 1
        elif c.kind == "gear":
            if not isinstance(c.args, list) or len(c.args) != 2 or len(c.opts.get("teeth", [])) != 2:
                raise SpecError(f"gear {c.name!r}: expected [joint, joint] and teeth: [n1, n2]")
            n1, n2 = (spec.num(t) for t in c.opts["teeth"])
            sign = -1 if truthy(c.opts.get("internal", "false")) else 1
            terms = [(n1, joint_coord(c.args[0])), (sign * n2, joint_coord(c.args[1]))]
            equations += coupling(terms, spec.num(c.opts.get("phase", 0)), trig)
            n_couplings += 1
        elif c.kind == "no_overlap":
            from .shapes import no_overlap
            f, note = no_overlap(spec, c, poses, red)
            formulas.append(f)
            if note:
                notes.append(note)

    # ---- normal form and deduplication ----
    unknowns = set(variables) | {th for j in tree for th in j.theta.values()}
    eqs = [r for r, what in relations if not (trig and what == "circle")]
    seen = set(eqs)
    for e in equations:
        e = red.canon(e)
        if e == 0 or e in seen:
            continue
        seen.add(e)
        eqs.append(e)
        if not e.free_symbols & unknowns:
            if not e.free_symbols:
                notes.append(f"equation {e} = 0 can never hold: the configuration space is empty")
            else:
                notes.append(f"equation {e} = 0 involves parameters only: the mechanism assembles "
                             "only for parameter values that satisfy it")
    ineqs = list(dict.fromkeys(expand(g) for g in inequalities))

    # ---- trigonometric view: (c, s) -> (cos th, sin th) for tree hinges ----
    if trig and angle_subs:
        cosines = {j.values[cr]: j.theta[cr] for j in tree for cr, _ in ANGLE_ROLES if cr in j.theta}
        sines = {j.values[sr] for j in tree for cr, sr in ANGLE_ROLES if cr in j.theta}
        variables = [cosines.get(v, v) for v in variables if v not in sines]
        eqs = [expand(e.subs(angle_subs)) for e in eqs]
        ineqs = [expand(g.subs(angle_subs)) for g in ineqs]
        formulas = [f.subs(angle_subs) for f in formulas]
        poses = {b: M.subs(angle_subs) for b, M in poses.items()}
        for j in joints:
            j.values = {k: e.subs(angle_subs) for k, e in j.values.items()}
            j.R, j.p = j.R.subs(angle_subs), j.p.subs(angle_subs)

    # ---- shapes ----
    shapes = {}
    for b in spec.bodies.values():
        if b.shape is not None:
            from .shapes import body_predicate
            shapes[b.name] = body_predicate(spec, b)

    naive = 6 * sum(1 for b in spec.bodies if b != "world") \
        - sum(KINDS[j.kind].codim for j in joints) - n_couplings
    notes.insert(0, "tree: " + (", ".join(f"{j.name} ({j.parent} -> {j.b if j.parent == j.a else j.a})"
                                          for j in tree) or "empty"))
    closing = [j.name for j in joints if not j.tree]
    if closing:
        notes.insert(1, "closing constraints: " + ", ".join(closing))

    return System(spec.name, list(spec.params), list(spec.assumptions), variables, eqs, ineqs,
                  formulas, {k: v for k, v in poses.items() if k != "world"}, shapes, joints,
                  notes, naive, coords, trig)
