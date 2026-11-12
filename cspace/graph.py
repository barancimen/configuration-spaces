"""The constraint graph of a mechanism, used to work on small parts of a large one.

Bodies are nodes. Each constraint is a hyperedge over the bodies it involves (a coupling
involves the bodies of the joints it couples). The world is fixed, so it is left out of the
graph: bodies connected only through the world move independently, and the configuration
space is the product of the spaces of the connected groups.
"""
from collections import deque
from dataclasses import replace

from sympy import S

from .joints import KINDS
from .spec import SpecError

WORLD = "world"


def _body(ref):
    return str(ref).partition(".")[0]


class Graph:
    def __init__(self, spec):
        self.spec = spec
        self.joint_bodies = {}
        for c in spec.constraints:
            if c.kind in KINDS:
                if not isinstance(c.args, list) or len(c.args) != 2:
                    raise SpecError(f"{c.kind} {c.name!r}: expected [frame, frame]")
                self.joint_bodies[c.name] = {_body(c.args[0]), _body(c.args[1])}
        self.edges = [(c, self._bodies(c) - {WORLD}) for c in spec.constraints]
        self.adjacent = {b: set() for b in spec.bodies if b != WORLD}
        for c, bodies in self.edges:
            for b in bodies:
                if b not in self.adjacent:
                    raise SpecError(f"constraint {c.name!r} refers to unknown body {b!r}")
                self.adjacent[b] |= bodies - {b}

    def _joints_of(self, c):
        if c.kind == "couple":
            return [str(k).partition(".")[0] for k in (c.args or {})]
        if c.kind == "gear":
            return [str(k).partition(".")[0] for k in (c.args or [])]
        return []

    def _bodies(self, c):
        if c.kind in KINDS:
            return set(self.joint_bodies[c.name])
        if c.kind == "no_overlap":
            return {str(b) for b in (c.args or [])}
        out = set()
        for j in self._joints_of(c):
            if j not in self.joint_bodies:
                raise SpecError(f"{c.kind} {c.name!r} refers to unknown joint {j!r}")
            out |= self.joint_bodies[j]
        return out

    def group(self, seeds):
        """All bodies connected to the seeds without passing through the world (exact)."""
        return self.ball(seeds, None)

    def ball(self, seeds, radius):
        """Bodies within `radius` steps of the seeds (None: no limit)."""
        seen = {b for b in seeds if b != WORLD}
        frontier = deque((b, 0) for b in seen)
        while frontier:
            b, d = frontier.popleft()
            if radius is not None and d >= radius:
                continue
            for n in self.adjacent[b]:
                if n not in seen:
                    seen.add(n)
                    frontier.append((n, d + 1))
        return seen

    def pieces(self, part):
        """Split `part` into pieces connected within it. Returns [(bodies, attached_to_world)]."""
        part, out, seen = set(part), [], set()
        for start in part:
            if start in seen:
                continue
            piece, todo = {start}, [start]
            while todo:
                b = todo.pop()
                for n in self.adjacent[b]:
                    if n in part and n not in piece:
                        piece.add(n)
                        todo.append(n)
            seen |= piece
            attached = any(WORLD in self.joint_bodies.get(c.name, ()) and bs <= piece
                           for c, bs in self.edges if bs & piece)
            out.append((piece, attached))
        return out

    def prune(self, sub, protected):
        """Remove dangling bodies: a body (not protected) whose only constraint is one joint can
        always follow the rest (some joint value within its limits fits any pose of the other
        side), so removing it and its joint leaves the other bodies' configurations unchanged.
        Repeated until nothing dangles."""
        protected = set(protected) | {WORLD}
        cons = list(sub.constraints)
        incident = {b: [] for b in sub.bodies}
        for i, c in enumerate(cons):
            for b in self._bodies(c):
                incident.setdefault(b, []).append(i)
        alive, removed = set(range(len(cons))), set()
        todo = [b for b in incident if b not in protected]
        while todo:
            b = todo.pop()
            if b in removed or b in protected:
                continue
            live = [i for i in incident[b] if i in alive]
            if len(live) != 1 or cons[live[0]].kind not in KINDS:
                continue
            (i,) = live
            alive.discard(i)
            removed.add(b)
            todo += [o for o in self._bodies(cons[i]) if o != b]
        if not removed:
            return sub
        bodies = {n: v for n, v in sub.bodies.items() if n not in removed}
        return replace(sub, bodies=bodies, constraints=[c for i, c in enumerate(cons) if i in alive])

    def restricted(self, part, group):
        """The connected group with every joint outside `part` held at zero (a `fixed` joint).
        Every configuration of the result is a real configuration. None if some joint outside
        the part cannot be held at zero because its limits exclude zero."""
        part = set(part)
        sub = self.subspec(group, first=None)
        cons = []
        for c in sub.constraints:
            bs = self.joint_bodies.get(c.name) if c.kind in KINDS else None
            if bs is not None and not (bs & part) and KINDS[c.kind].roles is not None \
                    and c.kind not in ("fixed", "free"):
                if not self._zero_allowed(c):
                    return None
                c = replace(c, kind="fixed", opts={})
            cons.append(c)
        return replace(sub, constraints=cons)

    def _zero_allowed(self, c):
        ranges = list((c.opts.get("limits") or {}).values())
        if "range" in c.opts:
            ranges.append(c.opts["range"])
        for rng in ranges:
            lo, hi = (self.spec.num(v) for v in rng)
            if (lo <= 0) != S.true or (hi >= 0) != S.true:
                return False
        return True

    def subspec(self, bodies, first=None):
        """The mechanism restricted to `bodies` (plus the world), keeping only the constraints
        among them. Dropping constraints can only add configurations. `first` is listed first,
        so it becomes the root of a piece that is not attached to the world."""
        keep = set(bodies) | {WORLD}
        kept = [c for c, bs in self.edges if c.kind in KINDS and (bs | {WORLD}) <= keep]
        names = {c.name for c in kept}
        for c, bs in self.edges:
            if c.kind in ("couple", "gear") and set(self._joints_of(c)) <= names:
                kept.append(c)
            elif c.kind == "no_overlap" and bs <= keep:
                kept.append(c)
        order = {id(c): i for i, c in enumerate(self.spec.constraints)}
        kept.sort(key=lambda c: order[id(c)])
        if not hasattr(self, "_order"):
            self._order = {n: i for i, n in enumerate(self.spec.bodies)}
        names = sorted((n for n in keep if n in self.spec.bodies),
                       key=lambda n: (n != WORLD, n != first, self._order[n]))
        sub = {n: self.spec.bodies[n] for n in names}
        return replace(self.spec, bodies=sub, constraints=kept, claims=[])
