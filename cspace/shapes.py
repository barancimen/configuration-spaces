"""Solid shapes as CSG trees, and exact no-overlap conditions between two bodies.

Leaves: box [lx, ly, lz] (centred), ball r, cylinder [r, h] (axis z, centred), hull [[x,y,z], ...]
(numeric vertices), where [relations in x, y, z]. Combinators: union, intersection, difference
(first child minus the others). Any node may carry a placement `at`, `z`, `x`.

Each node compiles to a closed predicate and an open one (its interior).
Two bodies do not overlap when no point is interior to both.
"""
from dataclasses import dataclass

import numpy as np
from sympy import And, Eq, Ge, Gt, Le, Lt, Matrix, Ne, Or, Symbol, expand, false, true, zeros
from sympy.core.relational import Relational

from .logic import ForAll

from .se3 import inv
from .spec import SpecError

X, Y, Z = (Symbol(n, real=True) for n in "xyz")
P = Matrix([X, Y, Z])
LEAVES = ("box", "ball", "cylinder", "hull", "where")
NODES = ("union", "intersection", "difference")


# ---------- formulas ----------

def negate(f):
    """Negation pushed down to the atoms, so the result contains no Not."""
    if f is true:
        return false
    if f is false:
        return true
    if isinstance(f, And):
        return Or(*[negate(a) for a in f.args])
    if isinstance(f, Or):
        return And(*[negate(a) for a in f.args])
    flip = {Gt: Le, Ge: Lt, Lt: Ge, Le: Gt}
    for cls, other in flip.items():
        if isinstance(f, cls):
            return other(f.lhs, f.rhs)
    if isinstance(f, Eq):
        return Or(Lt(f.lhs, f.rhs), Gt(f.lhs, f.rhs))
    if isinstance(f, Ne):
        return Eq(f.lhs, f.rhs)
    raise SpecError(f"cannot negate {f}")


def map_atoms(f, fn):
    if isinstance(f, (And, Or)):
        return f.func(*[map_atoms(a, fn) for a in f.args])
    if isinstance(f, Relational):
        return fn(f)
    return f


def _tidy(e, red=None):
    """Expanded, reduced, positive rational content removed (the sign is kept)."""
    e = red(e) if red else expand(e)
    if e.is_number:
        return e
    c, _ = e.as_content_primitive()
    return expand(e / c) if c.is_positive and c != 1 else e


def ge(e, red=None):
    return Ge(_tidy(e, red), 0)


def gt(e, red=None):
    return Gt(_tidy(e, red), 0)


def at_point(f, point, red=None):
    """Formula f(x, y, z) evaluated at a point given by three expressions."""
    g = f.subs(dict(zip((X, Y, Z), point)), simultaneous=True)
    return map_atoms(g, lambda r: r.func(_tidy(r.lhs - r.rhs, red), 0))


# ---------- CSG ----------

def _kind(spec, node):
    if not isinstance(node, dict):
        raise SpecError(f"shape must be a mapping, got {node!r}")
    found = [k for k in node if k in LEAVES + NODES]
    if len(found) != 1:
        raise SpecError(f"shape needs exactly one of {LEAVES + NODES}, got {sorted(node)}")
    return found[0]


def _placement(spec, node):
    return spec.placement({k: node[k] for k in ("at", "z", "x") if k in node})


def _box(spec, v):
    if not isinstance(v, list) or len(v) != 3:
        raise SpecError(f"box expects [lx, ly, lz], got {v!r}")
    return [spec.num(c) / 2 for c in v]


def _cyl(spec, v):
    if not isinstance(v, list) or len(v) != 2:
        raise SpecError(f"cylinder expects [r, h], got {v!r}")
    return spec.num(v[0]), spec.num(v[1]) / 2


def predicates(spec, node):
    """(closed, open) predicates of a CSG node in its parent's coordinates."""
    kind = _kind(spec, node)
    v = node[kind]
    if kind == "box":
        h = _box(spec, v)
        closed = And(*[ge(h[i] - P[i]) for i in range(3)], *[ge(h[i] + P[i]) for i in range(3)])
        opened = And(*[gt(h[i] - P[i]) for i in range(3)], *[gt(h[i] + P[i]) for i in range(3)])
    elif kind == "ball":
        r = spec.num(v)
        closed, opened = ge(r**2 - P.dot(P)), gt(r**2 - P.dot(P))
    elif kind == "cylinder":
        r, h = _cyl(spec, v)
        closed = And(ge(r**2 - X**2 - Y**2), ge(h - Z), ge(h + Z))
        opened = And(gt(r**2 - X**2 - Y**2), gt(h - Z), gt(h + Z))
    elif kind == "hull":
        poly = _hull(spec, v)
        closed = And(*[ge(d - n.dot(P)) for n, d in poly.faces])
        opened = And(*[gt(d - n.dot(P)) for n, d in poly.faces])
    elif kind == "where":
        rels = [spec.num(t, {"x": X, "y": Y, "z": Z}) for t in (v if isinstance(v, list) else [v])]
        if not all(isinstance(r, Relational) for r in rels):
            raise SpecError(f"where expects comparisons in x, y, z, got {v!r}")
        closed, opened = And(*[_closed_rel(r) for r in rels]), And(*[_open_rel(r) for r in rels])
    else:
        if not isinstance(v, list) or not v:
            raise SpecError(f"{kind} expects a list of shapes")
        kids = [predicates(spec, k) for k in v]
        if kind == "union":
            closed, opened = Or(*[c for c, _ in kids]), Or(*[o for _, o in kids])
        elif kind == "intersection":
            closed, opened = And(*[c for c, _ in kids]), And(*[o for _, o in kids])
        else:
            (c0, o0), rest = kids[0], kids[1:]
            closed = And(c0, *[negate(o) for _, o in rest])
            opened = And(o0, *[negate(c) for c, _ in rest])
    T = _placement(spec, node)
    if T != Matrix.eye(4):
        local = inv(T) * Matrix([X, Y, Z, 1])
        closed, opened = at_point(closed, local[:3]), at_point(opened, local[:3])
    return closed, opened


def _closed_rel(r):
    e = r.lhs - r.rhs
    if isinstance(r, (Ge, Gt)):
        return ge(e)
    if isinstance(r, (Le, Lt)):
        return ge(-e)
    if isinstance(r, Eq):
        return Eq(_tidy(e), 0)
    raise SpecError(f"unsupported relation {r}")


def _open_rel(r):
    e = r.lhs - r.rhs
    if isinstance(r, (Ge, Gt)):
        return gt(e)
    if isinstance(r, (Le, Lt)):
        return gt(-e)
    return false          # an equation has no interior


def body_predicate(spec, body):
    return predicates(spec, body.shape)[0]


# ---------- convex pieces ----------

@dataclass
class Polytope:
    verts: list      # 3x1 matrices
    faces: list      # (outward normal n, offset d): the solid is n . p <= d
    edges: list      # edge directions, one per parallel class

    def moved(self, T, red=None):
        R, t = T[:3, :3], T[:3, 3]
        f = red or expand
        mv = lambda v: (R * v).applyfunc(f)
        return Polytope([(R * v + t).applyfunc(f) for v in self.verts],
                        [(mv(n), f(d + (R * n).dot(t))) for n, d in self.faces],
                        [mv(e) for e in self.edges])

    def interior(self):
        return And(*[gt(d - n.dot(P)) for n, d in self.faces])


@dataclass
class Ball:
    centre: object
    r: object

    def moved(self, T, red=None):
        f = red or expand
        return Ball((T[:3, :3] * self.centre + T[:3, 3]).applyfunc(f), self.r)

    def interior(self):
        return gt(self.r**2 - (P - self.centre).dot(P - self.centre))


def _canon_dir(v):
    for c in v:
        if c != 0:
            return tuple(expand(x / c) for x in v)
    return None


def _hull(spec, pts):
    if not isinstance(pts, list) or len(pts) < 4:
        raise SpecError("hull expects a list of at least four [x, y, z] vertices")
    V = [Matrix(spec.vec(p)) for p in pts]
    if not all(c.is_number for v in V for c in v):
        raise SpecError("hull vertices must be numbers (no symbolic parameters)")
    from scipy.spatial import ConvexHull
    try:
        H = ConvexHull(np.array([[float(c) for c in v] for v in V]))
    except Exception as err:
        raise SpecError(f"hull is degenerate (flat or repeated points): {err}")
    verts = [V[i] for i in H.vertices]
    centre = sum(verts, zeros(3, 1)) / len(verts)
    faces, planes, edges, dirs = [], set(), [], set()
    for tri in H.simplices:
        a, b, c = (V[i] for i in tri)
        n = (b - a).cross(c - a).applyfunc(expand)
        d = expand(n.dot(a))
        if (n.dot(centre) - d).is_positive:
            n, d = -n, -d
        key = _canon_dir(list(n) + [d])
        if key not in planes:
            planes.add(key)
            faces.append((n, d))
        for u, w in ((a, b), (b, c), (c, a)):
            k = _canon_dir(w - u)
            if k not in dirs:
                dirs.add(k)
                edges.append(w - u)
    return Polytope(verts, faces, edges)


def pieces(spec, node, T=None):
    """The node as a list of convex pieces (Polytope, Ball), or None if it is not a union of them."""
    T = (T if T is not None else Matrix.eye(4)) * _placement(spec, node)
    kind = _kind(spec, node)
    v = node[kind]
    if kind == "box":
        h = _box(spec, v)
        E = Matrix.eye(3)
        verts = [Matrix([sx * h[0], sy * h[1], sz * h[2]]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        faces = [(sg * E[:, i], h[i]) for i in range(3) for sg in (1, -1)]
        return [Polytope(verts, faces, [E[:, i] for i in range(3)]).moved(T)]
    if kind == "hull":
        return [_hull(spec, v).moved(T)]
    if kind == "ball":
        return [Ball(zeros(3, 1), spec.num(v)).moved(T)]
    if kind == "union":
        out = [pieces(spec, k, T) for k in v]
        return None if any(o is None for o in out) else [p for o in out for p in o]
    return None


def _separated(A, B, red):
    """Interiors of convex pieces A, B disjoint: exact, quantifier-free when possible."""
    if isinstance(A, Ball) and isinstance(B, Ball):
        d = A.centre - B.centre
        return ge(d.dot(d) - (A.r + B.r)**2, red), True
    if isinstance(A, Polytope) and isinstance(B, Polytope):
        alts = [And(*[ge(n.dot(w) - d, red) for w in B.verts]) for n, d in A.faces]
        alts += [And(*[ge(n.dot(v) - d, red) for v in A.verts]) for n, d in B.faces]
        for e in A.edges:
            for f in B.edges:
                n = e.cross(f).applyfunc(red)
                guard = gt(n.dot(n), red)
                if guard is false:
                    continue
                diffs = [n.dot(w - v) for v in A.verts for w in B.verts]
                alts.append(And(guard, *[ge(x, red) for x in diffs]))
                alts.append(And(guard, *[ge(-x, red) for x in diffs]))
        return Or(*alts), True
    both = And(A.interior(), B.interior())
    return ForAll((X, Y, Z), map_atoms(negate(both), lambda r: r.func(_tidy(r.lhs - r.rhs, red), 0))), False


def separation(spec, a, b, poses, red=None):
    """Condition that the interiors of bodies a and b are disjoint, in a's coordinates.

    Returns (formula, quantifier_free)."""
    for n in (a, b):
        if n not in spec.bodies or spec.bodies[n].shape is None:
            raise SpecError(f"body {n!r} has no shape")
    f = red or expand
    M = (inv(poses[a]) * poses[b]).applyfunc(f)          # pose of b in a's coordinates
    pa, pb = pieces(spec, spec.bodies[a].shape), pieces(spec, spec.bodies[b].shape)
    if pa is not None and pb is not None:
        pb = [q.moved(M, red) for q in pb]
        parts, exact = zip(*[_separated(p, q, red) for p in pa for q in pb])
        return And(*parts), all(exact)
    _, open_a = predicates(spec, spec.bodies[a].shape)
    _, open_b = predicates(spec, spec.bodies[b].shape)
    local = (inv(M) * Matrix([X, Y, Z, 1])).applyfunc(f)
    return ForAll((X, Y, Z), negate(And(open_a, at_point(open_b, local[:3], red)))), False


def no_overlap(spec, c, poses, red):
    """The no_overlap constraint c. Returns (formula, note)."""
    if not isinstance(c.args, list) or len(c.args) != 2:
        raise SpecError(f"no_overlap {c.name!r}: expected [body, body]")
    a, b = (str(n) for n in c.args)
    try:
        f, qf = separation(spec, a, b, poses, red)
    except SpecError as err:
        raise SpecError(f"no_overlap {c.name!r}: {err}")
    note = None if qf else (f"no_overlap {c.name!r}: part of the condition is quantified "
                            "(for all points x, y, z) because the shapes are not all boxes/hulls, "
                            "or a ball meets a polytope")
    return f, note
