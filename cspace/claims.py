"""Claims about a mechanism, written as formulas, turned into exact SymPy logic.

A claim is `always: <formula>` (true in every configuration) or `possible: <formula>` (true
in some configuration). Formulas use Python-like syntax, read with `ast` and never executed:

  numbers, parameters, + - * / ** ^, comparisons (chains allowed), and / or / not
  slideX.d, elbow.c, elbow.s, ball.q0     a joint's coordinates (see the compiler's roles)
  pos(body) / pos(body.frame)             world position of a body's or frame's origin
  point(body, x, y, z)                    world position of a point given in body coordinates
  axis(body.frame)                        world direction of a frame's z axis
  [x, y, z]  p.x  dot(u, v)  cross(u, v)  norm2(u)  dist2(u, v)
  inside(p, body)  interior(p, body)      point in the closed solid / its interior
  overlap(A, B)                           interiors intersect
  meet(A, B)                              closed solids touch or intersect
  convex(A, B, ...)                       the union of the solids is convex
  forall(p, F)  exists(p, F)              p is a point (p.x, p.y, p.z)
  forall_real(t, F)  exists_real(t, F)    t is a number
  implies(F, G)  iff(F, G)
"""
import ast
from itertools import count

from sympy import (And, Equivalent, Implies, Integer, Matrix, Not, Or, Rational, Symbol, cos,
                   eye, false, pi, sin, sqrt, true)
from sympy.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational

from .logic import Exists, ForAll, map_relations
from .shapes import at_point, predicates, separation, _tidy
from .spec import SpecError

FUNCTIONS = {"pos", "point", "axis", "dot", "cross", "norm2", "dist2", "inside", "interior",
             "overlap", "meet", "convex", "forall", "exists", "forall_real", "exists_real",
             "implies", "iff", "sqrt", "cos", "sin"}
BINDERS = {"forall", "exists", "forall_real", "exists_real"}


class ClaimError(SpecError):
    pass


def parse(text):
    try:
        return ast.parse(str(text).replace("^", "**"), mode="eval").body
    except SyntaxError as err:
        raise ClaimError(f"cannot read claim {text!r}: {err.msg}")


def mentioned(tree):
    """Identifiers a claim uses that are not functions or variables it binds, in order."""
    out = {}

    def walk(node, bound):
        if isinstance(node, ast.Name):
            if node.id not in bound:
                out.setdefault(node.id)
        elif isinstance(node, ast.Call):
            fn = node.func.id if isinstance(node.func, ast.Name) else None
            args = node.args
            if fn in BINDERS and args and isinstance(args[0], ast.Name):
                for a in args[1:]:
                    walk(a, bound | {args[0].id})
                return
            for a in args:
                walk(a, bound)
            if fn is None:
                walk(node.func, bound)
        else:
            for child in ast.iter_child_nodes(node):
                walk(child, bound)

    walk(tree, frozenset())
    return list(out)


class _Ref:
    """A name: a body (maybe with a frame), a joint, or both when they share the name."""

    def __init__(self, body=None, frame=None, joint=None):
        self.body, self.frame, self.joint = body, frame, joint


class Evaluator:
    """Evaluate a claim against a compiled System (usually of a part of the mechanism)."""

    def __init__(self, spec, system):
        self.spec, self.system = spec, system
        self.joints = {j.name: j for j in system.joints}
        self.red = system.red
        self.bound = {}
        self.fresh = count(1)
        self._preds = {}

    # ----- helpers -----
    def fail(self, msg):
        raise ClaimError(msg)

    def pose(self, body):
        if body == "world":
            return eye(4)
        if body not in self.system.poses:
            self.fail(f"body {body!r} is not part of this mechanism")
        return self.system.poses[body]

    def frame_pose(self, ref):
        T = self.pose(ref.body)
        if ref.frame is None:
            return T
        frames = self.spec.bodies[ref.body].frames if ref.body in self.spec.bodies else {}
        if ref.frame not in frames:
            self.fail(f"body {ref.body!r} has no frame {ref.frame!r}")
        return T * frames[ref.frame]

    def shape_preds(self, body):
        if body not in self._preds:
            b = self.spec.bodies.get(body)
            if b is None or b.shape is None:
                self.fail(f"body {body!r} has no shape")
            self._preds[body] = predicates(self.spec, b.shape)
        return self._preds[body]

    def in_body(self, p, body, which):
        T = self.pose(body)
        R, t = T[:3, :3], T[:3, 3]
        local = (R.T * (p - t)).applyfunc(self.red)
        return at_point(self.shape_preds(body)[which], list(local), self.red)

    def vec(self, v, what):
        if not (isinstance(v, Matrix) and v.shape == (3, 1)):
            self.fail(f"{what} needs a point or vector")
        return v

    def scalar(self, v, what):
        if isinstance(v, (Matrix, _Ref)) or isinstance(v, (Relational, And, Or, Not)):
            self.fail(f"{what} needs a number")
        return v

    def body_name(self, v, what):
        if not isinstance(v, _Ref) or v.body is None or v.frame is not None:
            self.fail(f"{what} needs a body name")
        return v.body

    def point_vars(self, name):
        return Matrix([Symbol(f"{name}.{c}", real=True) for c in "xyz"])

    # ----- evaluation -----
    def formula(self, tree):
        f = self.ev(tree)
        if f is True or f is False:
            f = true if f else false
        if not (f in (true, false) or isinstance(f, (Relational, And, Or, Not, Implies, Equivalent,
                                                    ForAll, Exists))):
            self.fail("a claim must be a statement (a comparison or a logical formula), not a value")
        return map_relations(f, lambda r: r.func(_tidy(r.lhs - r.rhs, self.red), 0))

    def ev(self, n):
        m = getattr(self, "ev_" + type(n).__name__, None)
        if m is None:
            self.fail(f"unsupported syntax: {ast.dump(n)[:60]}")
        return m(n)

    def ev_Constant(self, n):
        if isinstance(n.value, bool):
            return true if n.value else false
        if isinstance(n.value, int):
            return Integer(n.value)
        if isinstance(n.value, float):
            return Rational(repr(n.value))
        self.fail(f"unsupported constant {n.value!r}")

    def ev_Name(self, n):
        name = n.id
        if name in self.bound:
            return self.bound[name]
        if name == "pi":
            return pi
        if name in self.spec.env:
            return self.spec.env[name]
        body = name if (name == "world" or name in self.spec.bodies) else None
        if body or name in self.joints:
            return _Ref(body, None, self.joints.get(name))
        self.fail(f"unknown name {name!r} (not a parameter, body, joint or bound variable)")

    def ev_Attribute(self, n):
        v = self.ev(n.value)
        if isinstance(v, _Ref) and v.frame is None:
            if v.joint is not None and n.attr in v.joint.values:
                return v.joint.values[n.attr]
            if v.body is not None:
                return _Ref(v.body, n.attr)
            self.fail(f"joint {v.joint.name!r} has coordinates {sorted(v.joint.values)}, not {n.attr!r}")
        if isinstance(v, Matrix) and n.attr in ("x", "y", "z"):
            return v["xyz".index(n.attr)]
        self.fail(f"cannot take .{n.attr} here")

    def ev_List(self, n):
        if len(n.elts) != 3:
            self.fail("a point is written [x, y, z]")
        return Matrix([self.scalar(self.ev(e), "a coordinate") for e in n.elts])

    ev_Tuple = ev_List

    def ev_UnaryOp(self, n):
        v = self.ev(n.operand)
        if isinstance(n.op, ast.Not):
            return Not(v)
        if isinstance(n.op, ast.USub):
            return -v
        if isinstance(n.op, ast.UAdd):
            return v
        self.fail("unsupported operator")

    def ev_BinOp(self, n):
        a, b = self.ev(n.left), self.ev(n.right)
        op = n.op
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, ast.Div):
            b = self.scalar(b, "division")
            if not b.is_number or b == 0:
                self.fail("division is only allowed by nonzero numbers; multiply the comparison out")
            return a / b
        if isinstance(op, ast.Pow):
            if not (getattr(b, "is_Integer", False) and b >= 0):
                self.fail("powers must be whole numbers")
            return self.scalar(a, "a power") ** b
        self.fail("unsupported operator")

    def ev_BoolOp(self, n):
        vals = [self.ev(v) for v in n.values]
        return And(*vals) if isinstance(n.op, ast.And) else Or(*vals)

    def ev_Compare(self, n):
        ops = {ast.Eq: Eq, ast.NotEq: Ne, ast.Lt: Lt, ast.LtE: Le, ast.Gt: Gt, ast.GtE: Ge}
        terms = [self.ev(n.left)] + [self.ev(c) for c in n.comparators]
        parts = []
        for (a, b), op in zip(zip(terms, terms[1:]), n.ops):
            rel = ops.get(type(op))
            if rel is None:
                self.fail("unsupported comparison")
            if isinstance(a, Matrix) or isinstance(b, Matrix):
                a, b = self.vec(a, "comparing points"), self.vec(b, "comparing points")
                if rel is Eq:
                    parts.append(And(*[Eq(x, y) for x, y in zip(a, b)]))
                elif rel is Ne:
                    parts.append(Or(*[Ne(x, y) for x, y in zip(a, b)]))
                else:
                    self.fail("points can only be compared with == or !=")
            else:
                parts.append(rel(self.scalar(a, "a comparison"), self.scalar(b, "a comparison")))
        return And(*parts)

    def ev_Call(self, n):
        if not isinstance(n.func, ast.Name) or n.func.id not in FUNCTIONS:
            self.fail(f"unknown function {ast.unparse(n.func)!r}; known: {sorted(FUNCTIONS)}")
        fn = n.func.id
        if n.keywords:
            self.fail(f"{fn}() takes no keyword arguments")
        if fn in BINDERS:
            return self.binder(fn, n.args)
        args = [self.ev(a) for a in n.args]
        return getattr(self, "fn_" + fn)(*args) if self._arity_ok(fn, args) else None

    def _arity_ok(self, fn, args):
        need = {"pos": 1, "axis": 1, "dot": 2, "cross": 2, "norm2": 1, "dist2": 2, "inside": 2,
                "interior": 2, "overlap": 2, "meet": 2, "implies": 2, "sqrt": 1, "cos": 1, "sin": 1}
        if fn in need and len(args) != need[fn]:
            self.fail(f"{fn}() takes {need[fn]} argument(s), got {len(args)}")
        return True

    def binder(self, fn, args):
        if len(args) != 2 or not isinstance(args[0], ast.Name):
            self.fail(f"{fn}(variable, formula)")
        name = args[0].id
        if name in self.spec.env or name in self.joints or name in self.spec.bodies:
            self.fail(f"bound variable {name!r} clashes with a parameter, joint or body")
        if fn.endswith("_real"):
            v = Symbol(name, real=True)
            if any(v.name == s.name for s in self.system.variables):
                self.fail(f"bound variable {name!r} clashes with a coordinate")
            syms = [v]
        else:
            v = self.point_vars(name)
            syms = list(v)
        saved = self.bound.get(name)
        self.bound[name] = v
        try:
            body = self.ev(args[1])
        finally:
            if saved is None:
                self.bound.pop(name)
            else:
                self.bound[name] = saved
        return (ForAll if fn.startswith("forall") else Exists)(syms, body)

    # ----- functions -----
    def fn_pos(self, ref):
        if not isinstance(ref, _Ref) or ref.body is None:
            self.fail("pos() needs a body or body.frame")
        return self.frame_pose(ref)[:3, 3]

    def fn_point(self, body, *xyz):
        body = self.body_name(body, "point()")
        if len(xyz) == 1:
            local = self.vec(xyz[0], "point()")
        elif len(xyz) == 3:
            local = Matrix(xyz)
        else:
            self.fail("point(body, x, y, z) or point(body, [x, y, z])")
        return (self.pose(body) * Matrix([*local, 1]))[:3, 0].applyfunc(self.red)

    def fn_axis(self, ref):
        if not isinstance(ref, _Ref) or ref.body is None:
            self.fail("axis() needs a body or body.frame")
        return self.frame_pose(ref)[:3, 2]

    def fn_dot(self, u, v):
        return self.vec(u, "dot()").dot(self.vec(v, "dot()"))

    def fn_cross(self, u, v):
        return self.vec(u, "cross()").cross(self.vec(v, "cross()"))

    def fn_norm2(self, u):
        return self.vec(u, "norm2()").dot(u)

    def fn_dist2(self, u, v):
        d = self.vec(u, "dist2()") - self.vec(v, "dist2()")
        return d.dot(d)

    def fn_inside(self, p, body):
        return self.in_body(self.vec(p, "inside()"), self.body_name(body, "inside()"), 0)

    def fn_interior(self, p, body):
        return self.in_body(self.vec(p, "interior()"), self.body_name(body, "interior()"), 1)

    def fn_overlap(self, a, b):
        a, b = self.body_name(a, "overlap()"), self.body_name(b, "overlap()")
        self.shape_preds(a), self.shape_preds(b)
        apart, _ = separation(self.spec, a, b, self._poses(), self.red)
        return Not(apart)

    def fn_meet(self, a, b):
        a, b = self.body_name(a, "meet()"), self.body_name(b, "meet()")
        p = self.point_vars(f"meet{next(self.fresh)}")
        return Exists(list(p), And(self.in_body(p, a, 0), self.in_body(p, b, 0)))

    def fn_convex(self, *bodies):
        if not bodies:
            self.fail("convex() needs at least one body")
        names = [self.body_name(b, "convex()") for b in bodies]
        k = next(self.fresh)
        p, q = self.point_vars(f"convex{k}p"), self.point_vars(f"convex{k}q")
        t = Symbol(f"convex{k}t", real=True)

        def in_union(x):
            return Or(*[self.in_body(x, b, 0) for b in names])

        mid = (1 - t) * p + t * q
        return ForAll([*p, *q, t], Implies(And(in_union(p), in_union(q), Ge(t, 0), Le(t, 1)),
                                           in_union(mid)))

    def fn_implies(self, a, b):
        return Implies(a, b)

    def fn_iff(self, *args):
        if len(args) < 2:
            self.fail("iff() takes two or more formulas")
        return Equivalent(*args)

    def _numeric(self, fn, x):
        if not getattr(x, "is_number", False):
            self.fail(f"{fn}() only applies to numbers")
        return x

    def fn_sqrt(self, x):
        return sqrt(self._numeric("sqrt", x))

    def fn_cos(self, x):
        return cos(self._numeric("cos", x))

    def fn_sin(self, x):
        return sin(self._numeric("sin", x))

    def _poses(self):
        P = dict(self.system.poses)
        P["world"] = eye(4)
        return P
