"""Shared test helpers: building examples, z3 translation, numeric poses."""
from pathlib import Path

import numpy as np
import z3
from sympy import (Add, And, Eq, Ge, Gt, Le, Lt, Mul, Ne, Or, Pow, Symbol, false, lambdify,
                   true)

from cspace import build
from cspace.shapes import ForAll

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def example(name, **kw):
    return build(EXAMPLES / f"{name}.yaml", **kw)


def to_z3(e, env):
    """SymPy polynomial formula -> z3 (rational coefficients only)."""
    if e is true:
        return z3.BoolVal(True)
    if e is false:
        return z3.BoolVal(False)
    if isinstance(e, Symbol):
        if e.name not in env:
            env[e.name] = z3.Real(e.name)
        return env[e.name]
    if e.is_Rational:
        return z3.RealVal(f"{e.p}/{e.q}")
    if isinstance(e, Add):
        return z3.Sum([to_z3(a, env) for a in e.args])
    if isinstance(e, Mul):
        out = to_z3(e.args[0], env)
        for a in e.args[1:]:
            out = out * to_z3(a, env)
        return out
    if isinstance(e, Pow) and e.exp.is_Integer and e.exp > 0:
        b = to_z3(e.base, env)
        out = b
        for _ in range(int(e.exp) - 1):
            out = out * b
        return out
    if isinstance(e, And):
        return z3.And([to_z3(a, env) for a in e.args])
    if isinstance(e, Or):
        return z3.Or([to_z3(a, env) for a in e.args])
    ops = {Ge: lambda a, b: a >= b, Gt: lambda a, b: a > b, Le: lambda a, b: a <= b,
           Lt: lambda a, b: a < b, Eq: lambda a, b: a == b, Ne: lambda a, b: a != b}
    for cls, op in ops.items():
        if isinstance(e, cls):
            return op(to_z3(e.lhs, env), to_z3(e.rhs, env))
    if isinstance(e, ForAll):
        bound = [to_z3(v, env) for v in e.args[0]]
        return z3.ForAll(bound, to_z3(e.args[1], env))
    raise TypeError(f"cannot translate {e!r} ({type(e).__name__})")


def numeric_poses(system, point):
    """Body poses (4x4 float arrays) at a numeric point of the system's variables."""
    out = {}
    for body, M in system.poses.items():
        f = lambdify(system.variables, M, "numpy")
        out[body] = np.array(f(*point), dtype=float)
    return out


def quaternion(R):
    """Unit quaternion (w, x, y, z) of a rotation matrix (Shepperd's method)."""
    t = np.trace(R)
    cands = [t, R[0, 0], R[1, 1], R[2, 2]]
    i = int(np.argmax(cands))
    if i == 0:
        w = np.sqrt(1 + t) / 2
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    j, k = (i % 3) + 1, ((i + 1) % 3) + 1
    a = i - 1
    b, c = j - 1, k - 1
    v = np.zeros(4)
    v[i] = np.sqrt(1 + 2 * R[a, a] - t) / 2
    v[0] = (R[c, b] - R[b, c]) / (4 * v[i])
    v[j] = (R[b, a] + R[a, b]) / (4 * v[i])
    v[k] = (R[c, a] + R[a, c]) / (4 * v[i])
    return v
