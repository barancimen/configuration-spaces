"""Diagnostics only (numerical): find real points of the space and measure its local dimension.

The local dimension at a point is the number of variables minus the rank of the Jacobian of the
equations there. It is exact at smooth points, which random sampling finds almost surely.
"""
from collections import Counter

import numpy as np
from scipy.optimize import least_squares
from sympy import Matrix, Rational, lambdify


def parameter_values(system, rng):
    """Random exact values in [1, 3] for symbolic parameters, satisfying the assumptions."""
    for _ in range(500):
        vals = {p: Rational(int(rng.integers(100, 301)), 100) for p in system.params}
        if all(bool(a.subs(vals)) for a in system.assumptions):
            return vals
    raise ValueError("could not pick parameter values that satisfy the assumptions")


def sample(system, tries=30, seed=0, values=None, tol=1e-9, starts=None):
    """Return (parameter values, [(point, local dimension), ...])."""
    rng = np.random.default_rng(seed)
    vals = values if values is not None else parameter_values(system, rng)
    X = system.variables
    eqs = [e.subs(vals) for e in system.equations]
    ineqs = [g.subs(vals) for g in system.inequalities]
    F = lambdify(X, eqs, "numpy")
    G = lambdify(X, ineqs, "numpy")
    J = lambdify(X, Matrix(eqs).jacobian(X) if eqs else Matrix.zeros(1, len(X)), "numpy")
    JG = lambdify(X, Matrix(ineqs).jacobian(X) if ineqs else Matrix.zeros(1, len(X)), "numpy")

    def resid(x):
        r = np.atleast_1d(np.array(F(*x), dtype=float))
        g = np.atleast_1d(np.array(G(*x), dtype=float)) if ineqs else np.zeros(0)
        return np.concatenate([r, np.minimum(g, 0.0)])

    def jac(x):                                     # exact, so SciPy need not difference
        rows = [np.array(J(*x), dtype=float).reshape(-1, len(X))[:len(eqs)]]
        if ineqs:
            g = np.atleast_1d(np.array(G(*x), dtype=float))
            JGx = np.array(JG(*x), dtype=float).reshape(-1, len(X))
            rows.append(JGx * (g < 0)[:, None])
        return np.vstack(rows)

    points = []
    for k in range(tries):
        x0 = starts[k] if starts is not None else rng.uniform(-2, 2, len(X))
        sol = least_squares(resid, x0, jac=jac if eqs or ineqs else "2-point",
                            xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        if np.max(np.abs(resid(sol.x)), initial=0.0) < tol:
            sv = np.linalg.svd(np.array(J(*sol.x), dtype=float), compute_uv=False)
            rank = int((sv > 1e-7 * max(1.0, sv[0])).sum())
            points.append((sol.x, len(X) - rank))
    return vals, points


def report(system, tries=30, seed=0):
    lines = [f"check: naive freedom count {system.naive_dof} "
             "(6 per body minus each constraint's codimension, minus 1 per coupling)"]
    try:
        vals, points = sample(system, tries, seed)
    except ValueError as err:
        return "\n".join(lines + [f"check: {err}"])
    if vals:
        lines.append("check: parameters set to " + ", ".join(f"{p} = {v}" for p, v in vals.items()))
    if not points:
        lines.append(f"check: no real point found in {tries} tries; the space may be empty")
    else:
        dims = Counter(d for _, d in points)
        lines.append(f"check: {len(points)} points found in {tries} tries; local dimension "
                     + ", ".join(f"{d} at {n}" for d, n in sorted(dims.items())))
    if system.formulas:
        lines.append("check: the no_overlap conditions were not used when sampling")
    return "\n".join(lines)
