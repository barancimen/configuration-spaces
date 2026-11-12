"""Milestone 3: CSG predicates and exact no-overlap conditions."""
import numpy as np
import pytest
import z3
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from sympy import And, Ge, Gt, Or, Symbol, lambdify, sqrt

from cspace import build
from cspace.logic import ForAll
from cspace.shapes import X, Y, Z, negate
from helpers import example, to_z3


def sym(name):
    return Symbol(name, real=True)


def overlap_lp(H1, H2):
    """Interiors of {A x <= b} and {C x <= d} meet?  Returns the largest common inner margin."""
    A = np.vstack([H1[0], H2[0]])
    b = np.concatenate([H1[1], H2[1]])
    norms = np.linalg.norm(A, axis=1)
    res = linprog(c=[0, 0, 0, -1], A_ub=np.hstack([A, norms[:, None]]), b_ub=b,
                  bounds=[(None, None)] * 3 + [(None, 1)], method="highs")
    return -res.fun


def box_h(half, R=np.eye(3), t=np.zeros(3), centre=np.zeros(3)):
    A = np.vstack([np.eye(3), -np.eye(3)])
    b = np.concatenate([half + centre, half - centre])
    return A @ R.T, b + A @ R.T @ t


def solver_says_overlap(formula, values):
    """z3 on the negation of a ForAll no-overlap formula: sat means some point is in both interiors."""
    assert isinstance(formula, ForAll)
    env = {}
    body = to_z3(formula.args[1].subs(values), env)
    s = z3.Solver()
    s.set("timeout", 20000)
    s.add(z3.Not(body))
    r = s.check()
    assert r != z3.unknown
    return r == z3.sat


def test_sliding_boxes_condition_is_exactly_two_apart():
    s = example("sliding_boxes")
    apart = next(f for f in s.formulas if not f.has(ForAll))
    env = {}
    d = z3.Real("dS")
    env["dS"] = d
    solver = z3.Solver()
    solver.add(z3.Not(to_z3(apart, env) == z3.Or(d >= 2, d <= -2)))
    assert solver.check() == z3.unsat


@pytest.mark.parametrize("which", ["ball_vs_box", "cylinder_vs_box"])
def test_quantified_fallback_agrees_with_geometry(which, tmp_path):
    if which == "ball_vs_box":
        s = example("sliding_boxes")
        f = next(f for f in s.formulas if f.has(ForAll))
        var = sym("dB")
    else:
        p = tmp_path / "m.yaml"
        p.write_text("""
bodies:
  world: {frames: {rail: {at: [0, 5/4, 0], z: [1, 0, 0]}}}
  cube: {shape: {box: [2, 2, 2]}}
  peg: {shape: {cylinder: [1/2, 2]}, frames: {rail: {z: [1, 0, 0]}}}
constraints:
  - fixed: [world, cube]
  - prismatic: [world.rail, peg.rail]
    name: p
  - no_overlap: [cube, peg]
""")
        s = build(p)
        (f,) = s.formulas
        var = sym("dP")
    assert isinstance(f, ForAll)
    limit = 1 + float(sqrt(3)) / 4          # overlap exactly when |d| < 1 + sqrt(3)/4
    from sympy import Rational
    for d in ["0", "1", "7/5", "3/2", "2", "-7/5", "-3/2", "-3"]:
        v = Rational(d)
        assert solver_says_overlap(f, {var: v}) == (abs(float(v)) < limit), d


def test_box_vs_hull_formula_agrees_with_linear_programming(tmp_path):
    pts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1 / 2]]
    p = tmp_path / "m.yaml"
    p.write_text(f"""
bodies:
  block: {{shape: {{box: [2, 1, 1]}}}}
  gem: {{shape: {{hull: {pts}}}}}
constraints:
  - fixed: [world, block]
  - no_overlap: [block, gem]
""")
    s = build(p)
    (f,) = s.formulas
    names = ["xGem", "yGem", "zGem", "q0Gem", "q1Gem", "q2Gem", "q3Gem"]
    assert [v.name for v in s.variables] == names
    F = lambdify(s.variables, f, "numpy")
    hull = ConvexHull(np.array(pts, dtype=float))
    A_gem, b_gem = hull.equations[:, :3], -hull.equations[:, 3]
    block = box_h(np.array([1, 0.5, 0.5]))
    rng = np.random.default_rng(7)
    agree = overlaps = checked = 0
    while checked < 200:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        t = rng.uniform(-2, 2, 3)
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        margin = overlap_lp(block, (A_gem @ R.T, b_gem + A_gem @ R.T @ t))
        if abs(margin) < 1e-6:
            continue                      # touching: either answer is within rounding
        checked += 1
        overlaps += margin > 0
        agree += bool(F(*t, *q)) == (margin <= 0)
    assert agree == checked
    assert 20 < overlaps < 180, "the sample should contain both outcomes"


def test_door_stop_formula_agrees_with_linear_programming():
    s = example("door_stop")
    (f,) = s.formulas
    F = lambdify(s.variables, f, "numpy")
    stop = box_h(np.array([0.25, 0.25, 1.0]), centre=np.array([1.0, 1.0, 0.0]))
    free = blocked = 0
    for th in np.linspace(-np.pi, np.pi, 721)[:-1]:
        R = Rotation.from_euler("z", th).as_matrix()
        door = box_h(np.array([1.0, 0.05, 1.0]), R=R, centre=np.array([1.0, 0.0, 0.0]))
        margin = overlap_lp(door, stop)
        if abs(margin) < 1e-6:
            continue
        ok = bool(F(np.cos(th), np.sin(th)))
        assert ok == (margin <= 0), th
        free += ok
        blocked += not ok
    assert free and blocked


def test_csg_predicates_at_points(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("""
bodies:
  part:
    shape:
      union:
        - difference:
            - box: [2, 2, 2]
            - {ball: 1/2, at: [1, 0, 0]}
        - {cylinder: [1/4, 1], at: [0, 0, 3/2]}
        - where: ["x^2 + y^2 + z^2 <= 1", "z >= 0"]
          at: [5, 0, 0]
""")
    s = build(p)
    pred = s.shapes["part"]
    inside = [(0, 0, 0), (0, 0.9, 0.9), (0, 0, 1.9), (5, 0, 0.5), (0.4, 0, 0)]
    outside = [(0.9, 0, 0), (0, 0, 2.1), (0.3, 0, 1.5), (5, 0, -0.1), (3, 0, 0)]
    for pt in inside:
        assert bool(pred.subs(dict(zip((X, Y, Z), pt)))), pt
    for pt in outside:
        assert not bool(pred.subs(dict(zip((X, Y, Z), pt)))), pt


def test_negate_is_logical_negation():
    a, b = sym("a"), sym("b")
    f = Or(And(Ge(a, 0), Gt(b - a, 1)), And(Ge(-b, 0), Gt(a * b, 2)))
    env = {}
    s = z3.Solver()
    s.add(z3.Not(to_z3(negate(f), env) == z3.Not(to_z3(f, env))))
    assert s.check() == z3.unsat
    assert "Not" not in str(negate(f))
