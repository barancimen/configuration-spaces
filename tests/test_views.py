"""Milestone 2: couplings, maximal coordinates, the trig view, the dimension check."""
import numpy as np
import pytest
from sympy import Symbol, cos, expand, lambdify, pi, sin

from cspace import build
from cspace.check import sample
from helpers import example, numeric_poses, quaternion


def sym(name):
    return Symbol(name, real=True)


@pytest.mark.parametrize("name, naive, dim, overrides", [
    ("bennett", -2, 1, None),
    ("tethered_puck", 2, 2, None),
    ("rssr", 2, 2, None),
    ("fourbar", -2, 1, {"a": "2", "b": "5", "c": "4", "d": "6"}),
    ("gears", 1, 1, None),
])
def test_dimension(name, naive, dim, overrides):
    s = example(name, overrides=overrides)
    assert s.naive_dof == naive
    _, points = sample(s, tries=8, seed=1)
    assert points, "no real point found"
    assert {d for _, d in points} == {dim}


@pytest.mark.parametrize("name, overrides", [
    ("fourbar", {"a": "2", "b": "5", "c": "4", "d": "6"}),
    ("bennett", None),
    ("rssr", None),
])
def test_reduced_and_maximal_views_agree(name, overrides):
    red = example(name, overrides=overrides)
    mx = example(name, overrides=overrides, coords="maximal")
    assert all(j.kind == "free" for j in mx.joints if j.tree)
    _, points = sample(red, tries=4, seed=2)
    assert points
    for x, _ in points:
        poses = numeric_poses(red, x)
        values = {}
        for j in mx.joints:
            if j.kind == "free" and j.tree:
                T = poses[j.b]
                q = quaternion(T[:3, :3])
                for role, v in zip(["x", "y", "z", "q0", "q1", "q2", "q3"], list(T[:3, 3]) + list(q)):
                    values[j.values[role]] = v
        pt = [values[v] for v in mx.variables]
        F = lambdify(mx.variables, mx.equations, "numpy")
        assert np.max(np.abs(F(*pt))) < 1e-8
        if mx.inequalities:
            G = lambdify(mx.variables, mx.inequalities, "numpy")
            assert np.min(G(*pt)) > -1e-8


def test_trig_view_is_the_substituted_polynomial_view():
    poly = example("slider_crank")
    trig = example("slider_crank", trig=True)
    thc, thw = sym("thCrank"), sym("thWrist")
    assert trig.variables == [thc, thw, sym("dSlide")]
    subs = {sym("cCrank"): cos(thc), sym("sCrank"): sin(thc), sym("cWrist"): cos(thw), sym("sWrist"): sin(thw)}
    circles = {e for e in poly.equations if e.func.__name__ == "Add" and len(e.free_symbols) == 2
               and all(v.name[0] in "cs" for v in e.free_symbols) and e.subs(subs).simplify() == 0}
    expected = {expand(e.subs(subs)) for e in poly.equations if e not in circles}
    assert set(trig.equations) == expected


def test_trig_couplings_are_linear_in_angles():
    g = example("gears", trig=True)
    assert g.equations == [sym("thP") + 2 * sym("thW")]
    s = example("screw", trig=True)
    pitch, th, d = sym("pitch"), sym("thDrive"), sym("dFeed")
    assert s.equations == [2 * d - pitch * th / pi]
    assert s.inequalities == [th, 10 * pi - th]


def test_couple_lengths_and_negative_angle_coefficients(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("""
bodies: {a: {}, b: {}, c: {}, e: {}}
constraints:
  - prismatic: [world, a]
    name: p1
  - prismatic: [world, b]
    name: p2
  - couple: {p1: 1, p2: -2}
    equals: 1
  - revolute: [world, c]
    name: h1
  - revolute: [world, e]
    name: h2
  - couple: {h1: 1, h2: -1}
""")
    s = build(p)
    eqs = set(s.equations)
    d1, d2 = sym("dP1"), sym("dP2")
    assert d1 - 2 * d2 - 1 in eqs or -(d1 - 2 * d2 - 1) in eqs
    c1, s1, c2, s2 = sym("cH1"), sym("sH1"), sym("cH2"), sym("sH2")
    # th1 = th2  ->  z1 = z2
    assert {c1 - c2, s1 - s2} <= {e if not e.could_extract_minus_sign() else -e for e in eqs} | eqs


def test_range_on_a_closing_joint(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("""
parameters: {a: 2, b: 5, c: 4, d: 6}
bodies:
  world: {frames: {o: {}, p: {at: [d, 0, 0]}}}
  crank: {frames: {base: {}, tip: {at: [a, 0, 0]}}}
  coupler: {frames: {base: {}, tip: {at: [b, 0, 0]}}}
  rocker: {frames: {base: {}, tip: {at: [c, 0, 0]}}}
constraints:
  - revolute: [world.o, crank.base]
    name: crank
  - revolute: [crank.tip, coupler.base]
    name: knee
  - revolute: [coupler.tip, rocker.base]
    name: elbow
  - revolute: [rocker.tip, world.p]
    name: ground
    range: [-pi/2, pi/2]
""")
    s = build(p)
    # the closing hinge's range is written in terms of its readout from the loop
    assert len(s.inequalities) == 1
    assert s.inequalities[0] == next(j for j in s.joints if j.name == "ground").values["c"]
    _, points = sample(s, tries=6, seed=3)
    assert points and {d for _, d in points} == {1}
