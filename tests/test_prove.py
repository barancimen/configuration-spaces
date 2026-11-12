"""The prover: claims, soundness of each kind of answer, and scaling by working on parts."""
import runpy
import time

import pytest

from cspace import ALL_KINDS, load
from cspace.claims import ClaimError
from cspace.prove import Prover, read_claims
from helpers import EXAMPLES

snake = runpy.run_path(str(EXAMPLES / "make_snake.py"))["snake"]


def run(path_or_text, claims, tmp_path=None, **kw):
    if tmp_path is not None:
        p = tmp_path / "m.yaml"
        p.write_text(path_or_text)
        path_or_text = p
    spec = load(path_or_text, ALL_KINDS)
    prover = Prover(spec, **kw)
    return {r.name: r for r in prover.prove_all(claims)}


def claims(**named):
    return [{"name": n, k: text} for n, (k, text) in named.items()]


CUBE = EXAMPLES / "cube_on_cube.yaml"


def test_cube_claims_from_the_example_file():
    spec = load(CUBE, ALL_KINDS)
    res = {r.name: r for r in Prover(spec, timeout=60).prove_all(spec.claims)}
    assert {n: r.status for n, r in res.items()} == {
        "neverConvex": "proved", "neverHangsOver": "proved", "restsOnTop": "proved",
        "neverOverlaps": "proved", "cornerReachable": "proved", "staysLeft": "refuted"}
    cx = res["staysLeft"].model
    assert float(eval(cx["dSlideX"])) > 0                    # a real counterexample


def test_wrong_cube_claims_are_refuted_not_proved():
    res = run(CUBE, claims(
        convexSometimes=("possible", "convex(base, slider)"),
        restsTooHigh=("always", "pos(slider).z == A/2 + a"),
        overlapsSomewhere=("possible", "overlap(base, slider)"),
        pastTheCorner=("possible", "slideX.d == (A - a)/2 + 1/1000"),
        hangsOver=("always", "forall(p, implies(inside(p, slider), inside([p.x, p.y, A/2 - 1/1000], base)))"),
    ), timeout=60)
    assert {r.status for r in res.values()} == {"refuted"}


def test_same_size_cubes_can_be_convex(tmp_path):
    text = """
parameters: {A: 2}
bodies:
  base: {shape: {box: [A, A, A]}, frames: {top: {at: [0, 0, A/2], z: [1, 0, 0]}}}
  slider: {shape: {box: [A, A, A]}, frames: {rail: {at: [0, 0, -A/2], z: [1, 0, 0]}}}
constraints:
  - fixed: [world, base]
  - prismatic: [base.top, slider.rail]
    name: s
    range: [-1, 1]
"""
    res = run(text, claims(convexWhenAligned=("possible", "convex(base, slider)"),
                           convexOnlyWhenAligned=("always", "implies(convex(base, slider), s.d == 0)")),
              tmp_path, timeout=60)
    assert res["convexWhenAligned"].status == "proved"
    assert res["convexWhenAligned"].model["dS"] == "0"
    assert res["convexOnlyWhenAligned"].status == "proved"


def test_algebraic_constants_in_frames(tmp_path):
    text = """
bodies:
  arm: {frames: {hub: {z: [1, 1, 0]}, tip: {at: [0, 0, 1]}}}
constraints:
  - revolute: [world, arm.hub]
    name: h
"""
    res = run(text, claims(
        tipOnSphere=("always", "dist2(pos(arm.tip), [0, 0, 0]) == 1"),
        tipStaysOnPlane=("always", "pos(arm.tip).x - pos(arm.tip).y == 0"),     # false
        tipReachesAxisPlane=("possible", "pos(arm.tip).z == 0"),
    ), tmp_path, timeout=30)
    assert res["tipOnSphere"].status == "proved"
    assert res["tipStaysOnPlane"].status == "refuted"
    assert res["tipReachesAxisPlane"].status == "proved"


def test_claim_errors(tmp_path):
    text = "bodies: {a: {}}\nconstraints:\n  - prismatic: [world, a]\n    name: p\n"
    res = run(text, claims(bad1=("always", "q.d > 0"), bad2=("always", "p.d / p.d > 0"),
                           bad3=("always", "p.d >"), bad4=("always", "p.d + 1")), tmp_path)
    assert {r.status for r in res.values()} == {"error"}
    assert "unknown name" in res["bad1"].detail
    assert "division" in res["bad2"].detail
    with pytest.raises(Exception):
        read_claims([{"always": "x", "possible": "y"}])


@pytest.fixture(scope="module")
def snake_results(tmp_path_factory):
    p = tmp_path_factory.mktemp("snake") / "snake.yaml"
    p.write_text(snake(1000))
    spec = load(p, ALL_KINDS)
    t0 = time.time()
    res = {r.name: r for r in Prover(spec, timeout=60, budget=60).prove_all(spec.claims)}
    return res, time.time() - t0


def test_large_chain_local_claims(snake_results):
    res, seconds = snake_results
    assert {n: r.status for n, r in res.items()} == {
        "linksTwoApartWithinReach": "proved", "bendLimitsTheGap": "proved",
        "canStraighten": "proved", "canBendFully": "proved", "canFoldTight": "refuted",
        "linksTwoApartNeverOverlap": "proved",
        "staysInPlane": "unknown",            # true, but local reasoning cannot reach the world
        "neverBendsLeft": "refuted"}
    assert all(r.bodies <= 12 for r in res.values())
    assert seconds < 240


def test_answers_from_parts_are_only_the_sound_kind(snake_results):
    res, _ = snake_results
    for r in res.values():
        if "others dropped" in r.detail:          # relaxation: proves always, refutes possible
            assert (r.kind, r.status) in (("always", "proved"), ("possible", "refuted"))
        if "held at zero" in r.detail:            # restriction: proves possible, refutes always
            assert (r.kind, r.status) in (("possible", "proved"), ("always", "refuted"))


def test_independent_groups_are_exact(tmp_path):
    # 300 sliders, each on its own rail fixed to the world: 300 independent groups
    n = 300
    world = "  world: {frames: {" + ", ".join(f"w{i}: {{at: [{i}, 0, 0]}}" for i in range(n)) + "}}"
    lines = ["bodies:", world] + [f"  b{i}: {{}}" for i in range(n)] + ["constraints:"]
    lines += [f"  - prismatic: [world.w{i}, b{i}]\n    name: p{i}\n    range: [0, 1]" for i in range(n)]
    text = "\n".join(lines)
    res = run(text, claims(inRange=("always", "pos(b150).z <= 1"),
                           atTop=("possible", "pos(b150).z == 1"),
                           belowZero=("possible", "pos(b150).z < 0")), tmp_path, timeout=30)
    assert {n: r.status for n, r in res.items()} == {"inRange": "proved", "atTop": "proved",
                                                       "belowZero": "refuted"}
    assert all(r.detail.startswith("exact") for r in res.values())
