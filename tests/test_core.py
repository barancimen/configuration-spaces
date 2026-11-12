"""Milestone 1: joints, loop closure, normal form, export."""
import re

import pytest
from sympy import (And, Eq, Ge, Symbol, cos, expand, groebner, pi, reduced, sin, symbols,
                   mathematica_code)
from sympy.parsing.mathematica import parse_mathematica

from cspace import SpecError, to_wl
from cspace.compile import Reducer
from cspace.logic import ForAll
from helpers import EXAMPLES, example

ALL = sorted(p.stem for p in EXAMPLES.glob("*.yaml") if p.stem != "screw")


def sym(name):
    return Symbol(name, real=True)


def test_fourbar_matches_textbook_loop_equations():
    s = example("fourbar", overrides={"a": "2", "b": "5", "c": "4", "d": "6"})
    c1, s1, c2, s2, c3, s3 = (sym(n) for n in "cCrank sCrank cKnee sKnee cElbow sElbow".split())
    assert s.variables == [c1, s1, c2, s2, c3, s3]
    # absolute directions of coupler and rocker from the relative hinge angles
    C2, S2 = c1 * c2 - s1 * s2, s1 * c2 + c1 * s2
    C3, S3 = C2 * c3 - S2 * s3, S2 * c3 + C2 * s3
    textbook = [2 * c1 + 5 * C2 + 4 * C3 - 6, 2 * s1 + 5 * S2 + 4 * S3,
                c1**2 + s1**2 - 1, c2**2 + s2**2 - 1, c3**2 + s3**2 - 1]
    order = dict(order="grevlex")
    assert groebner(s.equations, *s.variables, **order) == groebner(textbook, *s.variables, **order)


def test_slider_crank_distance_relation_is_in_the_ideal():
    s = example("slider_crank")
    c, sn, d, r, l = sym("cCrank"), sym("sCrank"), sym("dSlide"), sym("r"), sym("l")
    target = expand((d - r * c)**2 + r**2 * sn**2 - l**2)
    G = groebner(s.equations, *s.variables, order="lex")
    _, rem = reduced(target, list(G), *s.variables, order="lex")
    assert rem == 0
    assert set(s.inequalities) == {d + 5, 5 - d}


def test_gear_coupling_polynomials():
    s = example("gears")
    cp, sp, cw, sw = (sym(n) for n in "cP sP cW sW".split())
    # 20 thP + 40 thW = 0  ->  thP + 2 thW = 0  ->  zP * zW^2 = 1, written in normal form (sW^2 -> 1 - cW^2)
    re_part = cp * (2 * cw**2 - 1) - 2 * sp * cw * sw - 1
    im_part = sp * (2 * cw**2 - 1) + 2 * cp * cw * sw
    eqs = set(s.equations)
    assert expand(-re_part) in eqs or expand(re_part) in eqs
    assert expand(im_part) in eqs or expand(-im_part) in eqs
    assert s.inequalities == [cw]          # range [-pi/2, pi/2]  ->  cos(th) >= 0


def test_two_rods_matches_the_planned_output():
    s = example("two_rods")
    c, sn, l1 = sym("cElbow"), sym("sElbow"), sym("l1")
    assert s.variables == [c, sn]
    assert s.equations == [c**2 + sn**2 - 1]
    assert s.inequalities == [c]
    M = s.poses["rod2"]
    assert list(M[:3, 3]) == [l1, 0, 0] and M[0, 0] == c and M[1, 0] == sn


def test_range_inequality_is_exact_on_the_circle():
    th = symbols("th", real=True)
    s = example("gears")
    # a range [lo, hi] becomes c cos m + s sin m >= cos(w/2); check it on sample angles for [-pi/2, pi/2]
    g = s.inequalities[0]
    for k in range(-12, 13):
        a = k * pi / 12
        inside = abs(a) <= pi / 2
        val = g.subs({sym("cW"): cos(a), sym("sW"): sin(a)})
        assert (val >= 0) == inside or abs(float(val)) < 1e-12


def test_normal_form_reduces_squares():
    c, s_ = symbols("c s", real=True)
    red = Reducer()
    red.add(s_, 1 - c**2)
    assert red(s_**3 * c + (c**2 + s_**2)**2 - 1) == expand(s_ * c * (1 - c**2))
    assert red.canon(4 * c**2 + 4 * s_**2 - 4) == 0
    assert red.canon(-6 * c + 3) == 2 * c - 1


@pytest.mark.parametrize("name", ALL)
def test_mathematica_text_parses_back(name):
    s = example(name)
    text = to_wl(s)
    assert "_" not in re.sub(r'"[^"]*"|\(\*.*?\*\)', "", text), "underscore outside strings/comments"
    plain = {v: Symbol(v.name) for v in s.variables + s.params}
    for e in s.equations + s.inequalities:
        back = parse_mathematica(mathematica_code(e))
        assert expand(back - e.subs(plain)) == 0
    for f in s.formulas:
        if not f.has(ForAll):
            back = parse_mathematica(mathematica_code(f))
            assert back == f.subs(plain)
    for M in s.poses.values():
        for e in M:
            assert expand(parse_mathematica(mathematica_code(e)) - e.subs(plain)) == 0


def test_errors_are_reported_clearly(tmp_path):
    def run(text):
        p = tmp_path / "m.yaml"
        p.write_text(text)
        from cspace import build
        return build(p)

    with pytest.raises(SpecError, match="unknown body"):
        run("bodies: {a: {}}\nconstraints:\n  - revolute: [world, b]\n")
    with pytest.raises(SpecError, match="no frame"):
        run("bodies: {a: {}}\nconstraints:\n  - revolute: [world, a.tip]\n")
    with pytest.raises(SpecError, match="reversed"):
        run("bodies: {a: {}}\nconstraints:\n  - revolute: [world, a]\n    range: [1, 0]\n")
    with pytest.raises(SpecError, match="not polynomial"):
        run("bodies: {a: {}, b: {}}\nconstraints:\n  - revolute: [world, a]\n    name: h\n"
            "  - prismatic: [world, b]\n    name: p\n  - couple: {h: 1, p: 1}\n")
    with pytest.raises(SpecError, match="unknown name"):
        run("bodies: {a: {frames: {f: {at: [q, 0, 0]}}}}\n")
    with pytest.raises(SpecError, match="clashes"):
        run("parameters: {cH: 1}\nbodies: {a: {}}\nconstraints:\n  - revolute: [world, a]\n    name: h\n")
