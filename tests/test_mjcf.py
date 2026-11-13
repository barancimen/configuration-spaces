"""MJCF export. MuJoCo is not installed here, so the exported model is checked with a small
forward-kinematics interpreter that follows MuJoCo's conventions: body pos/quat relative to the
parent, then the body's joints applied in order in the body frame (hinge: rotation by qpos - ref
about axis through pos; slide: translation by qpos - ref along axis; ball: rotation by the qpos
quaternion about pos; freejoint: absolute position and quaternion), qpos in depth-first order."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from cspace.mjcf import export, quat_matrix, rot, trans
from helpers import EXAMPLES

vec = lambda s: np.array([float(v) for v in s.split()])


def mj_fk(xml, qpos):
    root = ET.fromstring(xml)
    poses, qpos, i = {}, list(qpos), 0

    def walk(el, T_parent):
        nonlocal i
        for b in el.findall("body"):
            T = T_parent @ trans(vec(b.get("pos", "0 0 0"))) @ quat_matrix(vec(b.get("quat", "1 0 0 0")))
            for j in b:
                if j.tag == "freejoint":
                    T = trans(qpos[i:i + 3]) @ quat_matrix(qpos[i + 3:i + 7])
                    i += 7
                elif j.tag == "joint":
                    kind, p = j.get("type", "hinge"), vec(j.get("pos", "0 0 0"))
                    if kind == "ball":
                        T = T @ trans(p) @ quat_matrix(qpos[i:i + 4]) @ trans(-p)
                        i += 4
                        continue
                    axis, d = vec(j.get("axis")), qpos[i] - float(j.get("ref", "0"))
                    i += 1
                    if kind == "hinge":
                        T = T @ trans(p) @ rot(axis, d) @ trans(-p)
                    else:
                        T = T @ trans(axis / np.linalg.norm(axis) * d)
            poses[b.get("name")] = T
            walk(b, T)

    walk(root.find("worldbody"), np.eye(4))
    assert i == len(qpos), "qpos length does not match the joints"
    return poses


def keyframes(xml):
    return [(k.get("name"), vec(k.get("qpos")), k.get("ctrl")) for k in ET.fromstring(xml).find("keyframe")]


def joint_values(xml, qpos):
    """Name -> qpos value for hinges and slides, in model order."""
    out, i = {}, 0
    for el in ET.fromstring(xml).find("worldbody").iter():
        if el.tag == "freejoint":
            i += 7
        elif el.tag == "joint" and el.get("name"):
            if el.get("type") == "ball":
                i += 4
            else:
                out[el.get("name")] = qpos[i]
                i += 1
    return out


COVERAGE = """
# every joint kind, in both orientations, with tilted frames; three short branches from the world
bodies:
  world: {frames: {w1: {at: [0, 0, 1]}, w2: {at: [1, 0, 0], z: [0, 1, 1]}, w3: {at: [0, 1, 0]}}}
  a: {frames: {p: {at: [0.3, 0, 0], z: [1, 1, 0]}, q: {at: [1, 0, 0.2]}}}
  b: {frames: {p: {at: [0, 0.2, 0], z: [0, 1, 1]}, q: {at: [0.1, 0, 0]}}}
  c: {frames: {p: {at: [0, 0, 0.1]}}}
  d: {frames: {p: {at: [0.2, 0, 0], x: [0, 1, 0]}}}
  e: {frames: {p: {at: [0, 0.3, 0]}}}
  f: {frames: {q: {at: [0.5, 0, 0]}}}
  g: {frames: {p: {at: [0, 0, 0.4], z: [1, 0, 1]}}}
constraints:
  - universal: [world.w1, a.p]
    name: u
  - cylindrical: [b.p, a.q]
    name: cy
  - planar: [b.q, c.p]
    name: pl
  - revolute: [d.p, world.w2]
    name: rv
  - free: [d.p, e.p]
    name: fr
  - spherical: [world.w3, f]
    name: sp
  - spherical: [g.p, f.q]
    name: sp2
"""


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    cov = tmp_path_factory.mktemp("cov") / "coverage.yaml"
    cov.write_text(COVERAGE)
    cases = {
        "stewart": (EXAMPLES / "stewart.yaml", {}, ["leg1", "leg2", "leg3", "leg4", "leg5", "leg6"]),
        "fourbar": (EXAMPLES / "fourbar.yaml", {"a": "2", "b": "5", "c": "4", "d": "6"}, ["crank"]),
        "slider_crank": (EXAMPLES / "slider_crank.yaml", {"r": "1", "l": "3"}, []),
        "bennett": (EXAMPLES / "bennett.yaml", {}, []),
        "rssr": (EXAMPLES / "rssr.yaml", {}, []),
        "gears": (EXAMPLES / "gears.yaml", {}, []),
        "door_stop": (EXAMPLES / "door_stop.yaml", {}, []),
        "tethered_puck": (EXAMPLES / "tethered_puck.yaml", {}, []),
        "coverage": (cov, {}, []),
    }
    return {name: export(path, given, keyframes=3, motors=motors) for name, (path, given, motors) in cases.items()}


@pytest.mark.parametrize("name", ["stewart", "fourbar", "slider_crank", "bennett", "rssr", "gears",
                                  "door_stop", "tethered_puck", "coverage"])
def test_mujoco_kinematics_reproduce_every_sample(exports, name):
    xml, ex, xs = exports[name]
    keys = keyframes(xml)
    assert len(keys) == len(xs) == 3
    for (key, qpos, _), x in zip(keys, xs):
        fk = mj_fk(xml, qpos)
        for body, T in fk.items():
            assert np.abs(T - ex.pose(body, x)).max() < 1e-8, (name, key, body)


@pytest.mark.parametrize("name", ["stewart", "fourbar", "slider_crank", "bennett", "rssr", "gears"])
def test_equalities_hold_in_every_keyframe(exports, name):
    xml, ex, xs = exports[name]
    root = ET.fromstring(xml)
    keys = keyframes(xml)
    ref = mj_fk(xml, keys[0][1])                  # the reference pose: MuJoCo derives anchors from it
    ref["world"] = np.eye(4)
    eqs = list(root.find("equality"))
    assert eqs or name == "gears"
    for _, qpos, _ in keys[1:]:
        fk = mj_fk(xml, qpos)
        fk["world"] = np.eye(4)
        jv, jv0 = joint_values(xml, qpos), joint_values(xml, keys[0][1])
        for e in eqs:
            if e.tag == "connect":
                b1, b2, a1 = e.get("body1"), e.get("body2"), np.append(vec(e.get("anchor")), 1)
                a2 = np.linalg.inv(ref[b2]) @ ref[b1] @ a1
                assert np.abs(fk[b1] @ a1 - fk[b2] @ a2).max() < 1e-8, (name, e.get("name"))
            elif e.tag == "weld":
                rel0 = np.linalg.inv(ref[e.get("body1")]) @ ref[e.get("body2")]
                rel = np.linalg.inv(fk[e.get("body1")]) @ fk[e.get("body2")]
                assert np.abs(rel - rel0).max() < 1e-8
            elif e.tag == "joint":
                k = vec(e.get("polycoef"))[1]
                j1, j2 = e.get("joint1"), e.get("joint2")
                d1 = (jv[j1] - jv0[j1] + np.pi) % (2 * np.pi) - np.pi
                d2 = (jv[j2] - jv0[j2] + np.pi) % (2 * np.pi) - np.pi
                assert abs(((d1 - k * d2) + np.pi) % (2 * np.pi) - np.pi) < 1e-8


def test_structure_names_and_controls(exports):
    xml, ex, xs = exports["stewart"]
    root = ET.fromstring(xml)
    for tag in ("body", "joint", "geom"):
        names = [e.get("name") for e in root.iter(tag)]
        assert len(names) == len(set(names)), tag
    assert [e.tag for e in root.find("equality")] == ["connect"] * 6
    assert len(root.find("actuator")) == 6
    for _, qpos, ctrl in keyframes(xml):
        vals = joint_values(xml, qpos)
        assert np.allclose(vec(ctrl), [vals[f"leg{i}"] for i in range(1, 7)])
    assert len(list(root.iter("freejoint"))) == 1          # the plate
    _, ex_puck, _ = exports["tethered_puck"]
    assert any("no MuJoCo equivalent" in w for w in ex_puck.warnings)
    xml_rssr, ex_rssr, _ = exports["rssr"]
    assert any("cone limit" in w for w in ex_rssr.warnings)
    xml_door = exports["door_stop"][0]
    assert len(ET.fromstring(xml_door).find("contact")) == 1
