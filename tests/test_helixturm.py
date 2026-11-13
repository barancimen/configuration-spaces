"""The helix tower: the generated files, its start pose, the lock proofs, a short MuJoCo run, the
replay page and the GIF renderer."""
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from cspace.mjcf import export, quat_matrix, trans
from helpers import EXAMPLES
from test_mjcf import geom_vertices, interiors_meet, keyframes, mj_fk, vec

MAKE = EXAMPLES / "make_helixturm.py"
RUN = EXAMPLES / "run_helixturm.py"
RENDER = EXAMPLES / "render_helixturm.py"


def generate(*args):
    return subprocess.run([sys.executable, str(MAKE), *args], capture_output=True, text=True, check=True).stdout


def test_generated_files_are_current():
    assert generate() == (EXAMPLES / "helixturm.yaml").read_text()
    assert generate("--lock") == (EXAMPLES / "helixturm_lock.yaml").read_text()


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """A tower with one segment per magazine: quick to export and to simulate."""
    d = tmp_path_factory.mktemp("helix")
    spec = d / "helixturm.yaml"
    spec.write_text(generate("--turns", "1"))
    xml, ex, xs = export(spec)
    model = d / "helixturm.xml"
    model.write_text(xml)
    return spec, model, xml, ex, xs


def test_export_settings_and_structure(small):
    _, _, xml, ex, xs = small
    root = ET.fromstring(xml)
    assert not ex.warnings
    assert root.find("option").get("timestep") == "0.001"
    default = root.find("default")
    assert default.find("geom").get("density") == "600"
    assert default.find("pair").get("solref") == "0.004 1"                  # contacts ...
    assert default.find("joint").get("solreflimit") == "0.004 1"            # ... and joint stops, equally stiff
    (motor,) = list(root.find("actuator"))
    assert motor.get("joint") == "rotor" and motor.get("ctrlrange") is None  # it turns without limit
    joints = {j.get("name"): j for j in root.find("worldbody").iter("joint")}
    assert joints["bolt0"].get("range") == "-0.019 0" and float(joints["bolt0"].get("stiffness")) > 0
    assert len(list(root.iter("freejoint"))) == 7                           # six segments and the head
    (key,) = keyframes(xml)
    fk = mj_fk(xml, key[1])
    for body, T in fk.items():
        assert np.abs(T - ex.pose(body, xs[0])).max() < 1e-8, body


def test_configuration_inverts_qpos(small):
    _, _, _, ex, xs = small
    rng = np.random.default_rng(1)
    x = xs[0].copy()
    index = {v: i for i, v in enumerate(ex.system.variables)}
    for j in ex.tree:                                   # move every coordinate, keeping circles and quaternions unit
        for r, v in j.values.items():
            if r in ("d", "x", "y", "z"):
                x[index[v]] += rng.uniform(-0.01, 0.01)
        if "c" in j.values:
            a = rng.uniform(-9, 9)                      # several turns: the rotor runs on
            x[index[j.values["c"]]], x[index[j.values["s"]]] = np.cos(a), np.sin(a)
        if "q0" in j.values:
            q = rng.normal(size=4)
            q *= np.sign(q[0]) / np.linalg.norm(q)
            for k in range(4):
                x[index[j.values[f"q{k}"]]] = q[k]
    back = ex.configuration(np.array(ex.qpos(x, xs[0])))
    assert np.abs(back - x).max() < 1e-12


def test_start_pose_has_no_part_inside_another(small):
    _, _, xml, _, _ = small
    root = ET.fromstring(xml)
    meshes = {m.get("name"): vec(m.get("vertex")).reshape(-1, 3) for m in root.find("asset")}
    (key,) = keyframes(xml)
    fk = mj_fk(xml, key[1])
    geoms, body_of = {}, {}
    for b in root.find("worldbody").iter("body"):
        for g in b.findall("geom"):
            geoms[g.get("name")], body_of[g.get("name")] = g, b.get("name")
    cache = {}

    def world(name):
        if name not in cache:
            g, T = geoms[name], fk[body_of[name]]
            if g.get("type") == "mesh":
                Tg = T @ trans(vec(g.get("pos"))) @ quat_matrix(vec(g.get("quat")))
                pts = np.array([(Tg @ np.append(p, 1))[:3] for p in meshes[g.get("mesh")]])
            elif g.get("type") == "sphere":             # a ball as a cloud of points on it
                c, r = (T @ np.append(vec(g.get("pos")), 1))[:3], float(g.get("size"))
                u = np.random.default_rng(0).normal(size=(60, 3))
                pts = c + r * u / np.linalg.norm(u, axis=1)[:, None]
            else:
                pts = geom_vertices(g, T)
            centre = pts.mean(axis=0)
            cache[name] = (pts, centre, np.linalg.norm(pts - centre, axis=1).max())
        return cache[name]

    pairs = list(root.find("contact"))
    assert len(pairs) > 1000
    close = 0
    for p in pairs:
        (P, cp, rp), (Q, cq, rq) = world(p.get("geom1")), world(p.get("geom2"))
        if np.linalg.norm(cp - cq) > rp + rq:           # far apart: nothing to check
            continue
        close += 1
        assert not interiors_meet(P, Q), (p.get("geom1"), p.get("geom2"))
    assert close > 50                                   # the check did look at the parts that touch


def test_lock_claims():
    """The joints between segments, decided by z3. The claims about turning take a minute each,
    so only the quick ones are run here."""
    from fractions import Fraction

    from cspace.compile import ALL_KINDS
    from cspace.prove import Prover
    from cspace.spec import load
    spec = load(EXAMPLES / "helixturm_lock.yaml", ALL_KINDS)
    quick = {"held": "proved", "comesOutLast": "proved", "neverComesOut": "refuted", "pinnedOutward": "proved",
             "pinnedSideways": "proved", "swingsWithoutLowerPin": "proved", "slidesOnItsOwnPins": "proved"}
    slow = {"pinnedInward", "pinnedTurningOneWay", "pinnedTurningOtherWay"}
    assert {c["name"] for c in spec.claims} == set(quick) | slow
    results = {r.name: r for r in Prover(spec, timeout=90).prove_all([c for c in spec.claims if c["name"] in quick])}
    for name, status in quick.items():
        assert results[name].status == status, (name, results[name].status, results[name].detail)
    assert Fraction(results["neverComesOut"].model["dOut"]) > Fraction(2, 1000)      # a real way out
    assert float(results["swingsWithoutLowerPin"].model["sSeg"].rstrip(".")) >= 0.03


def test_two_steps_in_mujoco(small):
    """The machine itself, simulated: two segments go in, latch and rise; then the rotor stops and
    the tower stands. Needs MuJoCo; about ten seconds of simulation on a slow machine."""
    pytest.importorskip("mujoco")
    _, model, _, _, _ = small
    run = subprocess.run([sys.executable, str(RUN), str(model), "--steps", "2", "--hold", "1"],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert "): 2 of 2" in run.stdout and "RESULT: the tower stands" in run.stdout


def test_replay_page(small, tmp_path):
    from cspace.viewer import page
    spec, _, _, ex, xs = small
    q0 = np.array(ex.qpos(xs[0], xs[0]))
    q1 = q0.copy()
    q1[0] += 0.5                                        # the rotor is the first joint of the model
    rec = tmp_path / "run.npz"
    np.savez(rec, time=np.array([0.0, 0.04, 0.08]), qpos=np.array([q0, q1, q0]))
    html = page([(spec, {})], replay=rec)
    assert "replay: {dt: 0.04, frames: [[" in html and 'id="replayPlay"' in html
    plain = page([(EXAMPLES / "gears.yaml", {})])
    assert "replay: null" in plain


def test_render_gif(small, tmp_path):
    """Three recorded frames, drawn by MuJoCo without a display, forwards and then backwards."""
    pytest.importorskip("mujoco")
    Image = pytest.importorskip("PIL.Image")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    _, model, _, ex, xs = small
    q0 = np.array(ex.qpos(xs[0], xs[0]))
    q1 = q0.copy()
    q1[0] += 0.5
    rec, gif = tmp_path / "run.npz", tmp_path / "tower.gif"
    np.savez(rec, time=np.array([0.0, 0.04, 0.08]), qpos=np.array([q0, q1, q0]))
    run = subprocess.run([sys.executable, str(RENDER), str(model), str(rec), "-o", str(gif), "--speed", "1", "--rewind",
                          "--pause", "0", "--height", "160"], capture_output=True, text=True, timeout=300,
                         env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")})
    if run.returncode != 0 and "GL" in run.stderr:
        pytest.skip("no OpenGL context on this machine: " + run.stderr.strip().splitlines()[-1])
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    with Image.open(gif) as im:
        assert im.n_frames == 6 and im.size[1] == 160 and im.size[0] > 160
        im.seek(1)
        colours = im.convert("RGB").getcolors(maxcolors=100000)
    assert len(colours) > 20                              # a drawn scene, not an empty picture
