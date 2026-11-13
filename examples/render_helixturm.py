"""Render a recorded helix-tower build as an animated GIF (pip install mujoco pillow; needs ffmpeg).

    python3 examples/run_helixturm.py helixturm.xml --record run.npz
    MUJOCO_GL=egl python3 examples/render_helixturm.py helixturm.xml run.npz --rewind -o helixturm.gif

Two views side by side on white, with no lettering: the whole tower, solid, and its base from
closer up. In the close view the walls in front (the segments, magazines, guides and floor of the
columns on the camera's side, and that side of the head) turn to glass during the rotor's first
half turn, so that the cam, its followers and the latch bolts show: their shade fades to 5%,
their lines stay.
Drawn like a comic: every part is grey, one shade per kind of part, with a black line around
every body and along every corner. MuJoCo draws the shaded picture; the lines are found afterwards
from how far away each pixel is and which body it shows. MUJOCO_GL=egl draws without a display.
The time the tower stands still at the end is left out. With --rewind the pictures then run
backwards to the start: the loop has no jump, and the walls come back at its end. That half is
the same recording in reverse order, not a simulation of the tower coming down.
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import mujoco
from mujoco import gl_context
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_helixturm as hx  # noqa: E402  (the dimensions)
from run_helixturm import Tower  # noqa: E402

PAPER, INK, DIVIDER = 255, 16, 226
# Everything is metal: one shade of grey per kind of part, whatever colours the file gives them
# (0 is black, 1 is white). Parts that meet in the picture are far apart in shade.
SHADES = {"panel": 0.88, "cap": 0.74, "stator": 0.62, "rotor": 0.50, "push": 0.44, "lift": 0.36, "foot": 0.36, "bolt": 0.20}
WALLS = ("panel", "stator", "push", "head")     # bodies that hide the mechanism
FRONT = 2                                       # geom group of the walls in front, so they can be left out
FADE = 0.5                                      # turns of the rotor over which the walls in front fade away
FAINT = 0.05                                    # what is left of them then: 5% of their shade, and their lines
CORNER = math.radians(24)                       # a surface that bends by more than this gets a line
REACH = 1.8                                     # metres in front of and behind the tower's axis that are drawn
FOVY = 20                                       # the cameras' field of view, degrees
# per view: the height above the floor where the picture ends (None: over the finished tower), where
# the camera stands (degrees around the axis, degrees above the horizon), whether the walls in
# front of it fade away, and by how many pixels the lines around bodies are widened (at twice
# the picture's size)
VIEWS = ((None, 14, 9, False, 0), (0.60, 14, 13, True, 1))


def aim(points, around, above, margin=1.07):
    """A camera standing at these angles that just fits the points into the picture's height: where it
    looks (a point on the axis), from how far, and the picture's width over its height."""
    a, e = math.radians(around), math.radians(above)
    back = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])   # from the scene to the camera
    right = np.array([-math.sin(a), math.cos(a), 0.0])
    up = np.cross(back, right)
    tan = math.tan(math.radians(FOVY / 2))
    pts, z, far = np.array(points, float), 0.0, 10.0
    for _ in range(40):                                        # nearer points look larger, so step toward the fit
        rel = pts - (0, 0, z)
        depth = far - rel @ back
        v, h = rel @ up / depth, rel @ right / depth
        z += (v.max() + v.min()) / 2 * far / up[2]
        far *= (v.max() - v.min()) / 2 / tan
    return z, far * margin, np.abs(h).max() / tan


class Stage:
    """MuJoCo drawing into memory: the shaded picture, how far away each pixel is, and which body it shows."""

    def __init__(self, m, width, height):
        self.m = m
        m.vis.global_.offwidth, m.vis.global_.offheight = width, height
        m.vis.global_.fovy = FOVY
        m.vis.quality.offsamples = 0                           # the pictures are drawn large and shrunk instead
        m.vis.headlight.ambient[:] = 0.50                      # the three add up to about 1, so no face burns out to white
        m.vis.headlight.diffuse[:] = 0.28
        m.vis.headlight.specular[:] = 0
        m.light_dir[:] = np.array([0.45, 0.55, -0.7]) / np.linalg.norm([0.45, 0.55, -0.7])   # from the upper left
        m.light_diffuse[:] = 0.27
        m.light_specular[:] = 0
        m.light_castshadow[:] = 0
        self.gl = gl_context.GLContext(width, height)
        self.gl.make_current()
        self.con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        self.scene = mujoco.MjvScene(m, maxgeom=m.ngeom + 100)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.opt = mujoco.MjvOption()

    def shot(self, d, cam, width, height, walls):
        """Colours (0 to 255), distances from the camera in metres, and body numbers (-1: nothing there)."""
        m, scene, flags = self.m, self.scene, mujoco.mjtRndFlag
        self.opt.geomgroup[FRONT] = walls
        near, far = cam.distance - REACH, cam.distance + REACH
        m.vis.map.znear, m.vis.map.zfar = near / m.stat.extent, far / m.stat.extent
        mujoco.mjv_updateScene(m, d, self.opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        rect = mujoco.MjrRect(0, 0, width, height)
        rgb, ids = np.empty((height, width, 3), np.uint8), np.empty((height, width, 3), np.uint8)
        raw = np.empty((height, width), np.float32)
        scene.flags[flags.mjRND_SEGMENT] = scene.flags[flags.mjRND_IDCOLOR] = 0
        mujoco.mjr_render(rect, scene, self.con)
        mujoco.mjr_readPixels(rgb, raw, rect, self.con)
        scene.flags[flags.mjRND_SEGMENT] = scene.flags[flags.mjRND_IDCOLOR] = 1     # each geom in a colour that is its number
        mujoco.mjr_render(rect, scene, self.con)
        mujoco.mjr_readPixels(ids, None, rect, self.con)
        number = ids[..., 0].astype(np.int32) + ids[..., 1].astype(np.int32) * 256 + ids[..., 2].astype(np.int32) * 65536
        owner = np.full(scene.ngeom + 1, -1, np.int32)
        for g in scene.geoms[:scene.ngeom]:
            if g.segid >= 0 and g.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                owner[g.segid + 1] = m.geom_bodyid[g.objid]
        # the depth buffer runs from 1 at the near plane to 0 at the far plane
        z = far * near / (raw.astype(np.float64) * (far - near) + near)
        return rgb[::-1], z[::-1], owner[number][::-1]

    def close(self):                                           # before Python exits, or the GL library complains
        self.con.free()
        self.gl.free()


def spread(mask, by):
    """The mask, widened by this many pixels."""
    for _ in range(by):
        wide = mask.copy()
        wide[1:] |= mask[:-1]
        wide[:-1] |= mask[1:]
        wide[:, 1:] |= mask[:, :-1]
        wide[:, :-1] |= mask[:, 1:]
        mask = wide
    return mask


def lines(z, body, widen):
    """Where the black lines go: where two bodies meet (or a body ends), and where the surface bends
    or jumps, which is where the slope of the distance from the camera changes from one pixel to
    the next."""
    h, w = body.shape
    pixel = 2 * z * math.tan(math.radians(FOVY / 2)) / h       # the size of a pixel at that distance
    corner, seam = np.zeros((h, w), bool), np.zeros((h, w), bool)
    for axis in (0, 1):
        zz, pp, cc, bb, ss = (np.moveaxis(a, axis, 0) for a in (z, pixel, corner, body, seam))
        before, after = np.arctan((zz[1:-1] - zz[:-2]) / pp[1:-1]), np.arctan((zz[2:] - zz[1:-1]) / pp[1:-1])
        cc[1:-1] |= np.abs(after - before) > CORNER
        differ = bb[1:] != bb[:-1]
        ss[1:] |= differ
        ss[:-1] |= differ
    return (corner & (body >= 0)) | spread(seam, widen)


def on_paper(rgb, body):
    """The shaded picture as numbers, white where there is nothing."""
    out = rgb.astype(np.float32)
    out[body < 0] = PAPER
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="MJCF file written by cspace.mjcf from examples/helixturm.yaml")
    ap.add_argument("recording", help=".npz file written by run_helixturm.py --record")
    ap.add_argument("-o", "--output", default="helixturm.gif")
    ap.add_argument("--speed", type=float, default=4.0, help="times faster than real time (default 4)")
    ap.add_argument("--fps", type=int, default=25, help="pictures per second: 50, 25 or 20 play evenly in browsers")
    ap.add_argument("--height", type=int, default=640, help="picture height in pixels (default 640)")
    ap.add_argument("--rewind", action="store_true", help="then run the pictures backwards to the start")
    ap.add_argument("--colors", type=int, default=32, help="greys in the GIF's palette (default 32)")
    ap.add_argument("--pictures", help="keep the single pictures (PNG files) in this folder")
    ap.add_argument("--frame", type=float, help="write only the picture at this simulated time, as a PNG")
    args = ap.parse_args()

    rec = np.load(args.recording)
    times, qpos = rec["time"], rec["qpos"]
    px = args.height
    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    if qpos.shape[1] != m.nq:
        raise SystemExit(f"{args.recording} has {qpos.shape[1]} coordinates per frame, the model has {m.nq}")
    tw = Tower(m, d)                                           # also sets the dimensions in hx
    turns = len(tw.panels) // tw.n
    rotor = qpos[:, tw.q_rotor] - qpos[0, tw.q_rotor]
    turning = np.nonzero(np.abs(rotor - rotor[-1]) > 1e-4)[0]
    if len(turning):                                           # once the rotor has stopped, nothing more happens
        times, qpos, rotor = (a[:turning[-1] + 2] for a in (times, qpos, rotor))

    body = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in m.geom_bodyid]
    kind = [b.rstrip("0123456789") for b in body]
    cap = [k == "head" and math.hypot(*m.geom_pos[g][:2]) < 0.01 for g, k in enumerate(kind)]   # the head's lid, on the axis
    for g, k in enumerate(kind):
        k = "cap" if cap[g] else "panel" if k == "head" else k    # the head's segment stubs are shaded like segments
        if k in SHADES:
            m.geom_rgba[g] = (SHADES[k],) * 3 + (1,)

    # what the views have to show (metres): the cam's base disc, the magazines, the finished tower's cap
    ring = lambda r, z, n=24: [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z) for i in range(n)]
    low = ring(hx.RHO + 0.045, hx.ZT0 - hx.OVER - 0.04) + ring(hx.X_END + 0.02, -0.02) + ring(hx.X_END + 0.02, hx.H)
    top = hx.level(tw.n) + hx.H + 0.03 + turns * hx.P
    views = []
    for cut, around, above, fades, widen in VIEWS:
        points = low + (ring(hx.R_OUT + 0.03, top) if cut is None else [(0, 0, cut)])
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        height, cam.distance, wide = aim(points, around, above)
        cam.lookat[:] = (0, 0, height)
        cam.azimuth, cam.elevation = around + 180, -above
        views.append((2 * math.ceil(px * wide * 1.03 / 2), cam, fades, widen))
        for g in range(m.ngeom if fades else 0):               # the walls on the camera's side of the axis
            if kind[g] not in WALLS or cap[g]:
                continue
            if kind[g] == "head":                              # one body: each piece by where it is
                column = math.atan2(m.geom_pos[g][1], m.geom_pos[g][0])
            else:
                column = int(body[g][len(kind[g]):]) % tw.n * tw.theta
            if math.cos(column - math.radians(around)) > 0:
                m.geom_group[g] = FRONT
    width = sum(v[0] for v in views)
    stage = Stage(m, 2 * max(v[0] for v in views), 2 * px)

    def picture(k):
        """The scene at recorded frame k."""
        d.qpos[:] = qpos[k]
        mujoco.mj_forward(m, d)
        s = min(1.0, max(0.0, rotor[k] / (2 * math.pi * FADE)))
        s = s * s * (3 - 2 * s)                                # how far the walls in front have faded: eased, 0 to 1
        out = np.full((px, width, 3), DIVIDER, np.float32)
        x = 0
        for w, cam, fades, widen in views:
            rgb, z, seen = stage.shot(d, cam, 2 * w, 2 * px, True)
            image, ink = on_paper(rgb, seen), lines(z, seen, widen)
            if fades and s > 0:                                # the walls in front thin out over what is behind them
                rgb, z, seen = stage.shot(d, cam, 2 * w, 2 * px, False)
                behind = on_paper(rgb, seen)
                behind[lines(z, seen, widen)] = INK
                keep = 1 - s * (1 - FAINT)
                image = keep * image + (1 - keep) * behind
            image[ink] = INK                                   # ... but they keep their lines
            image = image.reshape(px, 2, w, 2, 3).mean(axis=(1, 3))            # drawn at twice the size, for smooth lines
            out[:, x + (1 if x else 0):x + w] = image[:, (1 if x else 0):]     # a line between the views
            x += w
        return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))

    if args.frame is not None:
        k = int(np.abs(times - args.frame).argmin())
        out = args.output if args.output.endswith(".png") else str(Path(args.output).with_suffix(".png"))
        picture(k).save(out)
        stage.close()
        print(f"picture at {times[k]:.2f} s written to {out} ({width} x {px})")
        return 0

    step = args.speed / args.fps                               # simulated seconds between pictures
    ks = [int(np.abs(times - t).argmin()) for t in np.arange(times[0], times[-1] + step / 2, step)]
    with tempfile.TemporaryDirectory() as tmp:
        if args.pictures:
            tmp = args.pictures
            Path(tmp).mkdir(parents=True, exist_ok=True)
            for old in Path(tmp).glob("f[0-9]*.png"):
                old.unlink()
        for j, k in enumerate(ks):
            picture(k).save(f"{tmp}/f{j:05d}.png", compress_level=1)
            if j % 25 == 0:
                print(f"  picture {j + 1} of {len(ks)}", flush=True)
        stage.close()
        count = len(ks)
        if args.rewind:                                        # back again, without showing either end twice
            for j in range(len(ks) - 2, 0, -1):
                os.link(f"{tmp}/f{j:05d}.png", f"{tmp}/f{count:05d}.png")
                count += 1
        src = ["-framerate", str(args.fps), "-i", f"{tmp}/f%05d.png"]
        quiet = ["ffmpeg", "-y", "-loglevel", "error"]
        subprocess.run(quiet + src + ["-vf", f"palettegen=max_colors={args.colors}:stats_mode=diff", f"{tmp}/palette.png"], check=True)
        subprocess.run(quiet + src + ["-i", f"{tmp}/palette.png", "-lavfi", "paletteuse=dither=none:diff_mode=rectangle",
                                      args.output], check=True)
    size = Path(args.output).stat().st_size
    print(f"{args.output}: {count} pictures, {width} x {px}, {count / args.fps:.1f} s, {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
