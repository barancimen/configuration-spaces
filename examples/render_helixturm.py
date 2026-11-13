"""Render a recorded helix-tower build as an animated GIF (pip install mujoco pillow; needs ffmpeg).

    python3 examples/run_helixturm.py helixturm.xml --record run.npz
    MUJOCO_GL=egl python3 examples/render_helixturm.py helixturm.xml run.npz --rewind -o helixturm.gif

Two views side by side on white, drawn by MuJoCo's own renderer, with no lettering: the whole
tower, solid, and its base from closer up. In the close view the walls in front (the segments,
magazines, guides and floor of the columns on the camera's side, and that side of the head) fade
away completely during the rotor's first half turn, so that the cam, its followers and the latch
bolts show. Every part is grey, one shade per kind of part. The time the tower stands still at
the end is left out.
With --rewind the pictures then run backwards to the start: the loop has no jump, and the walls
come back at its end. That half is the same recording in reverse order, not a simulation of the
tower coming down. MUJOCO_GL=egl draws without a display.
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_helixturm as hx  # noqa: E402  (the dimensions)
from run_helixturm import Tower  # noqa: E402

BACKGROUND = (1.0, 1.0, 1.0)
DIVIDER = (226, 226, 226)
# Everything is metal: one shade of grey per kind of part, whatever colours the file gives them
# (0 is black, 1 is white). Parts that meet in the picture are far apart in shade.
SHADES = {"panel": 0.88, "head": 0.74, "stator": 0.62, "rotor": 0.50, "push": 0.44, "lift": 0.34, "foot": 0.34, "bolt": 0.06}
WALLS = ("panel", "stator", "push", "head")     # bodies that hide the mechanism
FADE = 0.5                                      # turns of the rotor over which the walls in front fade away
# per view: the height above the floor where the picture ends (None: over the finished tower), where
# the camera stands (degrees around the axis, degrees above the horizon), whether the walls in
# front of it fade away
VIEWS = ((None, 14, 9, False), (0.60, 14, 13, True))


def load(path, width, height):
    spec = mujoco.MjSpec.from_file(path)
    spec.add_texture(name="backdrop", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=BACKGROUND, rgb2=BACKGROUND, width=8, height=48)
    m = spec.compile()
    m.vis.global_.offwidth, m.vis.global_.offheight = width, height
    m.vis.global_.fovy = 20
    m.vis.quality.offsamples = 4
    m.vis.headlight.ambient[:] = 0.50                          # the three add up to about 1, so no face burns out to white
    m.vis.headlight.diffuse[:] = 0.28
    m.vis.headlight.specular[:] = 0
    m.light_dir[:] = np.array([0.45, 0.55, -0.7]) / np.linalg.norm([0.45, 0.55, -0.7])   # from the upper left
    m.light_diffuse[:] = 0.27
    m.light_specular[:] = 0
    m.light_castshadow[:] = 0
    return m


def aim(points, around, above, fovy, margin=1.07):
    """A camera standing at these angles that just fits the points into the picture's height: where it
    looks (a point on the axis), from how far, and the picture's width over its height."""
    a, e = math.radians(around), math.radians(above)
    back = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])   # from the scene to the camera
    right = np.array([-math.sin(a), math.cos(a), 0.0])
    up = np.cross(back, right)
    tan = math.tan(math.radians(fovy / 2))
    pts, z, far = np.array(points, float), 0.0, 10.0
    for _ in range(40):                                        # nearer points look larger, so step toward the fit
        rel = pts - (0, 0, z)
        depth = far - rel @ back
        v, h = rel @ up / depth, rel @ right / depth
        z += (v.max() + v.min()) / 2 * far / up[2]
        far *= (v.max() - v.min()) / 2 / tan
    return z, far * margin, np.abs(h).max() / tan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="MJCF file written by cspace.mjcf from examples/helixturm.yaml")
    ap.add_argument("recording", help=".npz file written by run_helixturm.py --record")
    ap.add_argument("-o", "--output", default="helixturm.gif")
    ap.add_argument("--speed", type=float, default=4.0, help="times faster than real time (default 4)")
    ap.add_argument("--fps", type=int, default=25, help="pictures per second: 50, 25 or 20 play evenly in browsers")
    ap.add_argument("--height", type=int, default=640, help="picture height in pixels (default 640)")
    ap.add_argument("--rewind", action="store_true", help="then run the pictures backwards to the start")
    ap.add_argument("--colors", type=int, default=128, help="colours in the GIF's palette (default 128)")
    ap.add_argument("--pictures", help="keep the single pictures (PNG files) in this folder")
    ap.add_argument("--frame", type=float, help="write only the picture at this simulated time, as a PNG")
    args = ap.parse_args()

    rec = np.load(args.recording)
    times, qpos = rec["time"], rec["qpos"]
    px = args.height
    m = load(args.model, 2 * px, px)
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
    rgba = m.geom_rgba.copy()
    for g, k in enumerate(kind):
        if k in SHADES:
            rgba[g, :3] = SHADES[k]

    # what the views have to show (metres): the cam's base disc, the magazines, the finished tower's cap
    ring = lambda r, z, n=24: [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z) for i in range(n)]
    low = ring(hx.RHO + 0.045, hx.ZT0 - hx.OVER - 0.04) + ring(hx.X_END + 0.02, -0.02) + ring(hx.X_END + 0.02, hx.H)
    cap = hx.level(tw.n) + hx.H + 0.03 + turns * hx.P
    views = []
    for cut, around, above, fades in VIEWS:
        points = low + (ring(hx.R_OUT + 0.03, cap) if cut is None else [(0, 0, cut)])
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        height, cam.distance, wide = aim(points, around, above, m.vis.global_.fovy)
        cam.lookat[:] = (0, 0, height)
        cam.azimuth, cam.elevation = around + 180, -above
        w = 2 * math.ceil(px * wide * 1.03 / 2)
        renderer = mujoco.Renderer(m, px, w)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        front = np.zeros(m.ngeom, bool)                        # the walls on the camera's side of the axis
        for g in range(m.ngeom if fades else 0):
            if kind[g] not in WALLS:
                continue
            if kind[g] == "head":                              # one body: each piece by where it is
                x, y = m.geom_pos[g][:2]
                if math.hypot(x, y) < 0.01:
                    continue                                   # the cap
                column = math.atan2(y, x)
            else:
                column = int(body[g][len(kind[g]):]) % tw.n * tw.theta
            front[g] = math.cos(column - math.radians(around)) > 0
        views.append((w, cam, renderer, front))
    width = sum(v[0] for v in views)

    def picture(k):
        """The scene at recorded frame k."""
        d.qpos[:] = qpos[k]
        mujoco.mj_forward(m, d)
        s = min(1.0, max(0.0, rotor[k] / (2 * math.pi * FADE)))
        s = s * s * (3 - 2 * s)                                # how far the walls in front have faded: eased, 0 to 1
        out = Image.new("RGB", (width, px), DIVIDER)
        x = 0
        for w, cam, renderer, front in views:
            m.geom_rgba[:] = rgba
            m.geom_rgba[front, 3] = 1 - s
            renderer.update_scene(d, cam)
            image = Image.fromarray(renderer.render())
            out.paste(image.crop((1, 0, w, px)) if x else image, (x + 1 if x else 0, 0))   # a line between the views
            x += w
        return out

    def close():                                               # before Python exits, or the GL library complains
        for view in views:
            view[2].close()

    if args.frame is not None:
        k = int(np.abs(times - args.frame).argmin())
        out = args.output if args.output.endswith(".png") else str(Path(args.output).with_suffix(".png"))
        picture(k).save(out)
        close()
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
        close()
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
