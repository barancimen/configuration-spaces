"""Render a recorded helix-tower build as an animated GIF (pip install mujoco pillow; needs ffmpeg).

    python3 examples/run_helixturm.py helixturm.xml --record run.npz
    MUJOCO_GL=egl python3 examples/render_helixturm.py helixturm.xml run.npz -o helixturm.gif

Two views side by side, drawn by MuJoCo's own renderer: the whole tower, and its base from closer
up. The walls of the columns that face the camera (their segments, magazines, guides and floor,
and that side of the head) are translucent, nearly clear in the close view, so that the cam, its
followers and the latch bolts inside show. MUJOCO_GL=egl draws without a display.
With --rewind the pictures then run backwards to the start, so the loop has no jump. That half is
the same recording in reverse order, not a simulation of the tower coming down.
"""
import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_helixturm as hx  # noqa: E402  (the dimensions)
from run_helixturm import Tower  # noqa: E402

BACKGROUND = (0.09, 0.10, 0.13)
INK, FAINT = (232, 234, 238), (150, 156, 168)
WALLS = ("panel", "stator", "push", "head")     # bodies that hide the mechanism
# per view: the height above the floor where the picture ends (None: over the finished tower), where
# the camera stands (degrees around the axis, degrees above the horizon), opacity of the walls
# that face it
VIEWS = {"whole": (None, 14, 9, 0.38), "base": (0.60, 14, 13, 0.13)}
CAPTIONS = {"whole": "the whole tower", "base": "its base, front walls clear"}


def load(path, width, height):
    spec = mujoco.MjSpec.from_file(path)
    spec.add_texture(name="backdrop", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=BACKGROUND, rgb2=BACKGROUND, width=8, height=48)
    m = spec.compile()
    m.vis.global_.offwidth, m.vis.global_.offheight = width, height
    m.vis.global_.fovy = 20
    m.vis.quality.offsamples = 4
    m.vis.headlight.ambient[:] = 0.42
    m.vis.headlight.diffuse[:] = 0.38
    m.vis.headlight.specular[:] = 0
    m.light_dir[:] = np.array([0.45, 0.55, -0.7]) / np.linalg.norm([0.45, 0.55, -0.7])   # from the upper left
    m.light_diffuse[:] = 0.45
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


def font(size):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="MJCF file written by cspace.mjcf from examples/helixturm.yaml")
    ap.add_argument("recording", help=".npz file written by run_helixturm.py --record")
    ap.add_argument("-o", "--output", default="helixturm.gif")
    ap.add_argument("--speed", type=float, default=5.0, help="times faster than real time (default 5)")
    ap.add_argument("--fps", type=int, default=25, help="pictures per second: 50, 25 or 20 play evenly in browsers")
    ap.add_argument("--height", type=int, default=640, help="picture height in pixels (default 640)")
    ap.add_argument("--rewind", action="store_true", help="then run the pictures backwards to the start")
    ap.add_argument("--pause", type=float, default=1.5, help="seconds to rest at the end (and at the start after --rewind)")
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

    # what the views have to show (metres): the cam's base disc, the magazines, the finished tower's cap
    ring = lambda r, z, n=24: [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z) for i in range(n)]
    low = ring(hx.RHO + 0.045, hx.ZT0 - hx.OVER - 0.10) + ring(hx.X_END + 0.02, -0.02) + ring(hx.X_END + 0.02, hx.H)   # and room for the captions
    cap = hx.level(tw.n) + hx.H + 0.03 + turns * hx.P
    views = []
    for name, (cut, around, above, alpha) in VIEWS.items():
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
        rgba = m.geom_rgba.copy()
        for g in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            kind = body.rstrip("0123456789")
            if kind not in WALLS:
                continue
            if kind == "head":                                 # one body: each piece by where it is
                x, y = m.geom_pos[g][:2]
                if math.hypot(x, y) < 0.01:
                    continue                                   # the cap
                column = math.atan2(y, x)
            else:
                column = int(body[len(kind):]) % tw.n * tw.theta
            if math.cos(column - math.radians(around)) > 0.3:  # this column faces the camera
                rgba[g, 3] = alpha
        views.append((name, w, cam, renderer, rgba))
    width = sum(v[1] for v in views)
    big, small = font(round(px / 34)), font(round(px / 46))
    pad = px / 40

    def picture(k):
        """The scene at recorded frame k, and what the status lines say about it."""
        d.qpos[:] = qpos[k]
        mujoco.mj_forward(m, d)
        out = Image.new("RGB", (width, px))
        draw = ImageDraw.Draw(out)
        x = 0
        for name, w, cam, renderer, rgba in views:
            m.geom_rgba[:] = rgba
            renderer.update_scene(d, cam)
            out.paste(Image.fromarray(renderer.render()), (x, 0))
            draw.text((x + pad, px - pad), CAPTIONS[name], font=small, fill=FAINT, anchor="ld")
            if x:
                draw.line([(x, 0), (x, px)], fill=(60, 64, 74), width=1)
            x += w
        placed = sum(1 for i in range(len(tw.panels)) if abs(tw.column(i)[0]) < 0.004)
        return out, (f"segment {placed} of {len(tw.panels)}", f"+{(d.xpos[tw.head][2] - head0) * 1000:.0f} mm")

    def status(image, lines, backwards=False):
        draw = ImageDraw.Draw(image)
        speed = f"{args.speed:g}× backwards" if backwards else f"{args.speed:g}× speed"
        draw.text((width - pad, px - pad - px / 26), lines[0], font=big, fill=INK, anchor="rd")
        draw.text((width - pad, px - pad), f"{lines[1]}  ·  {speed}", font=small, fill=FAINT, anchor="rd")
        return image

    def close():                                               # before Python exits, or the GL library complains
        for view in views:
            view[3].close()

    d.qpos[:] = qpos[0]
    mujoco.mj_forward(m, d)
    head0 = d.xpos[tw.head][2]
    if args.frame is not None:
        k = int(np.abs(times - args.frame).argmin())
        out = args.output if args.output.endswith(".png") else str(Path(args.output).with_suffix(".png"))
        status(*picture(k)).save(out)
        close()
        print(f"picture at {times[k]:.2f} s written to {out} ({width} x {px})")
        return 0

    step = args.speed / args.fps                               # simulated seconds between pictures
    ks = [int(np.abs(times - t).argmin()) for t in np.arange(times[0], times[-1] + step / 2, step)]
    rest = round(args.pause * args.fps)
    with tempfile.TemporaryDirectory() as tmp:
        if args.pictures:
            tmp = args.pictures
            Path(tmp).mkdir(parents=True, exist_ok=True)
        count = 0

        def put(image):
            nonlocal count
            image.save(f"{tmp}/f{count:05d}.png", compress_level=1)
            count += 1

        said = []
        for j, k in enumerate(ks):
            image, lines = picture(k)
            if args.rewind:
                image.save(f"{tmp}/plain{j:05d}.png", compress_level=1)
            said.append(lines)
            put(status(image, lines))
            if j % 25 == 0:
                print(f"  picture {j + 1} of {len(ks)}", flush=True)
        for _ in range(rest):
            put(image)
        close()
        if args.rewind:
            for j in reversed(range(len(ks))):
                image = status(Image.open(f"{tmp}/plain{j:05d}.png"), said[j], backwards=True)
                put(image)
            for _ in range(rest):
                put(image)
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
