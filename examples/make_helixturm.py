"""Write examples/helixturm.yaml: a helix tower that builds itself upward from stored segments.

    python3 examples/make_helixturm.py > examples/helixturm.yaml
    python3 examples/make_helixturm.py --lock > examples/helixturm_lock.yaml     # the joint, for the prover

After Konrad Zuse's Helixturm (HT1, 1993): a stator, a rotor with a control surface, vertically
moving cam followers with lifting arms that reach into cut-outs of the shell segments, radial
magazines, and identical segments that lock to each other along a helix. One motor turns the rotor.

The machine (metres; z up, the tower axis is z, the floor is z = 0; column k is at angle 2*pi*k/N):
  rotor      a staircase cam: per revolution N flats joined by N short ramps, then one steep return
  lift k     cam follower of column k: slides vertically, its foot rides the cam
  bolt k     the lifting arm: a spring-loaded latch bolt on the lifter, flat on top, bevelled below.
             It reaches into a segment's window and lifts it by the lintel; on the way down the
             bevel pushes it back so it passes the next segment and snaps into that one's window
  push k     magazine follower: a spring pushes the queue of segments toward the tower
  stator k   floor, inner guide pads and magazine guides of column k
  panel i    a shell segment (free body): two legs and a lintel around a window; a finger with a
             pin at each end; a slot at its right end and one at its lower left corner
  head       the tower's top: one turn of segment stubs under a cap. Its underside is the helix.
One step (1/N of a revolution): all bolts are at rest; the column whose bottom has reached one
segment height gets the next segment pushed under it; its lifter sets the column down on that
segment, drops to it and latches; then all N lifters rise by 1/N of a segment height. Segment i
goes into column i mod N, so each one sits 1/N of a height below the one before it: a helix.
Segment i slides in radially, and its two pins enter two slots that run the same way: one in
segment i-1 beside it, one in segment i-N+1 above its other end. That alone lets it slide back
out, or swing about either end. Two later segments then pin it, each coming in along its own
radial direction, turned by 2*pi/N either way: segment i+1 puts a pin into the slot at its right
end, and segment i+N-1 one into the slot at its lower left corner. Until then the magazine's
queue holds it against the guide pads. examples/helixturm_lock.yaml states this as claims for
the prover. Free bodies, contacts, springs and the motor are simulated in MuJoCo.
"""
import argparse
import math

# ---------- dimensions ----------
N = 6                      # segments per turn (columns)
TURNS = 3                  # segments per magazine: the tower grows by this many turns
P = 0.24                   # segment height, and the tower's rise per revolution
DROP = 0.001               # a column waits this far above its new segment, and is then set down on it
H = P
TH = 0.030                 # wall thickness
R_IN = 0.170               # axis to the inner face of a wall
WIN, ZL = 0.060, 0.150     # window width; lintel underside above the segment's bottom
GC = 0.001                 # gap between a wall's end and the inner corner

PIN_R, PIN_A, PIN_B = 0.0035, 0.015, 0.009   # pin radius; position: a from the inner face, b left of the corner
CH = 0.0002                                  # pin to rail, each side
Z_PIN, Z_FING = (0.122, 0.150), (0.150, 0.168)
B_DN, B_UP = 0.0018, 0.0018                  # finger to rail tops below, to the roof above (more than DROP)
Z_PIN2, Z_FING2 = (0.203, 0.218), (0.218, 0.234)   # the second pin, at the right end near the top

G_PAD, G_SIDE, G_Q = 0.0003, 0.0006, 0.0004
EB, B_RET = 0.016, 0.019   # bolt reach into the wall; bolt travel
OVER = 0.008               # the lifter drops this far below a lintel before it comes back up to it
RHO, RF, ROD = 0.130, 0.012, 0.50            # cam radius, foot radius, bolt top to foot centre
# Fractions of a step, as the cam is cut. The foot is a ball, so it reaches the bottom of the steep
# return about 0.08 of a step after the surface does.
U_INS, U_BOT, U_PRE, DWELL = 0.19, 0.31, 0.43, 0.50   # return starts, its foot, slack taken up from here, rise starts

COLORS = {"stator": "#46525f", "rotor": "#b8862b", "lift": "#8a96a3", "bolt": "#8d3d4a",
          "push": "#6c5c9e", "head": "#3f6f9f"}
TURN_COLORS = ["#3f8a78", "#a3592e", "#5b7f2e", "#2e7f9b", "#7d6f4e", "#8d3d4a"]


def derived():
    g = globals()
    g["TH_"] = 2 * math.pi / N                       # angle between columns
    g["R_OUT"], g["R_MID"] = R_IN + TH, R_IN + TH / 2
    g["YC"] = R_IN * math.tan(math.pi / N)           # inner corner, tangential distance from the middle
    g["W"] = 2 * (YC - GC)                           # wall width
    g["STEP"] = P / N
    g["B_MAX"] = min(TH * math.tan(TH_ / 2) - 0.0001, 0.020)   # the slot stays inside both walls' thickness
    g["X_END"] = R_OUT + TURNS * (TH + G_Q) + 0.07   # outer end of a magazine
    g["Z_FOOT"] = ZL - ROD
    g["ZT0"] = Z_FOOT - RF                           # cam surface height for lifter level 0


def num(v):
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def vec(v):
    return "[" + ", ".join(num(c) for c in v) + "]"


def rotz(p, ang):
    c, s = math.cos(ang), math.sin(ang)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])


def sub(p, q):
    return tuple(a - b for a, b in zip(p, q))


def place(centre, ang, origin):
    """Placement keys for a shape given in a frame turned by `ang` about the tower axis, written in a
    world-aligned body frame whose origin is `origin`."""
    out = f"at: {vec(sub(rotz(centre, ang), origin))}"
    if abs(ang) > 1e-12:
        out += f", x: {vec((math.cos(ang), math.sin(ang), 0))}"
    return out


def box(x, y, z, ang=0.0, origin=(0, 0, 0)):
    size = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    assert min(size) > 0, (x, y, z)
    centre = ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2)
    return f"{{box: {vec(size)}, {place(centre, ang, origin)}}}"


def cyl(r, z, centre_xy, ang=0.0, origin=(0, 0, 0)):
    centre = (centre_xy[0], centre_xy[1], (z[0] + z[1]) / 2)
    return f"{{cylinder: [{num(r)}, {num(z[1] - z[0])}], at: {vec(sub(rotz(centre, ang), origin))}}}"


def hull(points, ang=0.0, origin=(0, 0, 0)):
    return "{hull: [" + ", ".join(vec(sub(rotz(p, ang), origin)) for p in points) + "]}"


def left(b):
    """y of a point b to the left of a wall's inner corner."""
    return -(YC + b)


def key_boxes(bottom):
    """The finger at a segment's left end, in its own column's coordinates: (x, y, z) ranges."""
    return [((R_IN + 0.011, R_IN + 0.025), (left(PIN_B + PIN_R), left(-0.004)), (bottom + Z_FING[0], bottom + Z_FING[1]))]


def channel(z, flip=1):
    """Two rails and an inner end stop around a pin's path, as (x, y, z) ranges. flip = -1 mirrors
    them to the right-hand corner."""
    side = lambda lo, hi: (left(hi), left(lo)) if flip == 1 else (-left(lo), -left(hi))
    near, far = side(0.0004, PIN_B - PIN_R - CH), side(PIN_B + PIN_R + CH, B_MAX)
    both = side(0.0004, B_MAX)
    stop = R_IN + PIN_A - PIN_R - 0.0004                       # the pin can go this far in
    return [((stop, R_OUT), near, z),                          # rail next to the wall
            ((stop, R_OUT), far, z),                           # rail on the far side
            ((stop - 0.0045, stop), both, z)]                  # inner end stop


def slot_boxes(bottom):
    """The slot at a segment's right end, in the next column's coordinates. The next segment stands
    one step lower, so the slot's heights are measured from there."""
    zb = bottom - STEP
    roof = ((R_IN + 0.0105, R_OUT), (left(B_MAX), left(0.0004)), (zb + Z_FING[1] + B_UP, zb + 0.184))
    return channel((zb + 0.119, zb + Z_FING[0] - B_DN)) + [roof]


def key2_boxes(bottom):
    """The finger of the second pin, at a segment's right end, in its own column's coordinates."""
    return [((R_IN + 0.011, R_IN + 0.025), (-left(-0.004), -left(PIN_B + PIN_R)), (bottom + Z_FING2[0], bottom + Z_FING2[1]))]


def slot2_boxes(bottom):
    """The slot at a segment's lower left corner for the second pin of the segment that comes in
    below the left neighbour: in that (previous) column's coordinates. That segment stands a whole
    height less one step lower."""
    zb = bottom - (P - STEP)
    return channel((zb + Z_PIN2[0] - 0.0015, zb + Z_FING2[0] - 0.0035), flip=-1)


def segment(ang, bottom, origin, keys=True, slot=True, slot2=True):
    """Shapes of a segment in the column at angle `ang` with its bottom at height `bottom`.
    Column coordinates: x radial from the axis, y tangential, z up."""
    b0 = bottom
    wall = (R_IN, R_OUT)
    out = [box(wall, (-W / 2, -WIN / 2), (b0, b0 + H), ang, origin),          # left leg
           box(wall, (WIN / 2, W / 2), (b0, b0 + H), ang, origin),            # right leg
           # the lintel over the window, set back half a millimetre so that only the legs touch neighbours
           box((R_IN + 0.0005, R_OUT - 0.0005), (-WIN / 2, WIN / 2), (b0 + ZL, b0 + H - 0.0005), ang, origin)]
    if keys:                          # both pins slide in this column's own radial direction
        out += [box(*r, ang, origin) for r in key_boxes(b0)]
        out.append(cyl(PIN_R, (b0 + Z_PIN[0], b0 + Z_PIN[1]), (R_IN + PIN_A, left(PIN_B)), ang, origin))
        out += [box(*r, ang, origin) for r in key2_boxes(b0)]
        out.append(cyl(PIN_R, (b0 + Z_PIN2[0], b0 + Z_PIN2[1]), (R_IN + PIN_A, -left(PIN_B)), ang, origin))
    if slot:                          # at the right end, in the next column's directions
        out += [box(*r, ang + TH_, origin) for r in slot_boxes(b0)]
    if slot2:                         # at the lower left corner, in the previous column's directions
        out += [box(*r, ang - TH_, origin) for r in slot2_boxes(b0)]
    return out


def level(s):
    """Lifter level at rest on flat s of the cam. Level 0 is a bolt touching the lintel of a segment
    that stands on the floor. From the first rise on, the column is carried DROP higher than a
    plain staircase, which makes up for its having been set down by DROP; at level N it waits
    that far above its next segment."""
    return s * STEP + DROP if s else 0.0


def cam():
    """The rotor's shapes. A follower at phase u (rotor angle minus its column's angle, in steps)
    stands over the rotor's own angle -u. Each piece is a box whose top face joins two points of
    the profile; flats are solid down to the base disc."""
    base = ZT0 - OVER - 0.02
    top = level(N)
    pieces = [(0, top, U_INS, top, "flat"),                    # the column waits for its segment
              (U_INS, top, U_BOT, -OVER, "ramp"),              # steep return, down past the next lintel
              (U_BOT, -OVER, U_PRE, -OVER, "flat"),            # the bolt snaps into the window
              (U_PRE, -OVER, DWELL, 0, "ramp"),                # slack taken up: the bolt meets the lintel
              (DWELL, 0, 1, level(1), "join")]                 # first rise, with the others (and DROP more)
    for s in range(1, N):
        pieces.append((s, level(s), s + DWELL, level(s), "flat"))
        pieces.append((s + DWELL, level(s), s + 1, level(s + 1), "ramp"))
    out = [f"{{cylinder: [{num(RHO + 0.045)}, 0.02], at: {vec((0, 0, base - 0.01))}}}"]
    for u1, z1, u2, z2, kind in pieces:
        p1 = (RHO * math.cos(-u1 * TH_), RHO * math.sin(-u1 * TH_), ZT0 + z1)
        p2 = (RHO * math.cos(-u2 * TH_), RHO * math.sin(-u2 * TH_), ZT0 + z2)
        d = sub(p2, p1)
        L = math.sqrt(sum(c * c for c in d))
        g = tuple(c / L for c in d)
        n = tuple((1.0 if i == 2 else 0.0) - g[2] * g[i] for i in range(3))
        nl = math.sqrt(sum(c * c for c in n))
        n = tuple(c / nl for c in n)
        thick = (ZT0 + z1 - base) if kind == "flat" else 0.012
        ext = 0.004 if kind == "ramp" else 0.0  # a ramp next to a flat reaches a little past its lower end
        sign = -1 if z1 <= z2 else 1            # g points from p1 to p2; the lower end is p1 when rising
        mid = tuple((a + b) / 2 + sign * g[i] * ext / 2 for i, (a, b) in enumerate(zip(p1, p2)))
        centre = tuple(m - n[i] * thick / 2 for i, m in enumerate(mid))
        out.append(f"{{box: {vec((L + ext, 0.044, thick))}, at: {vec(centre)}, z: {vec(n)}, x: {vec(g)}}}")
    return out


def write_body(w, name, color, shapes, frames=None):
    w(f"  {name}:")
    w(f'    color: "{color}"')
    if frames:
        w(f"    frames: {frames}")
    w("    shape:")
    w("      union:")
    for s in shapes:
        w(f"        - {s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--columns", type=int, default=N, help="segments per turn (4 to 6)")
    ap.add_argument("--turns", type=int, default=TURNS, help="segments per magazine")
    ap.add_argument("--lock", action="store_true", help="write the small file that proves the lock instead")
    args = ap.parse_args()
    globals()["N"], globals()["TURNS"] = args.columns, args.turns
    if not 4 <= N <= 6:
        raise SystemExit("the slot fits in the corner between two walls for 4 to 6 columns only")
    derived()
    if args.lock:
        return lock_file()
    out = []
    w = out.append
    body = __doc__.split("\n\n", 2)[2].strip()
    w("# Helix tower: generated by examples/make_helixturm.py (edit that, then rerun it).\n# "
      + body.replace("\n", "\n# "))
    w("name: helixturm")
    w("bodies:")
    w("  world:")
    w("    frames:")
    r_push = R_OUT + G_Q + TURNS * (TH + G_Q) + 0.006            # middle of a follower plate at the start
    for k in range(N):
        a = f"2*pi*{k}/{N}"
        w(f'      col{k}: {{x: ["cos({a})", "sin({a})", 0]}}')
        w(f'      mag{k}: {{at: ["{num(r_push)}*cos({a})", "{num(r_push)}*sin({a})", 0.11], '
          f'z: ["-cos({a})", "-sin({a})", 0], x: ["-sin({a})", "cos({a})", 0]}}')
    write_body(w, "rotor", COLORS["rotor"], cam())

    for k in range(N):
        pads = [box((R_IN - G_PAD - 0.015, R_IN - G_PAD), y, (0, 1.5 * P)) for y in ((-0.08, -0.05), (0.05, 0.08))]
        # magazine guides beside the walls' ends, at heights where a segment's ends carry nothing
        rails = [box((R_OUT + 0.02, X_END), (-W / 2 - G_SIDE - 0.012, -W / 2 - G_SIDE), (0.025, 0.075)),
                 box((R_OUT + 0.02, X_END), (W / 2 + G_SIDE, W / 2 + G_SIDE + 0.012), (0, 0.07))]
        write_body(w, f"stator{k}", COLORS["stator"],
                   [box((R_IN - 0.018, X_END), (-W / 2 - 0.02, W / 2 + 0.02), (-0.02, 0))] + pads + rails)
        write_body(w, f"lift{k}", COLORS["lift"],
                   [box((RHO - 0.008, RHO + 0.008), (-0.008, 0.008), (Z_FOOT, ZL + 0.02)),        # rod
                    box((0.09, 0.148), (-0.03, 0.03), (ZL - 0.19, ZL - 0.05))],                   # bolt housing
                   "{b: {z: [1, 0, 0], x: [0, 1, 0]}}")
        w(f"  foot{k}:")
        w(f'    color: "{COLORS["lift"]}"')
        w(f"    shape: {{ball: {num(RF)}, at: {vec((RHO, 0, Z_FOOT))}}}")
        tip, back, low = R_IN + EB, 0.150, ZL - 0.005 - (EB + 0.004) * math.tan(math.radians(60))
        wedge = [(x, y, z) for y in (-0.02, 0.02) for x, z in
                 ((back, ZL - 0.0002), (tip, ZL - 0.0002), (tip, ZL - 0.005), (R_IN - 0.004, low), (back, low))]
        write_body(w, f"bolt{k}", COLORS["bolt"],
                   [box((back, tip), (-0.02, 0.02), (ZL - 0.004, ZL)), hull(wedge)],
                   "{b: {z: [1, 0, 0], x: [0, 1, 0]}}")
        write_body(w, f"push{k}", COLORS["push"],
                   # the plate is nearly as tall as a segment: the queue leans on the rising column by a
                   # strip at the top, and the plate has to be able to push there, or the queue tips
                   [box((-0.006, 0.006), (-0.08, 0.08), (0.01 - 0.11, H - 0.0025 - 0.11)),
                    box((0.006, 0.05), (-0.01, 0.01), (-0.01, 0.01))],
                   "{a: {z: [-1, 0, 0], x: [0, 1, 0]}}")

    # the head: stubs of one turn (their bottoms are where the lifters stand at the start) and a cap
    head = []
    for k in range(N):
        bottom = level((N - k) % N or N)
        head += segment(k * TH_, bottom, (0, 0, 0), keys=False, slot=(k == N - 1), slot2=(k > 0))
        if bottom < level(N):
            head.append(box((R_IN, R_OUT), (-W / 2, W / 2), (bottom + H, level(N) + H), k * TH_))
    head.append(f"{{cylinder: [{num(R_OUT + 0.03)}, 0.03], at: {vec((0, 0, level(N) + H + 0.015))}}}")
    write_body(w, "head", COLORS["head"], head)

    panels = N * TURNS
    start = {}
    for i in range(panels):
        k, j = i % N, i // N
        x = R_OUT + G_Q + j * (TH + G_Q) + TH / 2            # waiting in the magazine
        home = rotz((R_MID, 0, H / 2), k * TH_)                # the body's origin is the segment's centre
        write_body(w, f"panel{i}", TURN_COLORS[j % len(TURN_COLORS)], segment(k * TH_, 0, home))
        start[f"panel{i}"] = rotz((x, 0, H / 2), k * TH_)

    w("constraints:")
    w("  - revolute: [world, rotor]\n    name: rotor")
    travel = TURNS * (TH + G_Q)
    for k in range(N):
        w(f"  - fixed: [world.col{k}, stator{k}]")
        w(f"  - prismatic: [world.col{k}, lift{k}]\n    name: lift{k}")
        w(f"  - fixed: [lift{k}, foot{k}]")
        w(f"  - prismatic: [lift{k}.b, bolt{k}.b]\n    name: bolt{k}\n    range: [{num(-B_RET)}, 0]\n"
          "    spring: {stiffness: 20, rest: 0.03, damping: 1}")
        w(f"  - prismatic: [world.mag{k}, push{k}.a]\n    name: push{k}\n    range: [-0.002, {num(travel)}]\n"
          f"    spring: {{stiffness: 18, rest: {num(travel + 0.8)}, damping: 10}}")
    w("  # simulation only: the bodies that push on each other")

    def contact(a, b, friction):
        w(f"  - contact: [{a}, {b}]\n    friction: {friction}")

    for k in range(N):
        contact(f"foot{k}", "rotor", 0.02)
        contact("head", f"bolt{k}", 0.2)
        contact("head", f"stator{k}", 0.2)
        contact(f"panel{k + N * (TURNS - 1)}", f"push{k}", 0.2)
    for i in range(panels):
        k = i % N
        contact(f"panel{i}", f"stator{k}", 0.2)
        contact(f"panel{i}", f"bolt{k}", 0.2)
        if i < N:
            contact(f"panel{i}", "head", 0.2)
        for other in (i + 1, i + N - 1, i + N):               # next on the helix, below-left, below
            if other < panels:
                contact(f"panel{i}", f"panel{other}", 0.2)

    w("\nstart:")
    w("  rotor: 0")
    for k in range(N):
        w(f"  lift{k}: {num(level((N - k) % N or N))}")
        w(f"  bolt{k}: 0")
        w(f"  push{k}: 0")
    w("  head: [0, 0, 0]")
    for name, c in start.items():
        w(f"  {name}: {vec(c)}")
    w(f"""
simulation:
  timestep: 0.001
  density: 600
  solref: [0.004, 1]
  motors:
    rotor: {{kp: 3000, damping: 25}}
#   python3 -m cspace.mjcf examples/helixturm.yaml -o helixturm.xml
#   python3 examples/run_helixturm.py helixturm.xml""")
    print("\n".join(out))


def lock_file():
    """The joints between segments, alone, with exact numbers, as claims for the prover."""

    def exact(v):
        return num(v) if abs(v) > 5e-7 else "0"

    def turned(x, y, cols):
        """Exact coordinates of the point (x, y) of a frame turned by `cols` columns."""
        t = f"{cols}*2*pi/{N}"
        return f'"{exact(x)}*cos({t}) - ({exact(y)})*sin({t})", "{exact(x)}*sin({t}) + ({exact(y)})*cos({t})"'

    def xbox(r, cols, shift):
        """A box given by ranges in the frame of the column `cols` steps on, moved up by `shift`."""
        (x0, x1), (y0, y1), (z0, z1) = r
        cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2 + shift
        size = vec((x1 - x0, y1 - y0, z1 - z0))
        if cols == 0:
            return f"{{box: {size}, at: {vec((cx, cy, cz))}}}"
        t = f"{cols}*2*pi/{N}"
        return f'{{box: {size}, at: [{turned(cx, cy, cols)}, {num(cz)}], x: ["cos({t})", "sin({t})", 0]}}'

    def pin(cols, z, shift, right=False):
        """A pin as an eight-sided prism that lies inside the round pin (so it allows more motion)."""
        a, b = 0.0034, 0.0008
        assert a * a + b * b <= PIN_R * PIN_R
        cx, cy = R_IN + PIN_A, (-left(PIN_B) if right else left(PIN_B))
        pts = []
        for dx, dy in ((a, b), (b, a), (-b, a), (-a, b), (-a, -b), (-b, -a), (b, -a), (a, -b)):
            for zz in (z[0] + shift, z[1] + shift):
                x, y = cx + dx, cy + dy
                pts.append(f"[{num(x)}, {num(y)}, {num(zz)}]" if cols == 0 else f"[{turned(x, y, cols)}, {num(zz)}]")
        return "{hull: [" + ", ".join(pts) + "]}"

    out = []
    w = out.append

    def body(name, note, shapes, frames=None):
        w(f"  {name}:{' ' * max(1, 11 - len(name))}# {note}")
        if frames:
            w(f"    frames: {frames}")
        w("    shape:\n      union:")
        for sh in shapes:
            w(f"        - {sh}")

    t = f"2*pi/{N}"
    px, py = R_IN + PIN_A, left(PIN_B)               # a left pin's centre in its own column's frame
    w(f"""# The lock between helix-tower segments: generated by examples/make_helixturm.py --lock.
# Segment i has a pin at each end; both slide radially into slots of earlier segments (i-1 beside
# it, i-N+1 above its right end). On their own these two pins only keep it from moving along the
# wall: it can still slide out, and swing about either end. What holds it are two later segments,
# each of which comes in along its own radial direction, turned by 2*pi/{N} either way:
#   segment i+1    puts its left pin into the slot at segment i's right end
#   segment i+N-1  puts its right pin into the slot at segment i's lower left corner
# Two models of segment i, with the dimensions and clearances of helixturm.yaml.
#   solid   the joints' real shapes (fingers, pins, rails, end stops, the roof). Segment i moves
#           without turning (out, sideways, up); segments i+1 and i+N-1 may slide along their own
#           radial directions. Each round pin is replaced by an eight-sided one inside it, which
#           can only allow more motion.
#   flat    segment i moves in the plane and may turn. A round pin between two rails is a point
#           (its centre) in a channel as wide as the clearance; the claims state the four
#           channels as conditions. (With the real shapes and turning, z3 does not finish.)
# The 'held' claims say: within some neighbourhood of its place the segment is in a much smaller
# box. No pose in between is free of overlap, so it cannot leave the small box by moving.
#   python3 -m cspace examples/helixturm_lock.yaml --prove
name: helixturm_lock
bodies:
  world:
    frames:
      mid: {{at: [{num(R_MID)}, 0, 0]}}                # the middle of segment i's wall
      sx: {{z: [1, 0, 0]}}
      nextRad: {{z: ["cos({t})", "sin({t})", 0]}}      # the radial direction of segment i+1
      lowRad: {{z: ["cos(-{t})", "sin(-{t})", 0]}}     # ... and of segment i+N-1
      nextPin: {{at: [{turned(px, py, 1)}, 0]}}        # where the left pin of segment i+1 stands
      lowPin: {{at: [{turned(px, -py, -1)}, 0]}}       # where the right pin of segment i+N-1 stands
  flat:                 # segment i in the plane: frames only
    frames:
      mid: {{at: [{num(R_MID)}, 0, 0]}}
      pinL: {{at: [{num(px)}, {num(py)}, 0]}}          # its two pins
      pinR: {{at: [{num(px)}, {num(-py)}, 0]}}
      slot: {{at: [{turned(px, py, 1)}, 0], z: ["-sin({t})", "cos({t})", 0]}}          # its right slot: z across it
      slotAlong: {{at: [{turned(px, py, 1)}, 0], z: ["cos({t})", "sin({t})", 0]}}      # ... and z along it, outward
      corner: {{at: [{turned(px, -py, -1)}, 0], z: ["sin({t})", "cos({t})", 0]}}       # its corner slot, likewise
      cornerAlong: {{at: [{turned(px, -py, -1)}, 0], z: ["cos({t})", "-sin({t})", 0]}}
  carX: {{frames: {{sx: {{z: [1, 0, 0]}}, sy: {{z: [0, 1, 0]}}}}}}     # carriers: they only chain the slides
  carY: {{frames: {{sy: {{z: [0, 1, 0]}}}}}}""")
    body("prev", "fixed: the slot of segment i-1 (beside it, one step higher)", [xbox(r, 0, 0) for r in slot_boxes(STEP)])
    body("above", "fixed: the corner slot of segment i-N+1 (above its right end)", [xbox(r, 0, 0) for r in slot2_boxes(P - STEP)])
    body("keyL", "segment i: left finger and pin", [xbox(r, 0, 0) for r in key_boxes(0)] + [pin(0, Z_PIN, 0)])
    body("keyR", "segment i: right finger and pin", [xbox(r, 0, 0) for r in key2_boxes(0)] + [pin(0, Z_PIN2, 0, right=True)])
    body("slot", "segment i: the slot at its right end", [xbox(r, 1, 0) for r in slot_boxes(0)])
    body("corner", "segment i: the slot at its lower left corner", [xbox(r, -1, 0) for r in slot2_boxes(0)])
    body("next", "segment i+1: its left finger and pin (one step lower)",
         [xbox(r, 1, 0) for r in key_boxes(-STEP)] + [pin(1, (Z_PIN[0] - STEP, Z_PIN[1] - STEP), 0)],
         f'{{rad: {{z: ["cos({t})", "sin({t})", 0]}}}}')
    low = -(P - STEP)
    body("low", "segment i+N-1: its right finger and pin (below segment i-1)",
         [xbox(r, -1, 0) for r in key2_boxes(low)] + [pin(-1, (Z_PIN2[0] + low, Z_PIN2[1] + low), 0, right=True)],
         f'{{rad: {{z: ["cos(-{t})", "sin(-{t})", 0]}}}}')
    w("""constraints:
  # solid
  - fixed: [world, prev]
  - fixed: [world, above]
  - prismatic: [world.sx, carX.sx]      # out of the tower
    name: out
  - prismatic: [carX.sy, carY.sy]       # along the wall
    name: side
  - prismatic: [carY, keyL]             # up
    name: up
  - fixed: [keyL, keyR]
  - fixed: [keyL, slot]
  - fixed: [keyL, corner]
  - prismatic: [world.nextRad, next.rad]
    name: next
  - prismatic: [world.lowRad, low.rad]
    name: low
  - no_overlap: [keyL, prev]
  - no_overlap: [keyR, above]
  - no_overlap: [slot, next]
  - no_overlap: [corner, low]
  # flat
  - planar: [world.mid, flat.mid]       # x: out of the tower, y: along the wall, and the turn
    name: seg""")
    c, inn = num(CH), num(0.0004)                     # pin to rail; pin to the inner end stop
    own = (f"-{c} <= pos(flat.pinL).y - ({num(py)}) <= {c} and pos(flat.pinL).x >= {num(px - 0.0004)} and "
           f"-{c} <= pos(flat.pinR).y - ({num(-py)}) <= {c} and pos(flat.pinR).x >= {num(px - 0.0004)}")
    nxt = (f"-{c} <= dot(pos(world.nextPin) - pos(flat.slot), axis(flat.slot)) <= {c} and "
           f"dot(pos(world.nextPin) - pos(flat.slot), axis(flat.slotAlong)) >= -{inn}")
    lowp = (f"-{c} <= dot(pos(world.lowPin) - pos(flat.corner), axis(flat.corner)) <= {c} and "
            f"dot(pos(world.lowPin) - pos(flat.corner), axis(flat.cornerAlong)) >= -{inn}")
    w(f"""
claims:
  # solid: within 6 mm of its place in every direction, with the pin of segment i+1 at most 6 mm
  # from its place, segment i is within a millimetre (two millimetres up and down)
  - name: held
    always: "implies(-0.006 <= out.d <= 0.006 and -0.006 <= side.d <= 0.006 and -0.006 <= up.d <= 0.006 and -0.006 <= next.d <= 0.006, -0.001 <= out.d <= 0.001 and -0.001 <= side.d <= 0.001 and -0.002 <= up.d <= 0.002)"
  - name: comesOutLast                  # with both later segments drawn out, it slides straight out
    possible: "next.d >= 0.03 and low.d >= 0.03 and out.d >= 0.03 and side.d == 0 and up.d == 0"
  - name: neverComesOut                 # false on purpose: the prover should find the way out
    always: "out.d <= 0.002"
  # flat: turned less than 25 degrees, with all four pins in their channels, segment i is within
  # 0.6 mm and a third of a degree (one claim per bound: z3 decides them much faster apart)
  - name: pinnedOutward
    always: "implies(seg.c >= 9/10 and {own} and {nxt} and {lowp}, seg.x <= 0.0006)"
  - name: pinnedInward
    always: "implies(seg.c >= 9/10 and {own} and {nxt} and {lowp}, seg.x >= -0.0006)"
  - name: pinnedSideways
    always: "implies(seg.c >= 9/10 and {own} and {nxt} and {lowp}, -0.0004 <= seg.y <= 0.0004)"
  - name: pinnedTurningOneWay
    always: "implies(seg.c >= 9/10 and {own} and {nxt} and {lowp}, seg.s <= 0.005)"
  - name: pinnedTurningOtherWay
    always: "implies(seg.c >= 9/10 and {own} and {nxt} and {lowp}, seg.s >= -0.005)"
  - name: swingsWithoutLowerPin         # before segment i+N-1 is in, the left end can swing out by 6 mm
    possible: "seg.c >= 9/10 and {own} and {nxt} and seg.s >= 0.03"
  - name: slidesOnItsOwnPins            # and with neither later segment, it slides straight out
    possible: "{own} and seg.x >= 0.03 and seg.y == 0 and seg.s == 0 and seg.c == 1"
""".rstrip())
    print("\n".join(out))


if __name__ == "__main__":
    main()
