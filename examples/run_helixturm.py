"""Build the helix tower in MuJoCo and check the result (pip install mujoco).

    python3 -m cspace.mjcf examples/helixturm.yaml -o helixturm.xml
    python3 examples/run_helixturm.py helixturm.xml              # all segments, then hold still
    python3 examples/run_helixturm.py helixturm.xml --steps 7    # the first seven segments only
    python3 examples/run_helixturm.py helixturm.xml --view       # watch it (needs a display)

The only input is the rotor's motor, which turns at a steady rate. One step is 1/N of a
revolution and adds one segment. After the last step the rotor stops and the tower stands on the
lifting bolts for --hold seconds; --push then presses sideways on the head for a second.
The script prints one line per step and a final verdict, and exits 0 only if every segment ended
in its place, the head rose by the right amount and nothing moved while the rotor was stopped.
It also compares the motor's work with the height the parts gained.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_helixturm as hx  # noqa: E402  (the dimensions)


def names(m, kind, prefix):
    out = []
    for i in range(getattr(m, {"body": "nbody", "joint": "njnt"}[kind])):
        n = mujoco.mj_id2name(m, {"body": mujoco.mjtObj.mjOBJ_BODY, "joint": mujoco.mjtObj.mjOBJ_JOINT}[kind], i)
        if n and n.startswith(prefix) and n[len(prefix):].isdigit():
            out.append(n)
    return sorted(out, key=lambda n: int(n[len(prefix):]))


class Tower:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.n = len(names(m, "joint", "lift"))
        self.panels = names(m, "body", "panel")
        hx.N, hx.TURNS = self.n, len(self.panels) // self.n
        hx.derived()
        self.theta = 2 * math.pi / self.n
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.head, self.pid = bid("head"), [bid(p) for p in self.panels]
        jadr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        self.q_rotor = jadr("rotor")
        self.q_lift = [jadr(f"lift{k}") for k in range(self.n)]
        self.q_bolt = [jadr(f"bolt{k}") for k in range(self.n)]
        self.q_push = [jadr(f"push{k}") for k in range(self.n)]
        self.head0 = None

    def column(self, i):
        """Segment i in its column's coordinates: (radial - home, tangential, bottom height, tilt in rad)."""
        k = i % self.n
        a = k * self.theta
        p, R = self.d.xpos[self.pid[i]], self.d.xmat[self.pid[i]].reshape(3, 3)
        r = p[0] * math.cos(a) + p[1] * math.sin(a) - hx.R_MID
        t = -p[0] * math.sin(a) + p[1] * math.cos(a)
        return r, t, p[2] - hx.H / 2, math.acos(min(1.0, max(-1.0, R[2, 2])))

    def head_state(self):
        p, R = self.d.xpos[self.head], self.d.xmat[self.head].reshape(3, 3)
        return p.copy(), math.acos(min(1.0, max(-1.0, R[2, 2])))

    def expected_bottom(self, i, steps_done):
        """Bottom height of segment i once `steps_done` segments are in and the rotor rests at the
        start of the next step: its column stands on the bolt under its newest segment, and the
        older segments are stacked on that one."""
        newest = i + self.n * ((steps_done - 1 - i) // self.n)
        return hx.level(steps_done - newest) + (newest - i) // self.n * hx.H


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="MJCF file written by cspace.mjcf from examples/helixturm.yaml")
    ap.add_argument("--steps", type=int, default=None, help="segments to add (default: all of them)")
    ap.add_argument("--period", type=float, default=1.6, help="seconds per step (default 1.6)")
    ap.add_argument("--hold", type=float, default=2.0, help="seconds to stand still at the end")
    ap.add_argument("--push", type=float, default=0.0, help="then press sideways on the head with this force, in newtons")
    ap.add_argument("--trace", type=float, default=0.0, help="also print the state every this many seconds")
    ap.add_argument("--trace-from", type=float, default=0.0, help="start the trace at this time")
    ap.add_argument("--record", help="save the motion (time and qpos, 25 frames a second) to this .npz file")
    ap.add_argument("--view", action="store_true", help="open MuJoCo's interactive viewer")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "sample")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    tw = Tower(m, d)
    n, theta = tw.n, tw.theta
    steps = len(tw.panels) if args.steps is None else min(args.steps, len(tw.panels))
    omega = theta / args.period
    settle = 0.3                                   # the first segment goes in before the rotor starts
    alpha_end = (steps + 0.08) * theta             # rest early in the next step, before its lifter returns
    t_stop = settle + alpha_end / omega
    head0, _ = tw.head_state()
    rotor_dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "rotor")]
    print(f"{n} columns, {len(tw.panels)} segments in the magazines, total mass {m.body_mass.sum():.1f} kg; "
          f"adding {steps} segments, one every {args.period:g} s")

    potential = lambda: float(9.81 * (m.body_mass * d.xipos[:, 2]).sum())
    st = dict(frames=[], next_frame=0.0, next_trace=0.0, next_step=1, work=0.0, peak=0.0,
              ref=None, all=None, moved=0.0, part=0, far=0.0)
    bearing = float(m.dof_damping[rotor_dof])      # the servo's own damping, on the rotor's axis
    energy0, wall0 = potential(), time.time()
    viewer = None
    if args.view:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d)

    def advance(until, force=0.0):
        """Simulate up to time `until`, with a sideways force on the head. False if the viewer was closed."""
        while d.time < until:
            d.ctrl[0] = min(max(0.0, omega * (d.time - settle)), alpha_end)
            d.xfrc_applied[tw.head] = 0
            d.xfrc_applied[tw.head, 0] = force
            mujoco.mj_step(m, d)
            torque = float(d.actuator_force[0]) - bearing * d.qvel[rotor_dof]    # what reaches the cam
            st["work"] += torque * d.qvel[rotor_dof] * m.opt.timestep
            st["peak"] = max(st["peak"], abs(torque))
            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    return False
            if args.record and d.time >= st["next_frame"]:
                st["frames"].append((d.time, d.qpos.copy()))
                st["next_frame"] += 0.04
            if args.trace and d.time >= max(st["next_trace"], args.trace_from):
                st["next_trace"] = d.time + args.trace
                a = d.qpos[tw.q_rotor] / theta
                cur = min(int(a + 1e-9), len(tw.panels) - 1)
                r, t, z, tilt = tw.column(cur)
                hp, ht = tw.head_state()
                k = cur % n
                print(f"  t {d.time:6.3f}  rotor {a:6.3f}  lift{k} {d.qpos[tw.q_lift[k]] * 1000:7.2f}  bolt{k} "
                      f"{d.qpos[tw.q_bolt[k]] * 1000:6.2f}  panel{cur}: r {r * 1000:6.2f} t {t * 1000:5.2f} z {z * 1000:7.2f} "
                      f"tilt {math.degrees(tilt):4.2f}  head z {(hp[2] - head0[2]) * 1000:7.2f} tilt {math.degrees(ht):4.2f}")
            if st["next_step"] <= steps and d.qpos[tw.q_rotor] >= (st["next_step"] + 0.07) * theta:
                i = st["next_step"] - 1
                r, t, z, tilt = tw.column(i)
                hp, ht = tw.head_state()
                print(f"step {i + 1:2d}  t {d.time:6.2f} s  segment {i:2d}: radial {r * 1000:6.2f} mm, sideways "
                      f"{t * 1000:5.2f} mm, bottom {z * 1000:7.2f} mm (expected {tw.expected_bottom(i, i + 1) * 1000:7.2f})"
                      f"   head +{(hp[2] - head0[2]) * 1000:7.2f} mm, tilt {math.degrees(ht):.2f} deg")
                st["next_step"] += 1
            if d.time >= t_stop + 0.5:             # the rotor has stopped: nothing should move now
                hp = tw.head_state()[0]
                if st["ref"] is None:
                    st["ref"], st["all"] = hp, d.xpos.copy()
                st["far"] = max(st["far"], float(np.linalg.norm(hp - st["ref"])))
                if not force:
                    move = np.linalg.norm(d.xpos - st["all"], axis=1)
                    if move.max() > st["moved"]:
                        st["moved"], st["part"] = float(move.max()), int(move.argmax())
        return True

    def finish():
        if viewer is not None:
            viewer.close()
        if args.record:
            fr = st["frames"]
            np.savez_compressed(args.record, time=np.array([f[0] for f in fr]), qpos=np.array([f[1] for f in fr]))
            print(f"recorded {len(fr)} frames to {args.record}; to watch them:\n"
                  f"  python3 -m cspace.viewer --replay {args.record} examples/helixturm.yaml -o helixturm.html")

    if not advance(t_stop + max(args.hold, 0.6)):
        finish()
        return 1

    # ---------- verdict ----------
    def misplaced(tol=0.002, turn=0.5):
        bad = []
        for i in range(steps):
            r, t, z, tilt = tw.column(i)
            ez = z - tw.expected_bottom(i, steps)
            if abs(r) > tol or abs(t) > tol or abs(ez) > tol or tilt > math.radians(turn):
                bad.append(f"segment {i}: radial {r * 1000:.2f}, sideways {t * 1000:.2f}, height error {ez * 1000:.2f} mm, "
                           f"tilt {math.degrees(tilt):.2f} deg")
        return bad

    ok = True
    bad = misplaced()
    hp, ht = tw.head_state()
    rise = hp[2] - head0[2]
    gained = potential() - energy0
    print(f"\nsimulated {d.time:.1f} s in {time.time() - wall0:.0f} s")
    print(f"motor: peak torque on the cam {st['peak']:.1f} N m, work {st['work']:.1f} J; the parts gained {gained:.1f} J "
          f"of height ({100 * gained / st['work'] if st['work'] > 0 else 0:.0f}% of the work; the rest went into friction, "
          "impacts and the lifters' falls)")
    print(f"segments in place (within 2 mm and 0.5 deg): {steps - len(bad)} of {steps}")
    for line in bad[:10]:
        print("  " + line)
    ok = ok and not bad
    if steps % n == 0:
        want = steps // n * hx.H
        good = abs(rise - want) < 0.002
        print(f"head rose {rise * 1000:.2f} mm; {steps // n} turn(s) of segments are {want * 1000:.2f} mm: {'ok' if good else 'WRONG'}")
        ok = ok and good
    else:
        print(f"head rose {rise * 1000:.2f} mm")
    off = math.hypot(hp[0] - head0[0], hp[1] - head0[1])
    good = ht < math.radians(0.5) and off < 0.003
    print(f"head tilt {math.degrees(ht):.2f} deg, off axis {off * 1000:.2f} mm: {'ok' if good else 'WRONG'}")
    ok = ok and good
    good = st["moved"] < 0.0003
    print(f"standing with the rotor stopped: the head moved {st['far'] * 1000:.3f} mm, the part that moved most "
          f"({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, st['part'])}) {st['moved'] * 1000:.3f} mm: {'ok' if good else 'WRONG'}")
    ok = ok and good
    if args.push:                                  # one second of sideways force on the head, then let go
        t0, st["far"] = d.time, 0.0
        st["ref"] = tw.head_state()[0]
        alive = advance(t0 + 1.0, args.push)
        pushed = st["far"]
        alive = alive and advance(t0 + 2.0)
        left = float(np.linalg.norm(tw.head_state()[0] - st["ref"]))
        bad = misplaced(0.005, 1.0)
        good = alive and pushed < 0.02 and left < 0.004 and len(bad) == 0
        print(f"pushed sideways with {args.push:g} N for a second: the head gave {pushed * 1000:.2f} mm and came back to "
              f"{left * 1000:.2f} mm; segments still within 5 mm: {steps - len(bad)} of {steps}: {'ok' if good else 'WRONG'}")
        ok = ok and good
    finish()
    print("RESULT: " + ("the tower stands" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
