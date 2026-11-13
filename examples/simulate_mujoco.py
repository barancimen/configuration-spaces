"""Run a model exported by cspace.mjcf in MuJoCo (pip install mujoco).

    python3 -m cspace.mjcf examples/stewart.yaml -o stewart.xml --motors leg1,leg2,leg3,leg4,leg5,leg6
    python3 examples/simulate_mujoco.py stewart.xml --wiggle --view

It starts from the 'sample' keyframe (a valid pose with matching controls), optionally moves every
position actuator around its starting value with a sine wave, and prints the worst loop-closure
error (distance between the two anchors of each connect constraint) as the simulation runs.
Written for MuJoCo 3.x (run with 3.17).
"""
import argparse
import math
import time

import numpy as np
import mujoco


def loop_error(m, d):
    """Largest distance between the two anchor points of any connect constraint."""
    worst = 0.0
    for i in range(m.neq):
        if m.eq_type[i] != mujoco.mjtEq.mjEQ_CONNECT:
            continue
        b1, b2 = m.eq_obj1id[i], m.eq_obj2id[i]
        p1 = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ m.eq_data[i, 0:3]
        p2 = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ m.eq_data[i, 3:6]
        worst = max(worst, float(np.linalg.norm(p1 - p2)))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="MJCF file written by cspace.mjcf")
    ap.add_argument("--seconds", type=float, default=None,
                    help="simulated time to run (default: until the viewer is closed, or 10 s without it)")
    ap.add_argument("--wiggle", action="store_true", help="drive every position actuator with a sine wave")
    ap.add_argument("--amplitude", type=float, default=0.15, help="sine amplitude, in joint units")
    ap.add_argument("--frequency", type=float, default=0.3, help="sine frequency in Hz")
    ap.add_argument("--view", action="store_true", help="open MuJoCo's interactive viewer")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "sample")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    ctrl0 = d.ctrl.copy()
    print(f"{m.nbody - 1} bodies, {m.njnt} joints, {m.neq} equality constraints, {m.nu} actuators")
    print(f"t = 0.00 s   loop error {loop_error(m, d):.2e}")

    def control():
        if args.wiggle and m.nu:
            for i in range(m.nu):
                target = ctrl0[i] + args.amplitude * math.sin(2 * math.pi * args.frequency * d.time + i * math.pi / 3)
                if m.actuator_ctrllimited[i]:
                    target = min(max(target, m.actuator_ctrlrange[i, 0]), m.actuator_ctrlrange[i, 1])
                d.ctrl[i] = target

    report_every, next_report = 0.5, 0.5
    limit = args.seconds if args.seconds is not None else (math.inf if args.view else 10.0)
    if args.view:
        from mujoco import viewer as mj_viewer   # binds only mj_viewer, not the name `mujoco`
        with mj_viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < limit:
                start = time.time()
                control()
                mujoco.mj_step(m, d)
                viewer.sync()
                if d.time >= next_report:
                    print(f"t = {d.time:.2f} s   loop error {loop_error(m, d):.2e}")
                    next_report += report_every
                time.sleep(max(0.0, m.opt.timestep - (time.time() - start)))
            reason = "the viewer was closed" if d.time < limit else f"--seconds {limit:g} reached"
    else:
        while d.time < limit:
            control()
            mujoco.mj_step(m, d)
            if d.time >= next_report:
                print(f"t = {d.time:.2f} s   loop error {loop_error(m, d):.2e}")
                next_report += report_every
        reason = f"{limit:g} s simulated" + ("" if args.seconds is not None else " (the default; set --seconds)")
    print(f"stopped at t = {d.time:.2f} s: {reason}; final loop error {loop_error(m, d):.2e}")


if __name__ == "__main__":
    main()
