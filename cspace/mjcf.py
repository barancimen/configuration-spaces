"""Export a mechanism to MuJoCo's MJCF format, for simulating its motion.

    python3 -m cspace.mjcf examples/stewart.yaml -o stewart.xml --motors leg1,leg2,leg3,leg4,leg5,leg6
    python3 -m cspace.mjcf examples/fourbar.yaml:a=2,b=5,c=4,d=6 -o fourbar.xml --keyframes 3

How it maps:
  * the compiler's spanning tree becomes the MJCF body tree;
  * tree joints become MuJoCo joints: revolute -> hinge, prismatic -> slide, spherical -> ball,
    free -> freejoint, and universal / cylindrical / planar -> several hinges and slides in one body;
  * loop-closing joints become equality constraints: spherical -> connect, fixed -> weld,
    revolute -> two connects on its axis; two-joint linear couplings (gear, couple) -> joint equality;
  * shapes become geoms (box, sphere, cylinder, hull -> mesh); bodies without a shape get thin capsules
    between their frames; only no_overlap pairs collide;
  * the model is written at a valid sampled pose, with each hinge and slide's `ref` set to its
    coordinate there, so MuJoCo's reference pose closes every loop exactly. More samples can be
    added as keyframes.
Not representable, reported as warnings: closing prismatic / cylindrical / planar / universal joints,
the implicit constraints (point_on_line, point_on_plane, distance, parallel), couplings of more than
two joints, cone limits, and difference / intersection / where shapes.
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from sympy import lambdify

from .check import sample
from .compile import ALL_KINDS, compile_spec
from .logic import quantified
from .shapes import _box, _cyl, _kind, _placement
from .spec import SpecError, load

AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
# each joint kind as a sequence of elementary motions in its frame: (type, axis, coordinate role)
ELEMENTS = {
    "fixed": [],
    "revolute": [("hinge", "z", "c")],
    "prismatic": [("slide", "z", "d")],
    "cylindrical": [("hinge", "z", "c"), ("slide", "z", "d")],
    "planar": [("slide", "x", "x"), ("slide", "y", "y"), ("hinge", "z", "c")],
    "universal": [("hinge", "x", "c1"), ("hinge", "y", "c2")],
    "spherical": [("ball", None, "q")],
    "free": [("slide", "x", "x"), ("slide", "y", "y"), ("slide", "z", "z"), ("ball", None, "q")],
}
PALETTE = ["#3f6f9f", "#8d3d4a", "#3f8a78", "#6c5c9e", "#a3592e", "#5b7f2e", "#2e7f9b", "#7d6f4e"]


# ---------- rigid-transform helpers (numpy, 4x4) ----------

def rot(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return T


def trans(v):
    T = np.eye(4)
    T[:3, 3] = v
    return T


def quat_matrix(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    return T


def matrix_quat(R):
    """Unit quaternion (w, x, y, z) of a rotation matrix, w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q) / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def elementary(kind, axis, value):
    if kind == "hinge":
        return rot(axis, value)
    if kind == "slide":
        return trans(np.asarray(axis, float) * value)
    return quat_matrix(value)


def volume(g):
    t = g["type"]
    if t == "box":
        return 8 * np.prod(g["size"])
    if t == "sphere":
        return 4 / 3 * np.pi * g["size"][0] ** 3
    if t == "cylinder":
        return np.pi * g["size"][0] ** 2 * 2 * g["size"][1]
    if t == "capsule":
        r, L = g["size"], np.linalg.norm(g["fromto"][3:] - g["fromto"][:3])
        return np.pi * r * r * L + 4 / 3 * np.pi * r ** 3
    from scipy.spatial import ConvexHull
    return ConvexHull(g["vertices"]).volume


def num(M):
    """A SymPy matrix as floats; column vectors come back one-dimensional."""
    a = np.array(M.evalf(), dtype=float)
    return a.ravel() if a.ndim == 2 and a.shape[1] == 1 else a


def fmt(v):
    return " ".join(f"{x:.10g}" for x in np.ravel(v))


# ---------- the exporter ----------

class Exporter:
    def __init__(self, spec, system, density=1000.0, damping=0.05, gravity=True, timestep=0.002):
        self.spec, self.system = spec, system
        self.density, self.damping, self.gravity, self.timestep = density, damping, gravity, timestep
        self.warnings = []
        X = system.variables
        self._pose = {b: lambdify(X, M, "numpy") for b, M in system.poses.items()}
        self._vals = {j.name: lambdify(X, list(j.values.values()), "numpy") for j in system.joints if j.values}
        self.tree = [j for j in system.joints if j.tree]

    # numeric world pose of a body at the variable values x
    def pose(self, body, x):
        return np.eye(4) if body == "world" else np.array(self._pose[body](*x), dtype=float)

    def values(self, j, x):
        return dict(zip(j.values, np.array(self._vals[j.name](*x), dtype=float))) if j.values else {}

    def elements(self, j, x):
        """MuJoCo-side motions of tree joint j: (frame F in the child body, [(type, axis, value)]).
        Child pose = parent pose * M, with M = F_a J F_b^-1 when the child is j.b and
        M = F_b J^-1 F_a^-1 when it is j.a (J^-1 is the reversed sequence, axes negated)."""
        vals = self.values(j, x)
        els = []
        for kind, ax, role in ELEMENTS[j.kind]:
            if kind == "ball":
                v = np.array([vals[f"q{k}"] for k in range(4)])
            elif role.startswith("c"):
                v = float(np.arctan2(vals["s" + role[1:]], vals[role]))
            else:
                v = float(vals[role])
            els.append((kind, AXES[ax] if ax else None, v))
        if j.parent == j.a:
            return num(j.Fb), els
        rev = []
        for kind, ax, v in reversed(els):
            if kind == "ball":
                rev.append((kind, None, v * np.array([1, -1, -1, -1])))
            else:
                rev.append((kind, tuple(-c for c in ax), v))
        return num(j.Fa), rev

    def child(self, j):
        return j.b if j.parent == j.a else j.a

    def mj_joints(self, j, x0):
        """MuJoCo joints for tree joint j, with axes and positions in the child body's frame at the
        reference sample x0: factor i of J(x0)^-1 J(x) is G_i E_i(delta) G_i^-1, G_i = F (E_{i+1} ... E_k)^-1."""
        F, els = self.elements(j, x0)
        out = []
        for i, (kind, ax, v) in enumerate(els):
            S = np.eye(4)
            for kind2, ax2, v2 in els[i + 1:]:
                S = S @ elementary(kind2, ax2, v2)
            G = F @ np.linalg.inv(S)
            out.append({"type": kind, "axis": G[:3, :3] @ np.asarray(ax, float) if ax else None,
                        "pos": G[:3, 3], "ref": v if kind != "ball" else None, "G": G})
        return out

    def qpos(self, x, x0):
        """MuJoCo qpos for the configuration x, with the model written at x0 (same joint order)."""
        out = []
        for j in self.ordered:
            if self.freejoint(j):
                T = self.pose(self.child(j), x)
                out += list(T[:3, 3]) + list(matrix_quat(T[:3, :3]))
                continue
            _, els = self.elements(j, x)
            _, els0 = self.elements(j, x0)
            for (kind, ax, v), (_, _, v0), mj in zip(els, els0, self.mj_joints(j, x0)):
                if kind == "ball":
                    R = mj["G"][:3, :3]
                    delta = quat_matrix(v0)[:3, :3].T @ quat_matrix(v)[:3, :3]
                    out += list(matrix_quat(R @ delta @ R.T))
                else:
                    out.append(v)
        return out

    def freejoint(self, j):
        return (j.kind == "free" and j.parent == "world" and j.a == "world"
                and np.allclose(num(j.Fa), np.eye(4)) and np.allclose(num(j.Fb), np.eye(4)))

    # ---------- geometry ----------
    def geoms(self, body, scale):
        b = self.spec.bodies.get(body)
        out = []
        if b is not None and b.shape is not None:
            self._walk(b.shape, np.eye(4), out, body)
        elif b is not None and len(b.frames) >= 1:
            pts = [num(F[:3, 3]) for F in b.frames.values()]
            if len(pts) == 1:
                pts = [np.zeros(3)] + pts
            for p, q in zip(pts, pts[1:]):
                if np.linalg.norm(q - p) > 1e-9:
                    out.append({"type": "capsule", "fromto": np.concatenate([p, q]), "size": 0.012 * scale})
        return out

    def _walk(self, node, T, out, body):
        T = T @ num(_placement(self.spec, node))
        kind = _kind(self.spec, node)
        v = node[kind]
        if kind == "box":
            out.append({"type": "box", "size": [abs(float(h)) for h in _box(self.spec, v)], "T": T})
        elif kind == "ball":
            out.append({"type": "sphere", "size": [abs(float(self.spec.num(v)))], "T": T})
        elif kind == "cylinder":
            r, h = _cyl(self.spec, v)
            out.append({"type": "cylinder", "size": [abs(float(r)), abs(float(h))], "T": T})
        elif kind == "hull":
            pts = np.array([[float(self.spec.num(c)) for c in p] for p in v])
            out.append({"type": "mesh", "vertices": pts, "T": T})
        elif kind == "union":
            for k in v:
                self._walk(k, T, out, body)
        else:
            self.warnings.append(f"{body}: a '{kind}' shape is not exported (MuJoCo collides convex shapes only)")
            if kind == "difference":
                self._walk(v[0], T, out, body)

    # ---------- the document ----------
    def build(self, x0, keyframes=(), motors=(), kp=None, motor_settings=None):
        spec, system = self.spec, self.system
        root = ET.Element("mujoco", model=spec.name)
        ET.SubElement(root, "compiler", angle="radian", autolimits="true")
        ET.SubElement(root, "option", timestep=f"{self.timestep:g}", integrator="implicitfast",
                      gravity="0 0 -9.81" if self.gravity else "0 0 0")
        default = ET.SubElement(root, "default")
        ET.SubElement(default, "geom", contype="0", conaffinity="0", density=f"{self.density:g}")
        ET.SubElement(default, "joint", damping=f"{self.damping:g}")
        asset = ET.SubElement(root, "asset")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", pos="0 0 10", dir="0 0 -1", directional="true")

        geo_scale = max([1.0] + [float(abs(c)) for b in spec.bodies.values()
                                 for F in b.frames.values() for c in num(F[:3, 3])])
        children = {}
        for j in self.tree:
            children.setdefault(j.parent, []).append(j)
        self.ordered, geom_names, color_i = [], {}, 0
        self.joint_el, self.mass, self.scale = {}, 0.0, geo_scale
        motors = list(motors)

        def emit(parent_el, parent_body, j):
            nonlocal color_i
            body = self.child(j)
            T = np.linalg.inv(self.pose(parent_body, x0)) @ self.pose(body, x0)
            el = ET.SubElement(parent_el, "body", name=body, pos=fmt(T[:3, 3]), quat=fmt(matrix_quat(T[:3, :3])))
            self.ordered.append(j)
            if self.freejoint(j):
                ET.SubElement(el, "freejoint", name=j.name)
            else:
                mjs = self.mj_joints(j, x0)
                for k, mj in enumerate(mjs):
                    name = j.name if len(mjs) == 1 else f"{j.name}_{k + 1}"
                    attrs = {"name": name, "type": mj["type"], "pos": fmt(mj["pos"])}
                    if mj["type"] != "ball":
                        attrs["axis"] = fmt(mj["axis"])
                        attrs["ref"] = f"{mj['ref']:.10g}"
                        rng = self.joint_range(j, k)
                        if rng is not None:
                            attrs["range"] = fmt(rng)
                    if "spring" in j.opts:
                        if len(mjs) != 1 or mj["type"] == "ball":
                            self.warnings.append(f"{j.name}: springs are exported for single-axis joints only")
                        else:
                            attrs.update(self.spring(j))
                    self.joint_el[name] = ET.SubElement(el, "joint", **attrs)
                if "cone" in j.opts:
                    self.warnings.append(f"{j.name}: cone limit not exported (MuJoCo's ball range measures "
                                         "the total rotation, not the angle between two axes)")
            b = spec.bodies.get(body)
            hexcolor = (b.color if b is not None and b.color else None) or PALETTE[color_i % len(PALETTE)]
            color_i += 1
            rgba = " ".join(f"{int(hexcolor[i:i + 2], 16) / 255:.3f}" for i in (1, 3, 5)) + " 1"
            gs = self.geoms(body, geo_scale)
            geom_names[body] = []
            for k, g in enumerate(gs):
                self.mass += self.density * volume(g)
                gname = f"{body}_g{k + 1}"
                geom_names[body].append(gname)
                attrs = {"name": gname, "type": g["type"], "rgba": rgba}
                if g["type"] == "capsule":
                    attrs.update(fromto=fmt(g["fromto"]), size=f"{g['size']:.6g}")
                else:
                    attrs.update(pos=fmt(g["T"][:3, 3]), quat=fmt(matrix_quat(g["T"][:3, :3])))
                    if g["type"] == "mesh":
                        ET.SubElement(asset, "mesh", name=f"{gname}_mesh", vertex=fmt(g["vertices"]))
                        attrs["mesh"] = f"{gname}_mesh"
                    else:
                        attrs["size"] = fmt(g["size"])
                ET.SubElement(el, "geom", **attrs)
            if not gs:   # MuJoCo needs mass on every moving body
                ET.SubElement(el, "inertial", pos="0 0 0", mass="1e-3", diaginertia="1e-6 1e-6 1e-6")
                self.mass += 1e-3
            for c in children.get(body, []):
                emit(el, body, c)

        for c in children.get("world", []):
            emit(world, "world", c)

        self._equalities(root, x0)
        self._contacts(root, geom_names)
        mj_names = self._motors(root, motors, kp, x0, motor_settings)
        keys = ET.SubElement(root, "keyframe")
        for i, x in enumerate([x0] + list(keyframes)):
            attrs = {"name": "sample" if i == 0 else f"sample{i + 1}", "qpos": fmt(self.qpos(x, x0))}
            if mj_names:
                attrs["ctrl"] = fmt([self.qpos_named(x, x0)[n] for n in mj_names])
            ET.SubElement(keys, "key", **attrs)
        ET.indent(root, space="  ")
        header = (f"<!-- {spec.name}: exported by cspace. The model is written at a valid sample, so "
                  "loops start closed; load the 'sample' keyframe to start there with matching controls. -->\n"
                  f"<!-- total mass {self.mass:.4g} at density {self.density:g}; lengths in the file's units -->\n")
        for w in self.warnings:
            header += f"<!-- warning: {w.replace('--', '-')} -->\n"
        return header + ET.tostring(root, encoding="unicode") + "\n"

    def qpos_named(self, x, x0):
        """Joint name -> value for single-coordinate MuJoCo joints (hinges and slides)."""
        vals, q = {}, self.qpos(x, x0)
        i = 0
        for j in self.ordered:
            if self.freejoint(j):
                i += 7
                continue
            mjs = self.mj_joints(j, x0)
            for k, mj in enumerate(mjs):
                name = j.name if len(mjs) == 1 else f"{j.name}_{k + 1}"
                if mj["type"] == "ball":
                    i += 4
                else:
                    vals[name] = q[i]
                    i += 1
        return vals

    def spring(self, j):
        """MJCF attributes for spring: {stiffness, rest, damping} (rest in joint units)."""
        sp = j.opts["spring"]
        if not isinstance(sp, dict) or "stiffness" not in sp:
            raise SpecError(f"{j.name}: spring needs at least a stiffness: spring: {{stiffness: k, rest: value}}")
        out = {"stiffness": f"{float(self.spec.num(sp['stiffness'])):g}",
               "springref": f"{float(self.spec.num(sp.get('rest', 0))):.10g}"}
        if "damping" in sp:
            out["damping"] = f"{float(self.spec.num(sp['damping'])):g}"
        return out

    def joint_range(self, j, k):
        """Range of MuJoCo joint k of tree joint j, from the joint's range / limits, if any."""
        roles = [r for _, _, r in ELEMENTS[j.kind]]
        if j.parent != j.a:
            roles = roles[::-1]
        role = roles[k]
        name = {"c": "angle", "c1": "angle1", "c2": "angle2"}.get(role, role)
        lim = dict(j.opts.get("limits") or {})
        if "range" in j.opts:
            lim["angle" if "c" in j.values else "d"] = j.opts["range"]
        if name not in lim:
            return None
        return [float(self.spec.num(v)) for v in lim[name]]

    def _equalities(self, root, x0):
        eq = ET.SubElement(root, "equality")
        system = self.system
        for j in system.joints:
            if j.tree:
                continue
            Fa = num(j.Fa)
            if j.kind == "spherical":
                ET.SubElement(eq, "connect", name=j.name, body1=j.a, body2=j.b, anchor=fmt(Fa[:3, 3]))
            elif j.kind == "fixed":
                ET.SubElement(eq, "weld", name=j.name, body1=j.a, body2=j.b)
            elif j.kind == "revolute":
                L = 0.1 * max(1.0, float(np.linalg.norm(Fa[:3, 3])))
                for k, p in enumerate((Fa[:3, 3], Fa[:3, 3] + L * Fa[:3, 2]), 1):
                    ET.SubElement(eq, "connect", name=f"{j.name}_{k}", body1=j.a, body2=j.b, anchor=fmt(p))
            else:
                self.warnings.append(f"{j.name}: a loop-closing {j.kind} constraint has no MuJoCo equivalent "
                                     "and is not exported; list it earlier, or with closing: false, so it "
                                     "becomes a tree joint")
        for c in self.spec.constraints:
            if c.kind not in ("gear", "couple"):
                continue
            terms = self._coupling_terms(c)
            if terms is None:
                continue
            (j1, k1), (j2, k2) = terms
            # k1*q1 + k2*q2 = const  ->  q1 - q1_0 = -(k2/k1) (q2 - q2_0)
            ET.SubElement(eq, "joint", name=c.name, joint1=j1, joint2=j2, polycoef=fmt([0, -k2 / k1, 0, 0, 0]))

    def _coupling_terms(self, c):
        tree = {j.name: j for j in self.tree}
        if c.kind == "gear":
            n1, n2 = (float(self.spec.num(t)) for t in c.opts["teeth"])
            sign = -1 if str(c.opts.get("internal", "false")).lower() in ("true", "yes", "1") else 1
            items = [(str(c.args[0]), n1), (str(c.args[1]), sign * n2)]
        else:
            items = [(str(k), float(self.spec.num(v))) for k, v in c.args.items()]
        if len(items) != 2:
            self.warnings.append(f"{c.name}: couplings of more than two joints are not exported")
            return None
        out = []
        for key, k in items:
            jname = key.partition(".")[0]
            j = tree.get(jname)
            if j is None or len(ELEMENTS[j.kind]) != 1 or ELEMENTS[j.kind][0][0] == "ball":
                self.warnings.append(f"{c.name}: coupled joints must be single-axis tree joints; not exported")
                return None
            out.append((jname, k))
        return out

    def _contacts(self, root, geom_names):
        pairs = []
        for c in self.spec.constraints:
            if c.kind in ("no_overlap", "contact"):
                a, b = (str(n) for n in c.args)
                extra = {}
                if "friction" in c.opts:      # sliding friction; small torsional and rolling terms
                    f = float(self.spec.num(c.opts["friction"]))
                    extra["friction"] = f"{f:g} {f:g} 0.005 0.0001 0.0001"
                pairs += [(g1, g2, extra) for g1 in geom_names.get(a, []) for g2 in geom_names.get(b, [])]
        if pairs:
            contact = ET.SubElement(root, "contact")
            for g1, g2, extra in pairs:
                ET.SubElement(contact, "pair", geom1=g1, geom2=g2, **extra)

    def _motors(self, root, motors, kp, x0, settings=None):
        if not motors:
            return []
        act = ET.SubElement(root, "actuator")
        known = self.qpos_named(x0, x0)
        names = []
        n = max(1, len(motors))
        weight = self.mass * 9.81 / n                       # load per motor if all hold the weight
        for m in motors:
            if m not in known:
                raise SpecError(f"--motors: {m!r} is not a hinge or slide of the exported model "
                                f"(known: {', '.join(sorted(known))})")
            slide = self.joint_el[m].get("type") == "slide"
            own = (settings or {}).get(m) or {}
            if kp is not None:
                gain = kp
            elif "kp" in own:
                gain = float(self.spec.num(own["kp"]))
            else:   # gravity moves the joint by about 1% of the mechanism's size
                gain = weight / (0.01 * self.scale) if slide else weight * self.scale / 0.01
            inertia = self.mass / n * (1 if slide else self.scale ** 2)
            damp = float(self.spec.num(own["damping"])) if "damping" in own else 2 * np.sqrt(gain * inertia)
            self.joint_el[m].set("damping", f"{damp:.6g}")   # by default near critical
            attrs = {"name": f"{m}_motor", "joint": m, "kp": f"{gain:.6g}"}
            j = next((j for j in self.ordered if j.name == m), None)
            if j is not None:
                rng = self.joint_range(j, 0)
                if rng is not None:
                    attrs["ctrlrange"] = fmt(rng)
            ET.SubElement(act, "position", **attrs)
            names.append(m)
        return names


def valid_samples(system, count, seed=0):
    """Valid configurations (equations, limits and quantifier-free no_overlap all satisfied)."""
    checks = [lambdify(system.variables, f, "numpy") for f in system.formulas if not quantified(f)]
    out, s = [], seed
    while len(out) < count and s < seed + 50:
        _, pts = sample(system, tries=max(10, 3 * count), seed=s)
        for x, _ in pts:
            if all(bool(f(*x)) for f in checks):
                out.append(np.asarray(x, float))
        s += 1
    if len(out) < count:
        raise SpecError(f"found only {len(out)} valid configurations; is the mechanism assemblable?")
    return out[:count]


def start_pose(spec, system):
    """The configuration given by the file's start: block (joint name -> value, or a list for joints
    with several coordinates). Listed coordinates are held exactly while the rest is pulled onto the
    configuration space, so loops close. Joints not listed start at zero (or mid-range)."""
    from scipy.optimize import least_squares
    from sympy import Matrix, cos, sin
    X = system.variables
    index = {v: i for i, v in enumerate(X)}
    x = np.zeros(len(X))
    joints = {j.name: j for j in system.joints if j.tree}
    for j in joints.values():
        for cr in ("c", "c1", "c2"):
            if cr in j.values:
                x[index[j.values[cr]]] = 1.0
        if "q0" in j.values:
            x[index[j.values["q0"]]] = 1.0
        if "d" in j.values and "range" in j.opts:
            x[index[j.values["d"]]] = sum(float(spec.num(v)) for v in j.opts["range"]) / 2
    held = []                                    # extra equations: the listed coordinates
    for name, value in spec.start.items():
        if name not in joints:
            raise SpecError(f"start: {name!r} is not a tree joint (closing joints follow from the others)")
        j = joints[name]
        vals = value if isinstance(value, list) else [value]
        roles = [r for _, _, r in ELEMENTS[j.kind] if r != "q"]
        if len(vals) != len(roles):
            raise SpecError(f"start: {name!r} ({j.kind}) needs {len(roles)} value(s): {roles}")
        for role, v in zip(roles, vals):
            v = float(spec.num(v))
            if role.startswith("c"):
                c, s_ = j.values[role], j.values["s" + role[1:]]
                x[index[c]], x[index[s_]] = np.cos(v), np.sin(v)
                held += [c - cos(v), s_ - sin(v)]
            else:
                x[index[j.values[role]]] = v
                held.append(j.values[role] - v)
    eqs, ineqs = list(system.equations) + held, list(system.inequalities)
    F = lambdify(X, eqs, "numpy")
    J = lambdify(X, Matrix(eqs).jacobian(X), "numpy")
    G = lambdify(X, ineqs, "numpy")
    JG = lambdify(X, Matrix(ineqs).jacobian(X) if ineqs else Matrix.zeros(1, len(X)), "numpy")

    def resid(z):
        g = np.atleast_1d(np.array(G(*z), float)) if ineqs else np.zeros(0)
        return np.concatenate([np.atleast_1d(np.array(F(*z), float)), np.minimum(g, 0)])

    def jac(z):
        rows = [np.array(J(*z), float).reshape(-1, len(X))]
        if ineqs:
            g = np.atleast_1d(np.array(G(*z), float))
            rows.append(np.array(JG(*z), float).reshape(-1, len(X)) * (g < 0)[:, None])
        return np.vstack(rows)

    sol = least_squares(resid, x, jac=jac, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
    if np.max(np.abs(resid(sol.x)), initial=0.0) > 1e-9:
        raise SpecError("start: the listed values cannot all hold on the configuration space "
                        "(a closed loop or coupling contradicts them); list fewer joints")
    return sol.x


def export(path, given=None, keyframes=1, motors=None, kp=None, density=1000.0, damping=0.05,
           gravity=None, seed=0, timestep=None):
    """Returns (xml text, exporter, samples). Unset options come from the file's simulation: block."""
    from .spec import truthy
    from .viewer import _defaults
    spec = load(path, ALL_KINDS)
    if spec.params:
        vals = _defaults(spec, dict(given or {}))
        spec = load(path, ALL_KINDS, {k: str(v) for k, v in vals.items()})
    sim = spec.simulation
    if gravity is None:
        gravity = truthy(sim.get("gravity", "true"))
    if timestep is None:
        timestep = float(spec.num(sim.get("timestep", "0.002")))
    settings = sim.get("motors") or {}
    if isinstance(settings, list):
        settings = {str(m): {} for m in settings}
    if motors is None:
        motors = list(settings)
    system = compile_spec(spec, shapes=False)
    ex = Exporter(spec, system, density, damping, gravity, timestep)
    if spec.start:
        xs = [start_pose(spec, system)] + (valid_samples(system, keyframes - 1, seed) if keyframes > 1 else [])
    else:
        xs = valid_samples(system, max(1, keyframes), seed)
    if any(quantified(f) for f in system.formulas):
        ex.warnings.append("no_overlap conditions quantified over points were not checked for the sample")
    if any(c.kind == "contact" for c in spec.constraints) and len(xs) > (1 if spec.start else 0):
        ex.warnings.append("contact pairs are not checked when sampling; random keyframes may start with "
                           "parts inside each other" + ("" if spec.start else " (add a start: block)"))
    return ex.build(xs[0], xs[1:], motors, kp, settings), ex, xs


def main(argv=None):
    from .viewer import parse_entry
    ap = argparse.ArgumentParser(prog="cspace.mjcf", description="Export a mechanism to MuJoCo MJCF.")
    ap.add_argument("spec", help="mechanism file, optionally path:name=value,... for parameters")
    ap.add_argument("-o", "--output", help="default: <spec name>.xml")
    ap.add_argument("--keyframes", type=int, default=1, help="valid samples to include as keyframes")
    ap.add_argument("--motors", default="", help="comma-separated hinges/slides to drive with position actuators")
    ap.add_argument("--kp", type=float, default=None,
                    help="position actuator gain (default: gravity moves a joint by about 1%% of the size)")
    ap.add_argument("--density", type=float, default=1000.0)
    ap.add_argument("--damping", type=float, default=0.05)
    ap.add_argument("--no-gravity", action="store_true", help="default: the file's simulation: gravity")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    path, given = parse_entry(args.spec)
    try:
        motors = [m for m in args.motors.split(",") if m] or None
        text, ex, xs = export(path, given, args.keyframes, motors, args.kp, args.density, args.damping,
                              False if args.no_gravity else None, args.seed)
    except SpecError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    out = args.output or f"{Path(path).stem}.xml"
    Path(out).write_text(text)
    nq = len(ex.qpos(xs[0], xs[0]))
    print(f"written to {out}: {len(ex.ordered)} bodies, {nq} qpos entries", file=sys.stderr)
    for w in ex.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
