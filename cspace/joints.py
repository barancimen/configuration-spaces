"""Joint library.

Every pairwise constraint acts on the relative transform of two frames,
    Trel = (Ta*Fa)^-1 * (Tb*Fb) = (R, p),
and is defined once by: its variables (`roles`), a parametric transform `T(v)` used when the
joint is part of the spanning tree, implicit equations on (R, p) used when it closes a loop,
and a `readout` of its coordinates from (R, p), used for limits and couplings in both cases.
"""
from functools import reduce
from math import gcd

from sympy import I, cos, expand, eye, ilcm, pi, sin

from .se3 import Rx, Ry, Rz, quat, trans
from .spec import SpecError

_I3 = eye(3)


def _aligned(R):
    """z axes coincide: R*ez = ez and R^T*ez = ez."""
    return [R[0, 2], R[1, 2], R[2, 0], R[2, 1], R[2, 2] - 1]


def _identity(R):
    return [R[i, j] - _I3[i, j] for i in range(3) for j in range(3)]


class Kind:
    def __init__(self, codim, implicit, roles=None, T=None, readout=None):
        self.codim = codim            # freedoms removed, for the naive count only
        self.implicit = implicit      # (R, p, opts) -> polynomials that must vanish
        self.roles = roles            # variable roles; None if the kind has no parametric form
        self.T = T                    # {role: expr} -> 4x4
        self.readout = readout or (lambda R, p: {})


KINDS = {
    "fixed": Kind(6, lambda R, p, o: _identity(R) + list(p), [], lambda v: eye(4)),
    "revolute": Kind(5, lambda R, p, o: _aligned(R) + list(p), ["c", "s"],
                     lambda v: Rz(v["c"], v["s"]),
                     lambda R, p: {"c": R[0, 0], "s": R[1, 0]}),
    "prismatic": Kind(5, lambda R, p, o: _identity(R) + [p[0], p[1]], ["d"],
                      lambda v: trans(0, 0, v["d"]),
                      lambda R, p: {"d": p[2]}),
    "cylindrical": Kind(4, lambda R, p, o: _aligned(R) + [p[0], p[1]], ["c", "s", "d"],
                        lambda v: Rz(v["c"], v["s"]) * trans(0, 0, v["d"]),
                        lambda R, p: {"c": R[0, 0], "s": R[1, 0], "d": p[2]}),
    "planar": Kind(3, lambda R, p, o: _aligned(R) + [p[2]], ["x", "y", "c", "s"],
                   lambda v: trans(v["x"], v["y"], 0) * Rz(v["c"], v["s"]),
                   lambda R, p: {"x": p[0], "y": p[1], "c": R[0, 0], "s": R[1, 0]}),
    "spherical": Kind(3, lambda R, p, o: list(p), ["q0", "q1", "q2", "q3"],
                      lambda v: quat(v["q0"], v["q1"], v["q2"], v["q3"])),
    # rotation about a's x axis, then about b's y axis; the two axes stay perpendicular
    "universal": Kind(4, lambda R, p, o: list(p) + [R[0, 1]], ["c1", "s1", "c2", "s2"],
                      lambda v: Rx(v["c1"], v["s1"]) * Ry(v["c2"], v["s2"]),
                      lambda R, p: {"c1": R[1, 1], "s1": R[2, 1], "c2": R[0, 0], "s2": R[0, 2]}),
    "free": Kind(0, lambda R, p, o: [], ["x", "y", "z", "q0", "q1", "q2", "q3"],
                 lambda v: trans(v["x"], v["y"], v["z"]) * quat(v["q0"], v["q1"], v["q2"], v["q3"])),
    # implicit only: origin of b's frame on a's z axis / in a's xy plane / at a distance; z axes aligned
    "point_on_line": Kind(2, lambda R, p, o: [p[0], p[1]], readout=lambda R, p: {"d": p[2]}),
    "point_on_plane": Kind(1, lambda R, p, o: [p[2]], readout=lambda R, p: {"x": p[0], "y": p[1]}),
    "distance": Kind(1, lambda R, p, o: [p.dot(p) - o["length"]**2]),
    "parallel": Kind(2, lambda R, p, o: _aligned(R)),
}

# coordinate name used in the input file -> roles
ANGLES = {"angle": ("c", "s"), "angle1": ("c1", "s1"), "angle2": ("c2", "s2")}
LENGTHS = ("d", "x", "y", "z")


def coordinate(joint, cname=None):
    """Resolve a coordinate of a joint -> ('angle', c, s, theta) or ('length', value)."""
    vals = joint.values
    if cname is None:
        cname = "angle" if "c" in vals else "d"
    if cname in ANGLES and ANGLES[cname][0] in vals:
        c, s = ANGLES[cname]
        return ("angle", vals[c], vals[s], joint.theta.get(c))
    if cname in LENGTHS and cname in vals:
        return ("length", vals[cname])
    raise SpecError(f"joint {joint.name!r} ({joint.kind}) has no coordinate {cname!r}")


def _pair(spec, v, what):
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise SpecError(f"{what}: expected [low, high], got {v!r}")
    return spec.num(v[0]), spec.num(v[1])


def limits(joint, spec, trig):
    """Inequalities (each `expr >= 0`) from the joint's `range`, `limits` and `cone` options."""
    out = []
    wanted = dict(joint.opts.get("limits") or {})
    if "range" in joint.opts:
        wanted[None] = joint.opts["range"]
    for cname, rng in wanted.items():
        lo, hi = _pair(spec, rng, f"joint {joint.name!r} range")
        co = coordinate(joint, cname)
        if co[0] == "length":
            out += [co[1] - lo, hi - co[1]]
            continue
        _, c, s, theta = co
        width = hi - lo
        if width.is_negative:
            raise SpecError(f"joint {joint.name!r}: range {rng} is reversed")
        if trig and theta is not None:
            out += [theta - lo, hi - theta]
        elif (width - 2 * pi).is_positive:
            raise SpecError(f"joint {joint.name!r}: a range wider than a full turn is not a set of "
                            "poses; use --trig with this joint in the spanning tree")
        elif width != 2 * pi:
            mid = (lo + hi) / 2
            out.append(c * cos(mid) + s * sin(mid) - cos(width / 2))
    if "cone" in joint.opts:        # b's z axis within the given angle of a's z axis
        out.append(joint.R[2, 2] - cos(spec.num(joint.opts["cone"])))
    return out


def coupling(terms, phi, trig):
    """Equations for sum(coef * coordinate) = phi.

    terms: list of (coef, coordinate) with coordinate as returned by `coordinate`.
    Pure angle relations become polynomial through z = c + i*s: with the coefficients scaled to
    coprime integers a_i, the relation is prod z_i^a_i = exp(i*phi'), which is exactly the image
    of the real-angle relation.
    """
    angular = [(k, co) for k, co in terms if co[0] == "angle"]
    linear = [(k, co) for k, co in terms if co[0] == "length"]
    if trig and angular and all(co[3] is not None for _, co in angular):
        return [sum(k * co[3] for k, co in angular) + sum(k * co[1] for k, co in linear) - phi]
    if angular and linear:
        raise SpecError("a coupling between an angle and a length (screw, rack and pinion) is not "
                        "polynomial; use --trig with the hinges in the spanning tree")
    if not angular:
        return [sum(k * co[1] for k, co in linear) - phi]
    if not all(k.is_Rational for k, _ in angular):
        raise SpecError("angle couplings need rational coefficients")
    scale = ilcm(*[k.q for k, _ in angular])
    ints = [int(k * scale) for k, _ in angular]
    g = reduce(gcd, ints)
    ints, phi = [a // g for a in ints], phi * scale / g
    lhs, rhs = 1, cos(phi) + I * sin(phi)
    for a, (_, co) in zip(ints, angular):
        z = co[1] + I * co[2]
        if a > 0:
            lhs *= z**a
        else:
            rhs *= z**(-a)
    P = expand(lhs - rhs)
    imag = P.coeff(I)
    return [expand(P - I * imag), imag]
