"""Exact rigid transforms as 4x4 SymPy matrices whose entries are polynomials."""
from sympy import Matrix, expand, eye, sqrt


def Rz(c, s):
    return Matrix([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def Rx(c, s):
    return Matrix([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def Ry(c, s):
    return Matrix([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def trans(x, y, z):
    T = eye(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T


def quat(w, x, y, z):
    """Rotation by the unit quaternion w + x*i + y*j + z*k."""
    T = eye(4)
    T[:3, :3] = Matrix([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]])
    return T


def inv(T):
    """Inverse of a rigid transform; stays polynomial (no division)."""
    R, p = T[:3, :3], T[:3, 3]
    M = eye(4)
    M[:3, :3] = R.T
    M[:3, 3] = -R.T * p
    return M


def split(T):
    return T[:3, :3], T[:3, 3]


def _unit(v):
    return v / sqrt(expand(v.dot(v)))


def frame(at=(0, 0, 0), z=None, x=None):
    """Frame with origin `at`, z axis along `z` and x axis along `x` made perpendicular to z.

    `x` defaults to the x axis of the surrounding frame, or its y axis when z is parallel to that.
    """
    if z is None and x is None:          # the common case: a plain translation
        return trans(*at)
    zv = _unit(Matrix(z if z is not None else [0, 0, 1]))
    if x is None:
        x = [1, 0, 0] if zv.cross(Matrix([1, 0, 0])) != Matrix([0, 0, 0]) else [0, 1, 0]
    xv = Matrix(x)
    xv = xv - xv.dot(zv) * zv
    if xv.applyfunc(expand) == Matrix([0, 0, 0]):
        raise ValueError("frame: x direction is parallel to z")
    xv = _unit(xv)
    T = eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = xv, zv.cross(xv), zv
    T[:3, 3] = Matrix(at)
    return T
