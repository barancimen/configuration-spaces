"""Load a mechanism description (YAML) into exact SymPy data.

Every scalar is read as text and parsed symbolically, so `0.1` is 1/10 and `pi/3` stays `pi/3`.
"""
import re
from dataclasses import dataclass, field

import yaml
from sympy import S, Symbol, cos, eye, pi, sin, sqrt, tan
from sympy.core.relational import Relational
from sympy.parsing.sympy_parser import (convert_xor, parse_expr, rationalize,
                                        standard_transformations)

from .se3 import frame

_TRANSFORMS = standard_transformations + (convert_xor, rationalize)
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
# x, y, z are the coordinates of shape predicates; the capitals are Mathematica built-ins.
_RESERVED = {"x", "y", "z", "pi", "sqrt", "sin", "cos", "tan", "C", "D", "E", "I", "K", "N", "O"}


class SpecError(ValueError):
    pass


def check_name(name, what):
    if not _NAME.match(str(name)):
        raise SpecError(f"{what} name {name!r} must be letters and digits, starting with a letter")
    return str(name)


def truthy(v):
    return str(v).strip().lower() in ("true", "yes", "1")


@dataclass
class Body:
    name: str
    frames: dict
    shape: object = None


@dataclass
class Constraint:
    kind: str
    args: object
    name: str
    opts: dict


@dataclass
class Spec:
    name: str
    env: dict = field(default_factory=dict)          # parameter name -> symbol or exact value
    params: list = field(default_factory=list)       # the symbolic ones
    assumptions: list = field(default_factory=list)
    bodies: dict = field(default_factory=dict)
    constraints: list = field(default_factory=list)

    def num(self, v, extra=None):
        """Parse one scalar exactly. `extra` adds allowed names (used for shape coordinates)."""
        names = {"pi": pi, "sqrt": sqrt, "sin": sin, "cos": cos, "tan": tan}
        names.update(self.env)
        names.update(extra or {})
        try:
            e = parse_expr(str(v), local_dict=dict(names), transformations=_TRANSFORMS)
        except Exception as err:
            raise SpecError(f"cannot parse {v!r}: {err}")
        allowed = set(self.params) | set((extra or {}).values())
        unknown = getattr(e, "free_symbols", set()) - allowed
        if unknown:
            raise SpecError(f"unknown name(s) {sorted(map(str, unknown))} in {v!r}")
        return e

    def vec(self, v):
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise SpecError(f"expected three coordinates, got {v!r}")
        return [self.num(c) for c in v]

    def placement(self, d):
        """Frame from the optional keys `at`, `z`, `x` of a mapping."""
        return frame(self.vec(d.get("at", [0, 0, 0])),
                     self.vec(d["z"]) if "z" in d else None,
                     self.vec(d["x"]) if "x" in d else None)

    def ref(self, text):
        """'body.frame' or 'body' -> (body name, frame matrix in body coordinates)."""
        body, _, fname = str(text).partition(".")
        if body != "world" and body not in self.bodies:
            raise SpecError(f"unknown body {body!r} in {text!r}")
        frames = self.bodies[body].frames if body in self.bodies else {}
        if not fname:
            return body, eye(4)
        if fname not in frames:
            raise SpecError(f"body {body!r} has no frame {fname!r}")
        return body, frames[fname]


def load(path, kinds, overrides=None):
    """Read `path`. `kinds` is the set of known constraint keywords; `overrides` maps parameter
    names to values that replace the file's."""
    with open(path) as f:
        raw = yaml.load(f, Loader=yaml.BaseLoader) or {}
    spec = Spec(name=str(raw.get("name", "mechanism")))

    params = dict(raw.get("parameters") or {})
    for pname, value in (overrides or {}).items():
        if pname not in params:
            raise SpecError(f"--set {pname}: the file has no parameter {pname!r}")
        params[pname] = value
    for pname, value in params.items():
        check_name(pname, "parameter")
        if pname in _RESERVED:
            raise SpecError(f"parameter name {pname!r} is reserved")
        if str(value).strip() == "symbolic":
            sym = Symbol(pname, real=True)
            spec.env[pname] = sym
            spec.params.append(sym)
        else:
            spec.env[pname] = spec.num(value)

    for text in raw.get("assume") or []:
        rel = spec.num(text)
        if rel is S.true:            # all its parameters were given numbers
            continue
        if rel is S.false:
            raise SpecError(f"assumption {text!r} is false for the given parameter values")
        if not isinstance(rel, Relational):
            raise SpecError(f"assumption {text!r} is not a comparison")
        spec.assumptions.append(rel)

    for bname, b in (raw.get("bodies") or {}).items():
        check_name(bname, "body")
        b = b or {}
        frames = {check_name(fn, "frame"): spec.placement(fd or {})
                  for fn, fd in (b.get("frames") or {}).items()}
        if bname == "world" and "shape" in b:
            raise SpecError("the world body carries frames only; make a body and fix it to world")
        spec.bodies[bname] = Body(bname, frames, b.get("shape"))

    for i, c in enumerate(raw.get("constraints") or [], 1):
        found = [k for k in c if k in kinds]
        if len(found) != 1:
            raise SpecError(f"constraint {i} needs exactly one of {sorted(kinds)}, got {sorted(c)}")
        kind = found[0]
        opts = {k: v for k, v in c.items() if k not in (kind, "name")}
        spec.constraints.append(Constraint(kind, c[kind], check_name(c.get("name", f"j{i}"), "constraint"), opts))
    return spec
