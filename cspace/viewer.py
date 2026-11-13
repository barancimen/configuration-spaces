"""Build a self-contained web page that samples mechanisms' configuration spaces and draws them.

    python3 -m cspace.viewer -o explorer.html examples/fourbar.yaml:a=2,b=5,c=4,d=6 examples/gears.yaml
    python3 -m cspace examples/fourbar.yaml --view            # one mechanism, writes fourbar.html

Each mechanism is compiled to JavaScript: its equations and their exact Jacobian, the joint
limits, the quantifier-free no-overlap conditions, body poses, joint coordinates and solid
shapes, all as functions of the variables x and the parameters p. The page samples in the
browser (Levenberg-Marquardt projection onto the equations), so it needs no server.
`path:name=value,...` gives default values for symbolic parameters; the page lets you edit them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sympy import And, Matrix, S, Symbol, eye, pi, sympify
from sympy.printing.jscode import jscode

from .compile import ALL_KINDS, compile_spec
from .logic import quantified
from .shapes import _box, _cyl, _kind, _placement
from .spec import SpecError, load

TEMPLATE = Path(__file__).with_name("viewer_template.html")


class JS:
    """Print SymPy expressions as JavaScript over arrays x (variables) and p (parameters)."""

    def __init__(self, variables, params):
        self.sub = {v: Symbol(f"x[{i}]") for i, v in enumerate(variables)}
        self.sub.update({q: Symbol(f"p[{j}]") for j, q in enumerate(params)})

    def e(self, expr):
        expr = sympify(expr)
        if expr is S.true:
            return "true"
        if expr is S.false:
            return "false"
        return jscode(expr.xreplace(self.sub))

    def arr(self, exprs):
        return "[" + ", ".join(self.e(x) for x in exprs) + "]"

    def mat(self, M):
        return self.arr([M[i, j] for i in range(4) for j in range(4)])

    def fn(self, args, body):
        return f"function({args}) {{ return {body}; }}"


def _prims(spec, node, T, out, notes, body, js, flag=""):
    T = T * _placement(spec, node)
    kind = _kind(spec, node)
    v = node[kind]
    if kind == "box":
        h = _box(spec, v)
        out.append(f'{{type: "box", size: {js.arr([2 * c for c in h])}, m: {js.mat(T)}, flag: "{flag}"}}')
    elif kind == "ball":
        out.append(f'{{type: "ball", r: {js.e(spec.num(v))}, m: {js.mat(T)}, flag: "{flag}"}}')
    elif kind == "cylinder":
        r, h = _cyl(spec, v)
        out.append(f'{{type: "cylinder", r: {js.e(r)}, h: {js.e(2 * h)}, m: {js.mat(T)}, flag: "{flag}"}}')
    elif kind == "hull":
        from scipy.spatial import ConvexHull
        pts = np.array([[float(spec.num(c)) for c in p] for p in v])
        tris = [float(c) for s in ConvexHull(pts).simplices for i in s for c in pts[i]]
        out.append(f'{{type: "mesh", tris: {json.dumps(tris)}, m: {js.mat(T)}, flag: "{flag}"}}')
    elif kind == "union":
        for k in v:
            _prims(spec, k, T, out, notes, body, js, flag)
    elif kind == "intersection":
        notes.append(f"{body}: an intersection is drawn as all its parts, see-through")
        for k in v:
            _prims(spec, k, T, out, notes, body, js, "ghost")
    elif kind == "difference":
        notes.append(f"{body}: a difference is drawn without the cut")
        _prims(spec, v[0], T, out, notes, body, js, flag)
    else:
        notes.append(f"{body}: a 'where' shape (polynomial inequalities) is not drawn")


def system_js(spec, system, defaults, replay=None):
    """The JavaScript object literal for one compiled mechanism. `replay` is (seconds per frame,
    list of configurations) for a recorded motion that the page plays back instead of sampling."""
    X, P = system.variables, list(spec.params)
    js = JS(X, P)
    notes = list(system.notes[1:]) if system.notes else []
    J = Matrix(system.equations).jacobian(X) if system.equations and X else Matrix.zeros(0, len(X))
    JG = Matrix(system.inequalities).jacobian(X) if system.inequalities and X else Matrix.zeros(0, len(X))
    rows = lambda M: "[" + ", ".join(js.arr(list(M.row(i))) for i in range(M.rows)) + "]"
    qf = [f for f in system.formulas if not quantified(f)]
    if len(qf) < len(system.formulas):
        notes.append("some no-overlap conditions are quantified over points and are not checked here")
    unit_roles = {"c", "s", "c1", "s1", "c2", "s2", "q0", "q1", "q2", "q3"}
    kind_of = {v: ("unit" if r in unit_roles else "length")
               for j in system.joints if j.tree for r, v in j.values.items()}
    kinds = [kind_of.get(v, "length") for v in X]

    bodies = []
    for name, b in spec.bodies.items():
        frames = [js.arr(list(F[:3, 3])) for F in b.frames.values()]
        prims = []
        if b.shape is not None:
            _prims(spec, b.shape, eye(4), prims, notes, name, js)
        bodies.append(f'{{name: {json.dumps(name)}, color: {json.dumps(b.color)}, frames: [{", ".join(frames)}], prims: [{", ".join(prims)}]}}')
    poses = ", ".join(f"{json.dumps(b)}: {js.mat(M)}" for b, M in system.poses.items())
    coords = ", ".join(
        f'{{name: {json.dumps(j.name)}, kind: {json.dumps(j.kind)}, values: {js.fn("x, p", "{" + ", ".join(f"{json.dumps(k)}: {js.e(e)}" for k, e in j.values.items()) + "}")}}}'
        for j in system.joints if j.values)
    params = [{"name": q.name, "value": float(defaults[q.name])} for q in P]
    starts = _starts(spec, system, js)
    frames = "null" if replay is None else (
        f"{{dt: {replay[0]:g}, frames: [" + ", ".join("[" + ", ".join(f"{v:.5g}" for v in x) + "]" for x in replay[1]) + "]}")
    return f"""{{
  name: {json.dumps(spec.name)},
  vars: {json.dumps([v.name for v in X])},
  kinds: {json.dumps(kinds)},
  params: {json.dumps(params)},
  assumptions: {js.fn("p", js.e(And(*spec.assumptions)) if spec.assumptions else "true")},
  assumptionText: {json.dumps([str(a) for a in spec.assumptions])},
  F: {js.fn("x, p", js.arr(system.equations))},
  J: {js.fn("x, p", rows(J))},
  G: {js.fn("x, p", js.arr(system.inequalities))},
  JG: {js.fn("x, p", rows(JG))},
  OK: {js.fn("x, p", js.e(And(*qf)) if qf else "true")},
  poses: {js.fn("x, p", "{" + poses + "}")},
  joints: [{coords}],
  geometry: {js.fn("p", "[" + ", ".join(bodies) + "]")},
  starts: {js.fn("p", "[" + ", ".join(starts) + "]")},
  replay: {frames},
  notes: {json.dumps(notes)}
}}"""


def _starts(spec, system, js):
    """Where random starts should come from: each tree joint's angles on their circle and within
    their range, displacements within their range, rotations as unit quaternions."""
    index = {v: i for i, v in enumerate(system.variables)}
    out = []
    for j in system.joints:
        if not j.tree:
            continue
        lim = dict(j.opts.get("limits") or {})
        if "range" in j.opts:
            lim["angle" if "c" in j.values else "d"] = j.opts["range"]
        for cr, sr, name in (("c", "s", "angle"), ("c1", "s1", "angle1"), ("c2", "s2", "angle2")):
            if cr in j.values:
                lo, hi = (spec.num(v) for v in lim[name]) if name in lim else (-pi, pi)
                out.append(f'{{type: "angle", i: [{index[j.values[cr]]}, {index[j.values[sr]]}], lo: {js.e(lo)}, hi: {js.e(hi)}}}')
        for r in ("d", "x", "y", "z"):
            if r in j.values and r in lim:
                lo, hi = (spec.num(v) for v in lim[r])
                out.append(f'{{type: "length", i: {index[j.values[r]]}, lo: {js.e(lo)}, hi: {js.e(hi)}}}')
        if "q0" in j.values:
            out.append(f'{{type: "quat", i: [{", ".join(str(index[j.values[f"q{k}"]]) for k in range(4))}]}}')
    return out


def _defaults(spec, given):
    vals = {k: spec.num(v) for k, v in given.items()}
    missing = [q for q in spec.params if q.name not in vals]
    if missing:
        rng = np.random.default_rng(0)
        for _ in range(500):
            trial = dict(vals)
            trial.update({q.name: S(int(rng.integers(100, 301))) / 100 for q in missing})
            subs = {q: trial[q.name] for q in spec.params}
            if all(bool(a.subs(subs)) for a in spec.assumptions):
                return trial
        raise SpecError(f"{spec.name}: give values for {[q.name for q in missing]} (path:name=value)")
    return vals


def recording(path, spec, system):
    """A motion recorded from MuJoCo (an .npz file with `time` and `qpos`, one row per frame, from a
    model that cspace.mjcf exported from this mechanism) as (seconds per frame, configurations)."""
    from .mjcf import Exporter
    data = np.load(path)
    if "qpos" not in data or "time" not in data or len(data["time"]) < 2:
        raise SpecError(f"{path}: expected arrays 'time' and 'qpos' with at least two frames")
    ex = Exporter(spec, system)
    dt = float(np.median(np.diff(data["time"])))
    return dt, [ex.configuration(q) for q in data["qpos"]]


def page(entries, title="Configuration Explorer", standalone=True, replay=None):
    """entries: list of (spec path, {param: value}) -> HTML text. `standalone` adds the
    document skeleton; leave it off for hosts that add their own. `replay` is a recording of the
    first mechanism (see `recording`), which the page then plays back."""
    systems = []
    for n, (path, given) in enumerate(entries):
        spec = load(path, ALL_KINDS)
        for k in given:
            if k not in {q.name for q in spec.params}:
                raise SpecError(f"{path}: {k!r} is not a symbolic parameter")
        defaults = _defaults(spec, given)
        if replay and n == 0 and spec.params:      # the recording was made with these values
            spec = load(path, ALL_KINDS, {k: str(v) for k, v in defaults.items()})
        system = compile_spec(spec, shapes=False)
        systems.append(system_js(spec, system, defaults, recording(replay, spec, system) if replay and n == 0 else None))
    html = TEMPLATE.read_text()
    html = html.replace("/*SYSTEMS*/", "[\n" + ",\n".join(systems) + "\n]").replace("__TITLE__", title)
    if standalone:
        html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                '</head>\n<body>\n' + html + '\n</body>\n</html>\n')
    return html


def parse_entry(text):
    path, _, rest = text.partition(":")
    given = dict(kv.split("=", 1) for kv in rest.split(",") if kv) if rest else {}
    return path, given


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cspace.viewer", description=__doc__.split("\n\n")[0])
    ap.add_argument("specs", nargs="+", help="mechanism files, each optionally path:name=value,...")
    ap.add_argument("-o", "--output", default="explorer.html")
    ap.add_argument("--title", default="Configuration Explorer")
    ap.add_argument("--fragment", action="store_true", help="omit the document skeleton")
    ap.add_argument("--replay", metavar="FILE.npz", help="play back a motion recorded from MuJoCo for the "
                    "first mechanism (arrays 'time' and 'qpos'; examples/run_helixturm.py --record writes one)")
    args = ap.parse_args(argv)
    try:
        html = page([parse_entry(s) for s in args.specs], args.title, not args.fragment, args.replay)
    except (SpecError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    Path(args.output).write_text(html)
    print(f"written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
