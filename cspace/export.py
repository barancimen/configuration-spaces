"""Write a System as JSON, plus a text summary."""
import json

from sympy import sstr


def to_json(system):
    s = system
    data = {
        "name": s.name,
        "coords": s.coords,
        "trig": s.trig,
        "parameters": [sstr(p) for p in s.params],
        "assumptions": [sstr(a) for a in s.assumptions],
        "variables": [sstr(v) for v in s.variables],
        "equations": [sstr(e) for e in s.equations],
        "inequalities": [sstr(g) for g in s.inequalities],
        "formulas": [sstr(f) for f in s.formulas],
        "poses": {b: [[sstr(M[i, j]) for j in range(4)] for i in range(4)] for b, M in s.poses.items()},
        "coordinates": {j.name: {k: sstr(e) for k, e in j.values.items()} for j in s.joints},
        "shapes": {b: sstr(f) for b, f in s.shapes.items()},
        "notes": s.notes,
    }
    return json.dumps(data, indent=1) + "\n"


def summary(system):
    s = system
    lines = [f"{s.name}: {len(s.variables)} variables, {len(s.equations)} equations, "
             f"{len(s.inequalities)} inequalities, {len(s.formulas)} other conditions "
             f"({s.coords} coordinates{', trig' if s.trig else ''})",
             f"naive freedom count (6 per body minus each constraint's codimension): {s.naive_dof}"]
    lines += ["  " + n for n in s.notes]
    return "\n".join(lines)
