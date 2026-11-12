"""python3 -m cspace spec.yaml [-o out.wl|out.json|-] [--coords reduced|maximal] [--trig] [--check]"""
import argparse
import sys
from pathlib import Path

from .compile import ALL_KINDS, compile_spec
from .export import summary, to_json, to_wl
from .spec import SpecError, load


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cspace", description="Compile a mechanism description "
                                 "into the algebraic equations of its configuration space.")
    ap.add_argument("spec", help="mechanism file (YAML)")
    ap.add_argument("-o", "--output", help="output file; '-' for standard output "
                    "(default: <spec name>.wl in the current directory)")
    ap.add_argument("--format", choices=["wl", "json"], help="default: from the output suffix, else wl")
    ap.add_argument("--coords", choices=["reduced", "maximal"], default="reduced",
                    help="reduced: joint coordinates on a spanning tree (default); "
                         "maximal: every body gets 3 positions + a unit quaternion")
    ap.add_argument("--trig", action="store_true",
                    help="tree hinge angles as Cos/Sin of an angle variable instead of (c, s) pairs")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a parameter, e.g. --set l1=3 (repeatable)")
    ap.add_argument("--check", action="store_true",
                    help="also find sample points numerically and report the local dimension")
    args = ap.parse_args(argv)

    try:
        overrides = dict(item.split("=", 1) for item in args.set)
    except ValueError:
        ap.error("--set expects NAME=VALUE")
    try:
        spec = load(args.spec, ALL_KINDS, overrides)
        system = compile_spec(spec, args.coords, args.trig)
    except SpecError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    fmt = args.format or ("json" if (args.output or "").endswith(".json") else "wl")
    text = to_json(system) if fmt == "json" else to_wl(system)
    out = args.output or f"{Path(args.spec).stem}.{fmt}"
    if out == "-":
        sys.stdout.write(text)
    else:
        Path(out).write_text(text)
    print(summary(system) + ("" if out == "-" else f"\nwritten to {out}"), file=sys.stderr)
    if args.check:
        from .check import report
        print(report(system), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
