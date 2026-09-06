"""Generate a gravity table JSON for the brachistochrone solver.

The solver's x grid runs from 0 to ``x_max`` inclusive in steps that divide the
range exactly, so it has ``round(x_max / x_step) + 1`` columns.  A gravity table
must carry exactly one value per column; this tool computes that count for you
and evaluates a chosen function onto it.

Examples
--------
Constant field, matching the solver default::

    python make_gravity_json.py --x-max 157.08 --x-step 5 --func const:9.81 -o g.json

Earth gravity with a mild horizontal gradient::

    python make_gravity_json.py --x-max 157.08 --x-step 5 \\
        --func "poly:9.81,0.02,-1e-4" -o g.json

A ripple, offset so it stays positive::

    python make_gravity_json.py --x-max 157.08 --x-step 5 \\
        --func "sin:2,0.05,0,9.81" -o g.json

Then feed it back in::

    python brachistochrone.py --gravity-json g.json
"""

from __future__ import annotations

import argparse
import math
from typing import Sequence

import numpy as np

from gravity import (
    FAMILY_HELP,
    check_positive,
    field_from_spec,
    write_json,
)


def grid_for(x_max: float, x_step: float) -> np.ndarray:
    """The solver's x grid: 0 to x_max inclusive, steps snapped to fit exactly.

    Mirrors the construction in ``brachistochrone.solve`` so a table generated
    here always matches the grid the solver builds for the same arguments.
    """
    nx = max(2, int(round(x_max / x_step)) + 1)
    return np.linspace(0.0, x_max, nx)


def _build_parser() -> argparse.ArgumentParser:
    families = "\n".join(f"    {line}" for line in FAMILY_HELP.values())
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="function families:\n" + families,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--x-max", type=float, required=True,
                   help="horizontal extent of the grid (m)")
    p.add_argument("--x-step", type=float, required=True,
                   help="requested horizontal step (m); snapped to fit exactly")
    p.add_argument("--func", required=True,
                   help='function spec, e.g. "poly:9.81,0.02" or "sin:2,0.05,0,9.81"')
    p.add_argument("-o", "--out", default="gravity.json", help="output path")
    p.add_argument("--allow-non-positive", action="store_true",
                   help="skip the g > 0 check (the solver will still reject it)")
    p.add_argument("--preview", type=int, default=6,
                   help="how many sample rows to print (0 to silence)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.x_max <= 0:
        parser.error(f"--x-max must be positive, got {args.x_max:g}")
    if args.x_step <= 0:
        parser.error(f"--x-step must be positive, got {args.x_step:g}")
    if args.x_step > args.x_max:
        parser.error(
            f"--x-step ({args.x_step:g}) exceeds --x-max ({args.x_max:g}); "
            "the grid would collapse to a single interval"
        )

    x = grid_for(args.x_max, args.x_step)
    try:
        field = field_from_spec(args.func, x)
    except ValueError as exc:
        parser.error(str(exc))

    if not args.allow_non_positive:
        try:
            check_positive(field)
        except ValueError as exc:
            parser.error(str(exc))

    realised = float(x[1] - x[0])
    payload = write_json(args.out, args.x_max, realised, field.g, field.label)

    print(f"wrote {args.out}")
    print(f"  function    {field.label}")
    print(f"  grid        {x.size} columns, x = 0 .. {args.x_max:g} m")
    if not math.isclose(realised, args.x_step, rel_tol=1e-9):
        print(f"  x_step      {realised:g} m  (requested {args.x_step:g}, snapped to fit)")
    else:
        print(f"  x_step      {realised:g} m")
    print(f"  g range     [{field.g.min():g}, {field.g.max():g}]  mean {field.mean:g}")

    if args.preview > 0:
        print()
        print(f"  {'x (m)':>10} {'g (m/s^2)':>12}")
        step = max(1, x.size // args.preview)
        for k in range(0, x.size, step):
            print(f"  {x[k]:10.3f} {field.g[k]:12.4f}")
        if (x.size - 1) % step:
            print(f"  {x[-1]:10.3f} {field.g[-1]:12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
