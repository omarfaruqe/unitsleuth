"""Command-line interface: ``unitsleuth data.csv --col height:cm,in``."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence

import numpy as np

from unitsleuth import catalog
from unitsleuth.catalog import Conversion
from unitsleuth.multivariate import detect_rows, repair


def _resolve(spec: str) -> tuple[str, Conversion]:
    try:
        name, units = spec.rsplit(":", 1)
        a, b = (u.strip() for u in units.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad --col {spec!r}; expected NAME:FROM,TO such as height:cm,in"
        ) from None
    try:
        return name, catalog.get(a, b)
    except KeyError:
        pass
    try:
        c = catalog.get(b, a)
    except KeyError:
        raise argparse.ArgumentTypeError(
            f"no built-in conversion between {a!r} and {b!r}"
        ) from None
    return name, Conversion(c.quantity, a, b, 1.0 / c.scale, -c.offset / c.scale)


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="unitsleuth",
        description="Detect rows recorded in a different measurement unit.",
    )
    p.add_argument("csv", help="input CSV file with a header row")
    p.add_argument(
        "--col",
        action="append",
        required=True,
        type=_resolve,
        metavar="NAME:FROM,TO",
        help="column and the conversion to test, e.g. weight:kg,lb (repeatable)",
    )
    p.add_argument("--out", help="write a repaired CSV to this path")
    p.add_argument(
        "--target",
        choices=["majority", "base", "converted"],
        default="majority",
        help="unit to harmonise to when repairing (default: majority)",
    )
    p.add_argument("--min-delta-bic", type=float, default=10.0)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Exit code 1 means mixed units were detected."""
    args = build_parser().parse_args(argv)
    convs = dict(args.col)
    with open(args.csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [c for c in convs if c not in fields]
    if missing:
        print(f"error: columns not in file: {missing}", file=sys.stderr)
        return 2
    data = {c: np.array([_num(r[c]) for r in rows]) for c in convs}
    try:
        fit = detect_rows(data, convs, min_delta_bic=args.min_delta_bic)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(fit.summary())
    if args.out:
        fixed = repair(data, fit, target=args.target)
        for i, r in enumerate(rows):
            for c in convs:
                if np.isfinite(fixed[c][i]):
                    r[c] = repr(float(fixed[c][i]))
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote repaired data to {args.out}")
    return 1 if fit.detected else 0


if __name__ == "__main__":
    raise SystemExit(main())
