#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


KEY_FIELDS = ("problem_type", "dataset", "method", "setting", "budget")


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return [], []
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_rows(path: Path | None, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output = sys.stdout if path is None or str(path) == "-" else path.open("w", newline="")
    try:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if output is not sys.stdout:
            output.close()


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in KEY_FIELDS)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Overlay corrected chordal fair metric rows onto a base normalized CSV."
    )
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--correction-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    base_fields, base_rows = read_rows(args.base_csv)
    correction_fields, correction_rows = read_rows(args.correction_csv)

    fields = list(base_fields)
    seen = set(fields)
    for field in correction_fields:
        if field not in seen:
            fields.append(field)
            seen.add(field)

    corrections = {row_key(row): row for row in correction_rows}
    written: set[tuple[str, ...]] = set()
    rows: list[dict[str, str]] = []
    for row in base_rows:
        key = row_key(row)
        rows.append(corrections.get(key, row))
        written.add(key)

    for row in correction_rows:
        key = row_key(row)
        if key not in written:
            rows.append(row)
            written.add(key)

    write_rows(args.output, fields, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
