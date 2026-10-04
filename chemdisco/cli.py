"""Command-line entry point.

Thin by design: the library is the product and the CLI is one way to drive it.
Each subcommand prints what it discarded, because an attrition count is usually
more informative than the surviving rows.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def _cmd_fetch(args: argparse.Namespace) -> int:
    from .curate import curate
    from .data.chembl import ChEMBLClient, ChEMBLError, ResponseCache

    client = ChEMBLClient(
        cache=ResponseCache(pathlib.Path(args.cache)), offline=args.offline
    )
    try:
        records, provenance = client.fetch_records(
            args.accession,
            activity_types=tuple(args.activity_types.split(",")),
            max_records=args.max_records,
        )
    except ChEMBLError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Retrieved {len(records)} activities for {args.accession}")
    print(f"  target: {provenance['target_chembl_id']}")

    report = curate(records)
    print()
    print(report.describe())

    if args.out:
        import csv

        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [point.as_row() for point in report.kept]
        if rows:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nWrote {len(rows)} curated points to {path}")
        else:
            print("\nNothing survived curation; no file written.")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from scripts.validate_target import main as validate_main  # type: ignore

    sys.argv = ["validate_target.py", "--accession", args.accession, "--name", args.name]
    if args.offline:
        sys.argv.append("--offline")
    return validate_main()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chemdisco",
        description=(
            "Auditable computational drug discovery. Every reported number "
            "carries its provenance."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser(
        "fetch", help="Retrieve and curate activity data for a target"
    )
    fetch.add_argument("accession", help="UniProt accession, e.g. P56817 for BACE1")
    fetch.add_argument("--activity-types", default="IC50")
    fetch.add_argument("--max-records", type=int, default=8000)
    fetch.add_argument("--cache", default=".cache/chembl")
    fetch.add_argument("--offline", action="store_true")
    fetch.add_argument("--out", default="", help="Write curated points to this CSV")
    fetch.set_defaults(func=_cmd_fetch)

    validate = subparsers.add_parser(
        "validate", help="Run the end-to-end validation for a target"
    )
    validate.add_argument("--accession", default="P56817")
    validate.add_argument("--name", default="BACE1")
    validate.add_argument("--offline", action="store_true")
    validate.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
