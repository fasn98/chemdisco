#!/usr/bin/env python3
"""Probe ChEMBL endpoint forms to find which ones actually work.

Written because the target lookup this package relied on started returning
HTTP 500. A public API's documented filter syntax and its working filter syntax
are not always the same thing, and the difference is only discoverable by asking
it.

Kept in the repository rather than thrown away after use: ChEMBL's REST interface
changes between releases, and when the next lookup breaks this script is how the
replacement gets found. Run it from CI, where the network is unrestricted:

    python scripts/probe_chembl.py --accession P56817 --gene BACE1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from urllib.parse import urlencode

import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data"
TIMEOUT = 60
PAUSE = 0.5


def probe(label: str, path: str, params: dict[str, object]) -> dict[str, object]:
    """Request one endpoint form and summarise what came back."""
    url = f"{BASE}/{path}?{urlencode(params)}"
    result: dict[str, object] = {"label": label, "url": url}
    try:
        response = requests.get(
            url,
            timeout=TIMEOUT,
            headers={
                "Accept": "application/json",
                "User-Agent": "chemdisco-probe/0.1 (research use)",
            },
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        return result

    result["status"] = response.status_code
    if response.status_code != 200:
        # The body of an error response usually names the offending field, which
        # is the whole point of probing.
        result["body"] = response.text[:400]
        return result

    try:
        payload = response.json()
    except Exception:
        result["error"] = "response was not JSON"
        result["body"] = response.text[:200]
        return result

    collection = next(
        (
            key
            for key in ("targets", "activities", "assays", "molecules")
            if key in payload
        ),
        None,
    )
    if collection is None:
        result["keys"] = sorted(payload)[:10]
        return result

    items = payload.get(collection, [])
    result["collection"] = collection
    result["count"] = len(items)
    result["total"] = payload.get("page_meta", {}).get("total_count")
    result["sample"] = [
        {
            key: item.get(key)
            for key in (
                "target_chembl_id",
                "target_type",
                "pref_name",
                "organism",
                "assay_chembl_id",
                "confidence_score",
                "assay_type",
            )
            if key in item
        }
        for item in items[:4]
    ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accession", default="P56817")
    parser.add_argument("--gene", default="BACE1")
    parser.add_argument("--known-target", default="CHEMBL4822")
    args = parser.parse_args()

    probes = [
        # Baseline: does the API answer at all?
        ("status", "status.json", {}),
        ("molecule by id (known good)", "molecule/CHEMBL25.json", {}),
        # The form that started failing.
        (
            "target by component accession (current code)",
            "target.json",
            {"target_components__accession": args.accession, "limit": 50, "offset": 0},
        ),
        # Same filter without pagination parameters, to see whether those are
        # implicated rather than the filter itself.
        (
            "target by component accession, no paging",
            "target.json",
            {"target_components__accession": args.accession},
        ),
        # Singular relation name, in case the plural is wrong.
        (
            "target by component accession (singular relation)",
            "target.json",
            {"target_component__accession": args.accession, "limit": 5},
        ),
        # The documented nested lookup through target_component.
        (
            "target_component by accession",
            "target_component.json",
            {"accession": args.accession, "limit": 5},
        ),
        # Free-text search, which uses a different backend entirely.
        ("target search by gene", "target/search.json", {"q": args.gene, "limit": 5}),
        # Name filters.
        (
            "target by pref_name",
            "target.json",
            {"pref_name__icontains": args.gene, "limit": 5},
        ),
        (
            "target by synonym",
            "target.json",
            {"target_synonym__icontains": args.gene, "limit": 5},
        ),
        # Does the activity endpoint still work with the filters curation needs?
        (
            "activities for a known target",
            "activity.json",
            {
                "target_chembl_id": args.known_target,
                "standard_type__in": "IC50",
                "canonical_smiles__isnull": "false",
                "limit": 3,
            },
        ),
        (
            "activities without the isnull filter",
            "activity.json",
            {
                "target_chembl_id": args.known_target,
                "standard_type__in": "IC50",
                "limit": 3,
            },
        ),
        (
            "activities with a single standard_type",
            "activity.json",
            {"target_chembl_id": args.known_target, "standard_type": "IC50", "limit": 3},
        ),
        # The assay join that supplies confidence_score.
        (
            "assay batch lookup",
            "assay.json",
            {"assay_chembl_id__in": "CHEMBL829584,CHEMBL829585", "limit": 5},
        ),
    ]

    results = []
    for label, path, params in probes:
        outcome = probe(label, path, params)
        results.append(outcome)
        status = outcome.get("status", outcome.get("error", "?"))
        summary = ""
        if outcome.get("count") is not None:
            summary = f" -> {outcome['count']} item(s), total={outcome.get('total')}"
        print(f"[{status}] {label}{summary}")
        if outcome.get("sample"):
            for item in outcome["sample"]:  # type: ignore[union-attr]
                print(f"       {item}")
        if outcome.get("body"):
            print(f"       body: {outcome['body']}")
        if outcome.get("keys"):
            print(f"       keys: {outcome['keys']}")
        time.sleep(PAUSE)

    print("\n=== JSON ===")
    print(json.dumps(results, indent=2, default=str))

    working = [r for r in results if r.get("status") == 200 and r.get("count")]
    print(f"\n{len(working)} of {len(results)} probes returned usable records.")
    # Always exit 0: a probe reporting that everything failed is a successful
    # diagnostic run, and failing the job would hide the report.
    return 0


if __name__ == "__main__":
    sys.exit(main())
