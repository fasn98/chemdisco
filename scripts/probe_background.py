"""Does a background from unrelated targets let the feature check discriminate?

A measurement, not a test. The question has one honest answer per target and it
may be no: if the generic background does not separate the anchoring motif from
the incidental ones either, the conserved-feature check stays withheld and that
is the result.

What makes an answer here meaningful is the pair of features watched. On BACE1:

* ``amidine`` (or the basic-nitrogen family) is what engages the catalytic
  aspartyl dyad. Every known inhibitor carries one. If the background works, this
  should be strongly enriched.
* ``halogen_on_aromatic`` is incidental. It is in most of the actives because it
  is in most modern medicinal chemistry. If the background works, this should
  *not* be enriched -- and the previous background, drawn from the target's own
  weak binders, gave it 1.8x against amidine's 1.9x, which is why a warheadless
  biaryl passed.

So the probe is a two-sided check. A background that enriches everything is as
useless as one that enriches nothing; the point is the gap between them.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from chemdisco.curate import CurationPolicy, curate  # noqa: E402
from chemdisco.data.background import (  # noqa: E402
    fetch_unrelated_background,
    summarise,
)
from chemdisco.data.chembl import ChEMBLClient, ChEMBLError, ResponseCache  # noqa: E402
from chemdisco.generate.pharmacophore import (  # noqa: E402
    FeatureProfile,
    profile_actives,
)

#: Features expected to anchor an aspartyl-protease ligand, and features expected
#: to be incidental. Used only to report whether the measurement came out the way
#: the chemistry says it should -- not to filter anything.
EXPECTED_ANCHORS = ("amidine", "guanidine", "basic_nitrogen_any",
                    "primary_aliphatic_amine", "secondary_aliphatic_amine")
EXPECTED_INCIDENTAL = ("halogen_on_aromatic", "aromatic_ring", "fused_aromatic",
                       "amide")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accession", default="P56817")
    parser.add_argument("--name", default="BACE1")
    parser.add_argument("--active-threshold", type=float, default=8.0)
    parser.add_argument("--n-actives", type=int, default=30)
    parser.add_argument("--background-size", type=int, default=1200)
    parser.add_argument("--max-records", type=int, default=8000)
    parser.add_argument("--cache", default=".cache")
    args = parser.parse_args()

    from chemdisco.chem.standardize import inchikey_of, silence_rdkit_logs

    silence_rdkit_logs()

    client = ChEMBLClient(cache=ResponseCache(pathlib.Path(args.cache) / "chembl"))

    print(f"=== Curating {args.name} ({args.accession}) ===")
    try:
        records, _ = client.fetch_records(
            args.accession, activity_types=("IC50",), max_records=args.max_records
        )
    except ChEMBLError as error:
        print(f"Retrieval failed: {error}")
        return 1

    report = curate(
        records,
        policy=CurationPolicy(),
        compound_key=lambda record: inchikey_of(record.smiles)
        or f"unparseable:{record.compound_id}",
    )
    points = sorted(report.kept, key=lambda p: -p.pactivity.require())
    actives = [
        p for p in points if p.pactivity.require() >= args.active_threshold
    ][: args.n_actives]
    print(f"  {len(points)} curated, {len(actives)} actives profiled")
    if len(actives) < 10:
        print("Too few actives to profile.")
        return 1

    print("\n=== Fetching the unrelated-target background ===")
    background = fetch_unrelated_background(
        client,
        exclude_molecule_ids=[p.compound_id for p in points],
        per_target=max(args.background_size // 6, 50),
        max_total=args.background_size,
    )
    print(background.describe())

    print("\n=== Profiling against it ===")
    profile = profile_actives(
        [p.smiles for p in actives], background_smiles=list(background.smiles)
    )
    print(profile.describe())

    # The two-sided comparison. Printed whichever way it comes out.
    def best(names: tuple[str, ...]) -> tuple[str, float]:
        found = ("none", 0.0)
        for name in names:
            if profile.prevalence.get(name, 0.0) <= 0:
                continue
            ratio = profile.enrichment(name)
            if ratio is None:
                continue
            value = 1000.0 if ratio == float("inf") else ratio
            if value > found[1]:
                found = (name, value)
        return found

    anchor_name, anchor_ratio = best(EXPECTED_ANCHORS)
    incidental_name, incidental_ratio = best(EXPECTED_INCIDENTAL)
    floor = FeatureProfile.MIN_DISCRIMINATING_ENRICHMENT

    print("\n=== The two-sided check ===")
    print(
        f"  best expected anchor:     {anchor_name} at "
        f"{'absent from background' if anchor_ratio >= 1000 else f'{anchor_ratio:.1f}x'}"
    )
    print(
        f"  best expected incidental: {incidental_name} at "
        f"{'absent from background' if incidental_ratio >= 1000 else f'{incidental_ratio:.1f}x'}"
    )
    print(f"  discrimination floor:     {floor:.1f}x")

    separates = anchor_ratio >= floor and incidental_ratio < floor
    if separates:
        print(
            "\n  SEPARATES. The anchoring motif clears the floor and the "
            "incidental one does not, so the check can tell them apart and is "
            "applied rather than withheld."
        )
    elif anchor_ratio >= floor and incidental_ratio >= floor:
        print(
            "\n  DOES NOT SEPARATE. Both clear the floor, so the check would "
            "pass a candidate carrying only the incidental feature -- exactly "
            "the failure that produced the warheadless biaryl. A background "
            "that enriches everything discriminates nothing."
        )
    elif anchor_ratio < floor:
        print(
            "\n  DOES NOT SEPARATE. The anchoring motif does not clear the "
            "floor, so nothing stands out and the check stays withheld. That is "
            "a negative result about this background, not about the chemistry."
        )

    print("\n=== JSON ===")
    print(
        json.dumps(
            {
                "target": args.name,
                "n_actives": profile.n_actives,
                "background": summarise(background),
                "conserved": list(profile.conserved),
                "most_discriminating": list(profile.most_discriminating),
                "ubiquitous": list(profile.ubiquitous),
                "can_discriminate": profile.can_discriminate,
                "best_anchor": [anchor_name, round(anchor_ratio, 2)],
                "best_incidental": [incidental_name, round(incidental_ratio, 2)],
                "separates": separates,
            },
            indent=2,
        )
    )
    # Exit 0 either way: a negative result is a result, and a red job would
    # suggest the probe failed rather than that the background did not work.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
