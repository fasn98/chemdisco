#!/usr/bin/env python3
"""End-to-end validation of the pipeline against one real target.

Run this before trusting any result from this package on a new target. It walks
the whole chain -- fetch, curate, featurise, split, train, evaluate against
baselines, generate candidates, score them -- and prints what each stage did,
including what it discarded.

The default target is BACE1 (UniProt P56817), chosen deliberately as the
validation case:

* It has thousands of measured IC50 values in ChEMBL across many chemical series,
  so a scaffold split has something to hold out.
* It is a well-studied aspartyl protease with many PDB structures, so docking can
  be added later against real geometry.
* It is a neurodegeneration target, which makes it the nearest well-populated
  neighbour to the prion-disease work this pipeline is ultimately aimed at.

A note on prion disease, since that is the motivating goal. Creutzfeldt-Jakob
disease is a hard case for this kind of pipeline, and the reasons are worth
knowing before pointing it there:

* The pathogenic species is PrP-Sc, a misfolded conformer that forms fibrillar
  aggregates. It presents no well-defined small-molecule binding pocket, and
  there is no high-resolution structure of the human form suitable for docking.
  PrP-C, the normal cellular conformer, does have solved structures and is a
  legitimate alternative target.
* The measured anti-prion data in ChEMBL comes largely from cell-based assays
  (ScN2a and similar), which report EC50 against a phenotype rather than binding
  affinity. Potency there is confounded by permeability and metabolism, and the
  curation policy in this package will treat those measurements accordingly.
* The clinical record for small molecules is a sequence of failures: quinacrine,
  pentosan polysulfate, doxycycline, flupirtine. Astemizole and anle138b showed
  effects in mouse models. The approach with the most current traction is not a
  small molecule at all but antisense reduction of PRNP expression.

None of that makes the target unreasonable to work on. It does mean a pipeline
validated only on prion data would have no way to tell a working model from a
broken one, which is why validation happens here first.

Usage:
    python scripts/validate_target.py
    python scripts/validate_target.py --accession P00533 --name EGFR
    python scripts/validate_target.py --offline      # cached data only
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from chemdisco.chem.standardize import RDKIT_AVAILABLE  # noqa: E402
from chemdisco.curate import AggregationPolicy, CurationPolicy, curate  # noqa: E402
from chemdisco.data.chembl import ChEMBLClient, ChEMBLError, ResponseCache  # noqa: E402
from chemdisco.qsar import (  # noqa: E402
    QSARModel,
    compute_baselines,
    detect_target_leakage,
    evaluate,
    make_fit_predict,
)
from chemdisco.qsar.evaluate import SplitComparison  # noqa: E402
from chemdisco.split import random_split, scaffold_split, verify_disjoint  # noqa: E402


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accession", default="P56817", help="UniProt accession")
    parser.add_argument("--name", default="BACE1", help="Human-readable target name")
    parser.add_argument(
        "--activity-types",
        default="IC50",
        help="Comma-separated ChEMBL standard_type values",
    )
    parser.add_argument("--max-records", type=int, default=8000)
    parser.add_argument("--cache", default=".cache/chembl")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip BRICS candidate generation, which is the slow stage",
    )
    parser.add_argument("--output", default="", help="Write a JSON summary here")
    args = parser.parse_args()

    if not RDKIT_AVAILABLE:
        print(
            "RDKit is not installed, so this script cannot run: it needs "
            "structures for descriptors and scaffolds.\n"
            "Install it with 'pip install rdkit' (or 'pip install -e \".[chem]\"').\n"
            "The curation, splitting and evaluation tests run without it."
        )
        return 2

    from chemdisco.chem.descriptors import combined_features
    from chemdisco.chem.scaffold import scaffold_summary, scaffolds_for_dataset
    from chemdisco.chem.standardize import inchikey_of, silence_rdkit_logs

    silence_rdkit_logs()
    started = time.monotonic()
    summary: dict[str, object] = {"target": args.name, "accession": args.accession}

    # -- 1. Retrieval ------------------------------------------------------
    heading(f"1. Retrieving {args.name} ({args.accession}) activity data from ChEMBL")
    cache = ResponseCache(pathlib.Path(args.cache))
    client = ChEMBLClient(cache=cache, offline=args.offline)

    try:
        records, provenance = client.fetch_records(
            args.accession,
            activity_types=tuple(args.activity_types.split(",")),
            max_records=args.max_records,
        )
    except ChEMBLError as error:
        print(f"Retrieval failed: {error}")
        return 1

    release = client.release()
    print(f"ChEMBL release: {release or 'unknown'}")
    print(f"Target resolved to: {provenance['target_chembl_id']}")
    print(f"Activities retrieved: {len(records)}")
    print(f"Distinct assays: {provenance['n_distinct_assays']}")
    print(cache.describe())
    summary["retrieval"] = provenance

    if len(records) < 100:
        print(
            f"\nOnly {len(records)} activities. This target does not have enough "
            "measured data for the validation to mean anything. Pick a "
            "better-populated target to validate the pipeline, then come back."
        )
        return 1

    # -- 2. Curation -------------------------------------------------------
    heading("2. Curating: which measurements are defensible")
    policy = CurationPolicy()
    aggregation = AggregationPolicy()
    print(f"Curation policy: {policy.describe()}")
    print(f"Aggregation policy: {aggregation.describe()}\n")

    # Group replicates by standardised structure rather than ChEMBL id, so salt
    # forms of one compound do not survive as separate rows and leak across the
    # split.
    def structure_key(record):
        return inchikey_of(record.smiles) or f"unparseable:{record.compound_id}"

    report = curate(
        records, policy=policy, aggregation=aggregation, compound_key=structure_key
    )
    print(report.describe())
    summary["curation"] = {
        "n_input": report.n_input,
        "n_kept": report.n_kept,
        "n_rejected": report.n_rejected,
        "retention": report.retention,
        "rejections": report.rejection_summary(),
    }

    if report.n_kept < 100:
        print(
            f"\nCuration left {report.n_kept} compounds. Below about 100 a QSAR "
            "model cannot be validated meaningfully. Inspect the rejection "
            "breakdown above before loosening the policy -- the filters may be "
            "correct and the data genuinely thin."
        )
        return 1

    smiles = [point.smiles for point in report.kept]
    y = np.array([point.pactivity.require() for point in report.kept], dtype=float)
    print(
        f"\npActivity range {y.min():.2f} to {y.max():.2f}, "
        f"mean {y.mean():.2f}, sd {y.std():.2f}"
    )
    if y.std() < 0.5:
        print(
            "  WARNING: under 0.5 log units of spread. There is very little signal "
            "to model and R-squared will be unstable."
        )

    # -- 3. Chemical diversity --------------------------------------------
    heading("3. Scaffold diversity: can a split measure generalisation at all")
    print(scaffold_summary(smiles))
    scaffolds, acyclic, unparseable = scaffolds_for_dataset(smiles)
    summary["diversity"] = {
        "n_distinct_scaffolds": len({s for s in scaffolds if s}),
        "n_acyclic": len(acyclic),
        "n_unparseable": len(unparseable),
    }

    # -- 4. Featurisation --------------------------------------------------
    heading("4. Featurising")
    X, feature_names, valid, errors = combined_features(smiles, n_bits=1024)
    print(f"Feature matrix: {X.shape[0]} x {X.shape[1]}")
    print(f"Valid rows: {int(valid.sum())} of {len(valid)}")
    for error in errors[:5]:
        print(f"  {error}")

    if not valid.all():
        # Dropped rather than imputed, and the loss is stated.
        keep = np.where(valid)[0]
        print(f"  dropping {len(valid) - len(keep)} compound(s) with failed descriptors")
        X, y = X[keep], y[keep]
        smiles = [smiles[i] for i in keep]
        scaffolds = [scaffolds[i] for i in keep]

    # -- 5. Leakage screen -------------------------------------------------
    heading("5. Screening for target leakage")
    warnings = detect_target_leakage(X, y, list(feature_names))
    if warnings:
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("  No single feature determines the label.")
    print(
        "  Note: this screen cannot detect a label synthesised from several\n"
        "  features. The real guarantee is that every label here descends from a\n"
        "  measured ChEMBL value, enforced by chemdisco.qsar.dataset."
    )
    summary["leakage_warnings"] = warnings

    # -- 6. Scaffold split vs random split --------------------------------
    heading("6. Training and evaluating: scaffold split against random split")
    try:
        split = scaffold_split(scaffolds, test_fraction=0.2, seed=0)
    except Exception as error:
        print(f"Scaffold split failed: {error}")
        return 1

    leaked = verify_disjoint(split, scaffolds)
    print(split.describe())
    print(f"Scaffolds leaking across partitions: {len(leaked)} (must be 0)")
    if leaked:
        print("  ABORTING: the split leaks, so any metric from it is invalid.")
        return 1

    results: dict[str, object] = {}
    evaluations = {}

    for label, indices in (
        ("scaffold", split),
        ("random", random_split(len(y), test_fraction=0.2, seed=0)),
    ):
        train_idx, test_idx = list(indices.train), list(indices.test)
        model = QSARModel(
            kind="random_forest",
            label_name=f"p{args.activity_types.split(',')[0]} ({args.name})",
            curation_summary=policy.describe(),
        ).fit(X[train_idx], y[train_idx], feature_names=list(feature_names))

        predictions = model.predict_raw(X[test_idx])
        baselines = compute_baselines(
            y[train_idx],
            y[test_idx],
            fit_predict=make_fit_predict(
                "random_forest", X[train_idx], X[test_idx]
            ),
            n_permutations=10,
        )
        evaluations[label] = evaluate(
            y[test_idx],
            predictions,
            n_train=len(train_idx),
            split_strategy=indices.strategy,
            baselines=baselines,
            leakage_warnings=warnings,
        )

    comparison = SplitComparison(
        scaffold=evaluations["scaffold"], random=evaluations["random"]
    )
    print("\n" + comparison.describe())

    results["scaffold_r2"] = evaluations["scaffold"].r2.estimate
    results["scaffold_r2_interval"] = [
        evaluations["scaffold"].r2.low,
        evaluations["scaffold"].r2.high,
    ]
    results["scaffold_rmse"] = evaluations["scaffold"].rmse.estimate
    results["random_r2"] = evaluations["random"].r2.estimate
    results["leakage_gap"] = comparison.leakage_gap
    results["is_defensible"] = evaluations["scaffold"].is_defensible
    summary["evaluation"] = results

    # -- 7. Generation -----------------------------------------------------
    if not args.skip_generation:
        heading("7. Generating candidates from the most potent known actives")
        from chemdisco.generate import GenerationPolicy, generate_candidates, score_candidates

        train_idx = list(split.train)
        order = np.argsort(y[train_idx])[::-1]
        seed_actives = [smiles[train_idx[i]] for i in order[:40]]
        print(
            f"Using the {len(seed_actives)} most potent training compounds as the "
            "fragment source."
        )

        final_model = QSARModel(
            kind="random_forest",
            label_name=f"p{args.activity_types.split(',')[0]} ({args.name})",
            curation_summary=policy.describe(),
        ).fit(X[train_idx], y[train_idx], feature_names=list(feature_names))

        generation = generate_candidates(
            seed_actives,
            policy=GenerationPolicy(max_generated=1500),
            reference_smiles=smiles,
            seed=0,
        )
        generation = score_candidates(
            generation,
            final_model,
            lambda candidate_smiles: combined_features(candidate_smiles, n_bits=1024)[0],
            training_smiles=[smiles[i] for i in train_idx],
        )
        print("\n" + generation.describe())

        ranked = generation.ranked()
        if ranked:
            print(f"\nTop {min(5, len(ranked))} rankable candidates:")
            for candidate in ranked[:5]:
                print("\n" + candidate.summary())
        summary["generation"] = {
            "n_fragments": generation.n_fragments,
            "n_generated": generation.n_generated,
            "n_retained": len(generation.candidates),
            "n_rankable": len(ranked),
            "attrition": generation.attrition,
        }

    # -- Verdict -----------------------------------------------------------
    heading("Verdict")
    scaffold_result = evaluations["scaffold"]
    if scaffold_result.is_defensible:
        print(
            f"The pipeline produced a defensible model for {args.name}: "
            f"R2 {scaffold_result.r2.label()} on held-out scaffolds, clearing "
            "every baseline."
        )
    else:
        print(
            f"The model for {args.name} did NOT clear the bar for a defensible "
            "result. The baseline comparison above says which check it failed. "
            "This is information, not a malfunction -- publish the number with "
            "that caveat or not at all."
        )
    print(f"\nElapsed: {time.monotonic() - started:.1f}s")

    if args.output:
        path = pathlib.Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"Summary written to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
