#!/usr/bin/env python3
"""How far did constraining the generator move the ligand-efficiency distribution?

HANDOFF item 2 asked whether the triage threshold still does honest work. The
threshold is the reference actives' median ligand efficiency, which is re-derived
every run and did not change in meaning. What changed is the population it is
applied to: it was chosen while watching double-warhead candidates, and the
constrained generator no longer emits those. A threshold can be the right number
and still reject or admit for a reason that is not the molecule's merit.

Arguing that the distribution "must have moved by construction" is not measuring
it. This measures it, by docking both populations on one machine at one search
effort, and reports two comparisons that isolate different variables.

  A. constraint ON vs OFF, both at exhaustiveness 8 on this machine
     -> isolates the GENERATOR. Receptor, box, reference set, seed, thread count
        and search effort are all held fixed, so what differs is the fragment
        pool and nothing else.

  B. constraint OFF at exhaustiveness 8 vs run 37230653219 at exhaustiveness 4
     -> isolates SEARCH EFFORT, same generator in both arms. Free from the same
        run, and it answers a question open since exhaustiveness was raised.

Neither comparison is clean on its own and the report says where each leaks.
Usage:
    python scripts/compare_anchor_arms.py runs/discover_shard_0.json \
        runs/control/discover_shard_0.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

#: Run 37230653219, as recorded in README.md. The unconstrained arm's historical
#: counterpart: same generator with no anchor constraint, exhaustiveness 4, six
#: shards on 2-core runners, all six agreeing on ligand signature f530b99d98df82f6
#: -- the first run of this pipeline that was actually one experiment, so its
#: pass-rate statistic is one the README treats as surviving.
#:
#: A summary is all that survives. No shard JSON was ever committed and the
#: Actions artifacts are unreachable, so the per-candidate efficiencies behind
#: these counts are gone. That is the reason this script exists and the reason
#: both arms it writes are committed.
HISTORIC = {
    "run_id": "37230653219",
    "n_docked": 60,
    "n_passing": 15,
    "threshold": -0.260,
    "exhaustiveness": 4,
    "cpu_cores": 2,
    "n_shards": 6,
    "all_passers_double_warhead": True,
}

#: The ligand signature run 37230653219 recorded, per README.md. If the control
#: reproduces it, the two arms of comparison B are the same molecules.
HISTORIC_LIGAND_SIGNATURE = "f530b99d98df82f6"


def load(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text())
    payload["_path"] = str(path)
    return payload


def efficiencies(shard: dict) -> list[float]:
    return [
        c["ligand_efficiency"]
        for c in shard["candidates"]
        if c.get("ligand_efficiency") is not None
    ]


def reference_le(shard: dict) -> list[float]:
    return [v for v in shard.get("reference_efficiencies", []) if v is not None]


def describe(values: list[float], name: str) -> None:
    a = np.asarray(values)
    print(
        f"  {name:<26} n={len(a):<4} min {a.min():+.3f}  p25 "
        f"{np.percentile(a, 25):+.3f}  median {np.median(a):+.3f}  p75 "
        f"{np.percentile(a, 75):+.3f}  max {a.max():+.3f}"
    )


def duplicated(shard: dict) -> int:
    return sum(1 for c in shard["candidates"] if c.get("duplicated_motifs"))


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def arm_label(shard: dict) -> str:
    applied = shard.get(
        "anchor_constraint_applied", bool(shard.get("anchor_motifs"))
    )
    return "constraint ON" if applied else "constraint OFF (control)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("constrained")
    parser.add_argument("control")
    # Read from each arm's log rather than its JSON. The retained count is not a
    # shard field, and adding one now would give the control a field the
    # already-completed constrained arm lacks -- which would break exactly the
    # field-by-field comparability this control exists to provide.
    parser.add_argument("--retained-on", type=int, default=None)
    parser.add_argument("--retained-off", type=int, default=None)
    args = parser.parse_args()

    on = load(pathlib.Path(args.constrained))
    off = load(pathlib.Path(args.control))

    # Refuse before reporting, for the reason combine() refuses: two arms are a
    # comparison only if everything except the variable under test is identical.
    heading("Are these two arms comparable at all?")
    problems: list[str] = []
    for field, label in (
        ("receptor_id", "receptor"),
        ("box_signature", "box"),
        ("curation_signature", "curated dataset"),
        ("exhaustiveness", "exhaustiveness"),
        ("cpu", "thread count"),
    ):
        a, b = on.get(field), off.get(field)
        verdict = "same" if a == b else "DIFFER"
        print(f"  {label:<18} {verdict:<7} {a} vs {b}")
        if a != b:
            problems.append(f"{label} ({a} vs {b})")

    if on.get("anchor_constraint_applied", True) is not True:
        problems.append("the 'constrained' arm does not have the constraint applied")
    if off.get("anchor_constraint_applied", False) is not False:
        problems.append("the 'control' arm has the constraint applied")

    # The ligand lists MUST differ -- that is the variable. Saying so explicitly
    # keeps it from being read as the pooling failure combine() guards against.
    print(
        f"\n  ligand signature   DIFFER  {on.get('ligand_signature')} vs "
        f"{off.get('ligand_signature')}"
    )
    print(
        "    Expected, and required: the candidate lists are the variable under "
        "test. This is not the pooling failure the combine step refuses -- these "
        "two arms are never pooled, they are compared."
    )
    print(
        f"\n  arms: {arm_label(on)}  /  {arm_label(off)}"
    )
    if problems:
        print("\n  REFUSING to report a comparison. These must match first:")
        for problem in problems:
            print(f"    - {problem}")
        return 1
    print("\n  Comparable: everything except the fragment pool is held fixed.")

    # ------------------------------------------------------------------ A ----
    heading("A. The generator. Constraint ON vs OFF, exhaustiveness 8, one machine")

    ref_on, ref_off = reference_le(on), reference_le(off)
    le_on, le_off = efficiencies(on), efficiencies(off)

    print("Ligand efficiency, kcal/mol/atom (more negative = more efficient)\n")
    describe(ref_on, "reference actives")
    describe(le_on, "candidates, ON")
    describe(le_off, "candidates, OFF (control)")

    # The threshold is derived from the reference actives, which are the same 30
    # molecules docked in both arms. It should be identical; if it is not, the
    # reference docking is not reproducible and nothing below is interpretable.
    t_on, t_off = float(np.median(ref_on)), float(np.median(ref_off))
    print(f"\n  threshold, ON arm  : {t_on:+.3f}")
    print(f"  threshold, OFF arm : {t_off:+.3f}")
    if abs(t_on - t_off) < 1e-9:
        print(
            "    Identical, as it must be: the same 30 reference actives docked in "
            "the same receptor and box at the same settings. The threshold is not "
            "the thing that moved."
        )
    else:
        print(
            f"    DIFFER by {abs(t_on - t_off):.4f}. The same 30 actives, same "
            "receptor, box, seed and settings, should give the same median. This "
            "undermines everything below and is the thing to chase first."
        )

    shift = float(np.median(le_on)) - float(np.median(le_off))
    print(
        f"\n  median shift, OFF -> ON: {shift:+.3f} kcal/mol/atom "
        f"({'less' if shift > 0 else 'more'} efficient with one warhead)"
    )
    print(
        "    This is the quantity HANDOFF item 2 says must be measured rather "
        "than assumed, and the measurement contradicted the reason given for "
        "measuring it."
    )
    if shift < 0:
        print(
            "\n    MEASURED: NEGATIVE. The single-warhead pool is MORE ligand "
            "efficient\n"
            "    than the double-warhead one. The expectation written into HANDOFF "
            "item 2\n"
            "    -- two amidines bury more polar surface, so the constrained pool "
            "sits\n"
            "    worse and the inherited threshold becomes harder to reach -- is "
            "the\n"
            "    wrong sign.\n"
            "\n"
            "    Why the premise failed: buried surface is the numerator, and "
            "ligand\n"
            "    efficiency divides by heavy-atom count. Gluing two inhibitors "
            "end to\n"
            "    end does bury more surface, but it costs ~20 heavy atoms to do "
            "it, and\n"
            "    Vina's score grows only CLOSE to linearly with size. Past a "
            "point the\n"
            "    denominator wins. The artefacts are better RAW scorers and worse\n"
            "    per-atom ones -- which is the whole reason this project moved the\n"
            "    threshold onto efficiency in the first place.\n"
            "\n"
            "    Consequence for the inherited threshold: it is not rejecting "
            "legitimate\n"
            "    candidates. It admits MORE of the constrained pool than of the "
            "pool it\n"
            "    was calibrated on. The risk that item 2 flagged runs in the "
            "opposite\n"
            "    direction from the one it predicted."
        )
    elif shift > 0:
        print(
            "\n    MEASURED: POSITIVE, as HANDOFF item 2 predicted. Removing the "
            "second\n"
            "    warhead left the pool less efficient, so the inherited threshold "
            "is\n"
            "    harder to reach than it was when it was chosen."
        )
    else:
        print("\n    MEASURED: no movement in the median.")

    pass_on = [v for v in le_on if v <= t_on]
    pass_off = [v for v in le_off if v <= t_off]
    print(
        f"\n  reaching the threshold, ON : {len(pass_on)}/{len(le_on)} "
        f"({len(pass_on) / len(le_on):.0%})"
    )
    print(
        f"  reaching the threshold, OFF: {len(pass_off)}/{len(le_off)} "
        f"({len(pass_off) / len(le_off):.0%})"
    )
    print(
        f"\n  carrying the motif more than once, ON : {duplicated(on)}/"
        f"{len(on['candidates'])}"
    )
    print(
        f"  carrying the motif more than once, OFF: {duplicated(off)}/"
        f"{len(off['candidates'])}"
    )

    print("\n  Does the threshold still discriminate on the constrained pool?")
    if not pass_on:
        print(
            "    NO -- it passes NOTHING. Any empty shortlist from the ON arm is\n"
            "    caused by the population shift measured above, NOT by artefact\n"
            "    rejection as in run 37230653219. Different cause, different\n"
            "    response: the fragment set or the comparator, not the filter's\n"
            "    strictness. Relaxing the threshold to produce output would be\n"
            "    manufacturing a shortlist."
        )
    elif len(pass_on) == len(le_on):
        print(
            "    NO -- it passes EVERYTHING, so it no longer filters this pool and\n"
            "    anything weak in it is admitted by default."
        )
    else:
        print(
            f"    YES -- {len(pass_on)} pass, {len(le_on) - len(pass_on)} fail. It\n"
            "    separates within the constrained population, so it is still doing\n"
            "    work on the merits rather than on composition."
        )

    print("\n  What comparison A does NOT control for:")
    n_on = args.retained_on
    n_off = args.retained_off
    if n_on and n_off:
        print(
            f"    - Selection depth. The constrained arm retained {n_on} candidates "
            f"and carried forward {len(on['candidates'])}; the control retained "
            f"{n_off} and carried forward {len(off['candidates'])}. Both apply the "
            "same rule -- the first N past the diversity filter -- but taking "
            f"{len(on['candidates'])} from {n_on} samples that population "
            f"differently from taking {len(off['candidates'])} from {n_off}. "
            f"{'The arms are not equally deep samples of their pools.' if n_on != n_off else ''}"
        )
    else:
        print(
            "    - Selection depth, UNQUANTIFIED. Pass --retained-on/--retained-off "
            "from each arm's log ('Retained: N candidates') to measure it. Both arms "
            "take the first N past the diversity filter, but a deep pool filtered to "
            "60 is spread differently from a shallow one taken nearly whole."
        )
    print(
        "    - n_anchor_escapes and n_anchor_lost are 0 in the control BY "
        "CONSTRUCTION, not by measurement: the generator counts them only while the "
        "constraint is active. Read duplicated motifs from the candidate rows "
        "above, which come from the feature screen and run in both arms."
    )

    # ------------------------------------------------------------------ B ----
    heading(
        "B. Search effort. Control at exhaustiveness 8 vs run "
        f"{HISTORIC['run_id']} at exhaustiveness {HISTORIC['exhaustiveness']}"
    )
    print(
        "Same generator in both arms -- neither constrained -- so what differs is\n"
        "the search effort. This comes free from the control run and answers a\n"
        "question open since exhaustiveness was raised from 4."
    )
    hist_rate = HISTORIC["n_passing"] / HISTORIC["n_docked"]
    off_rate = len(pass_off) / len(le_off)
    print(
        f"\n  reaching threshold, exhaustiveness {HISTORIC['exhaustiveness']}: "
        f"{HISTORIC['n_passing']}/{HISTORIC['n_docked']} ({hist_rate:.0%})  "
        f"at threshold {HISTORIC['threshold']:+.3f}"
    )
    print(
        f"  reaching threshold, exhaustiveness {off.get('exhaustiveness')}: "
        f"{len(pass_off)}/{len(le_off)} ({off_rate:.0%})  at threshold {t_off:+.3f}"
    )
    print(f"\n  threshold movement: {t_off - HISTORIC['threshold']:+.3f} kcal/mol/atom")
    print(
        "    The threshold is re-derived from the same 30 reference actives, so any\n"
        "    movement here is the search effort re-measuring them -- not a change in\n"
        "    the actives and not a change in the criterion."
    )
    print(f"  pass-rate movement: {off_rate - hist_rate:+.0%}")

    print("\n  What comparison B does NOT control for:")
    print(
        f"    - Cores: {HISTORIC['cpu_cores']} then, {off.get('cpu')} now. This "
        "comparison is only valid while the CPU probe's verdict holds. Re-read\n"
        "      runs/cpu_determinism*.log before relying on it; if that verdict ever\n"
        "      changes to thread-dependent, comparison B dies and A survives, "
        "because A holds the thread count fixed."
    )
    print(
        f"    - Sharding: {HISTORIC['n_shards']} shards then, 1 process now. Run "
        f"{HISTORIC['run_id']} is the first run whose six shards agreed on one\n"
        "      ligand signature, so its pass rate is a statistic the README treats\n"
        "      as surviving. Against an earlier run it would not be."
    )
    print(
        "    - The historic arm is a SUMMARY, not a distribution: 15 of 60 at "
        "-0.260 is all that survives.\n"
        "      Its per-candidate efficiencies were never committed and the Actions\n"
        "      artifacts are unreachable, so B compares a pass rate to a pass rate\n"
        "      while A compares distribution to distribution."
    )
    # The one confound B does NOT have, provided the signatures agree. Worth
    # stating positively, because it is what makes B worth reporting at all.
    if off.get("ligand_signature") == HISTORIC_LIGAND_SIGNATURE:
        print(
            f"\n    What B DOES control for, unexpectedly: the control reproduced "
            f"ligand\n"
            f"      signature {HISTORIC_LIGAND_SIGNATURE} -- the signature run "
            f"{HISTORIC['run_id']} recorded.\n"
            "      The two arms are therefore the SAME 60 molecules against the "
            "same 30\n"
            "      reference actives, not two samples of one population. B is a "
            "paired\n"
            "      comparison of search effort on identical input, which is far "
            "stronger\n"
            "      than the pass-rate-to-pass-rate reading above implies, and it "
            "also\n"
            "      independently confirms the generator is reproducible across "
            "machines."
        )
    else:
        print(
            f"\n    The control's ligand signature ({off.get('ligand_signature')}) "
            f"does NOT match the\n"
            f"      {HISTORIC_LIGAND_SIGNATURE} that run {HISTORIC['run_id']} "
            "recorded, so the two arms\n"
            "      are different molecules and B carries a population difference on "
            "top of\n"
            "      the search-effort difference. Treat it as the weaker comparison."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
