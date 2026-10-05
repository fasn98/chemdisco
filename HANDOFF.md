# Handoff

Written for a session starting fresh on a machine with open network and real cores
— which is most of what the previous environment lacked. Read `README.md` for the
method and the measurements; this file is only what is *pending*, what is
*unverified*, and what to run first.

Last session: 2026-10-04. Repository state at handoff: 64 commits.

## Run these first, in this order

```bash
./scripts/run_local.sh setup     # apt + venv + rdkit, vina, meeko, openbabel
./scripts/run_local.sh tests     # see "unverified" below — this is the point
./scripts/run_local.sh cpu-probe # settles whether scores are machine-comparable
```

### Why `tests` first

Four commits went in after GitHub Actions stopped running (spending limit), so
**they have never been executed anywhere**:

| What | Verified how far |
|---|---|
| `ruff` over everything | passed locally |
| 178 toolkit-free tests (+809 subtests) | passed locally |
| `TestRunSettingsAreRecorded` in `test_dock_engine.py` | **never run** — needs numpy |
| `scripts/probe_cpu_determinism.py` | **never run** — checked against the real function signatures only |
| `scripts/run_local.sh` | **never run** — `bash -n` only |
| `scripts/setup_self_hosted.sh` | **never run** — `bash -n` only |

If something in that list is broken, it was broken on arrival. Fix it rather than
working around it.

### Why `cpu-probe` second

Every number in the README was produced on a 2-core GitHub runner with Vina's
`cpu=0`, meaning "use what you find". A machine with more cores runs more threads.
Whether Vina 1.2.7's output depends on the thread count with a fixed seed is not
settled — its documentation implies the seed suffices, user reports disagree — so
the probe measures it instead of assuming.

It prints one of three verdicts. **Thread-dependent** means `--cpu` has to be pinned
for any set of scores meant to be compared, and the README's numbers carry a caveat.
**Not reproducible even at a fixed thread count** would be much worse than that: it
would mean the seed is not controlling the search at all and nothing in this project
should be pooled until that is understood.

Do not skip this because the pipeline appears to work. It appearing to work while
its numbers quietly depended on the machine is precisely the failure that cost the
previous session most of a day (see "Generation was never reproducible" in the
README).

## Pending work

### 1. The shortlist from run 37233542592 — DONE

Produced end to end on a 6-core machine (`./scripts/run_local.sh discover`, 101
minutes, one process), reproducing ligand signature `a8ae9075598df11d` and the
per-shard figures exactly: 26 amidine seeds, 55 motif-free reagents, 197 retained,
0 double-motif. Pooled outcome: 60 docked, 22 reach the reference median ligand
efficiency, 60/60 retain the amidine, 0 artefacts. The committed run is in `runs/`.
What that shortlist turned out to mean is item 2.

### 2. Make `triage_candidates` withhold, rather than filter, on this target

The population-shift worry that stood here has been measured, and it was the wrong
worry. Full result in the README under "The triage threshold has never
discriminated"; the short version:

- The premise came out with the **wrong sign.** The single-warhead pool is *more*
  ligand-efficient than the double-warhead control (median −0.250 vs −0.203), not
  less, at every matched heavy-atom band. The threshold is not rejecting legitimate
  candidates — it admits 37% of the new pool against 27% of the old.
- The threshold **never discriminated at Vina's precision**, in either population.
  None of the 22 survivors clears it by more than the 2.5 kcal/mol method error
  (margins median 0.49, max 1.17); the whole shortlist spans 0.029 kcal/mol/atom.
  The control gives the same answer: 0 of 16 passers clear the margin.

The decision taken: **withhold the triage as non-discriminating** and report all 60
candidates with their efficiencies and no pass/fail, exactly as the conserved-feature
check is withheld when nothing clears its floor. The README now states this as the
standard.

What is **pending** is the code: `triage_candidates` still filters and labels
survivors. It needs to stop ranking on this target — report the efficiencies, keep
the reference median as a stated reference point only, and describe survivors as
indistinguishable from a median known active at Vina's precision. The README
describes the intended behaviour; the implementation does not match it yet. Close
that gap before the next `discover` run is treated as producing a ranked shortlist.

### 3. Exhaustiveness has never been raised

Fixed at 4 throughout, because two cores could not pay for more. Vina's default is 8
and published work commonly uses 16–32. On a machine with cores, raise it.

There is a specific question waiting: redocking 4FRS put the correct pose third, at
1.82 Å, with the top-ranked pose at 4.19 Å — and going from exhaustiveness 16 to 64
did not change that, which points at the scoring function rather than the sampling.
That was measured in isolated redocking. Whether it holds across a full pipeline run
at higher exhaustiveness has never been tested.

### 4. Cross-docking is not implemented

Redocking puts a ligand back into the receptor conformation it induced — the easiest
version of the problem. Whether the setup places a *different* ligand correctly is
what cross-docking measures, and this pipeline has never asked.

### 5. Housekeeping

- Branch `verify-determinism-test` should be deleted. It was created deliberately,
  with a fix removed, to confirm a test failed without it; the proxy in the previous
  environment could not delete remote refs.
- `scripts/setup_self_hosted.sh` exists if GitHub Actions is ever wanted on this
  machine (free, even for private repos). It is optional and unrelated to running
  the pipeline directly. Note the security constraint written in it: a self-hosted
  runner and a public repository are mutually exclusive.

## Things that are settled, so they do not get re-litigated

- **The feature-profile background comes from unrelated targets.** A target's own
  data contains no "does not bind" population; measured on BACE1, amidine reaches
  22.5× against the generic background while the aromatic halogen falls to 2.8×,
  below the 3× floor. The target's own weak binders gave both ~1.8× and let a
  warheadless biaryl through.
- **Generation must not be left to the ambient random state.** `BRICS.BRICSBuild`
  with `scrambleReagents=True` shuffles using the global `random` module.
- **Curation output order does not depend on retrieval order.** Aggregation sorts by
  `(target_id, compound_id)`. An earlier session guessed otherwise and started
  fixing the wrong thing.
- **Withholding a check is not failing it.** A candidate that could not be assessed
  is not a candidate that failed assessment.

## The standard this project holds itself to

Nothing reports a number it cannot justify. Quantities carry provenance and degrade
rather than upgrade; a check that cannot discriminate is withheld and says so; an
empty shortlist is a result and is reported as one. Where a limitation is known it
is written down rather than left to be inferred — which is why this file exists.
