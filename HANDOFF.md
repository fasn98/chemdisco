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

### 1. The shortlist from run 37233542592 was never produced

Six shards docked successfully with the newly constrained generator — all reporting
ligand signature `a8ae9075598df11d` — and the combine job could not start on the
spending limit. Its artifacts live for 90 days.

Simplest path now: re-run the whole thing locally, which is free and fast:

```bash
./scripts/run_local.sh discover
```

This is the first pipeline output where the candidates are constrained to one
anchoring motif. What the per-shard logs already showed: 26 fragments carry the
amidine (used as build seeds), 55 are motif-free (used as reagents), 197 candidates
retained from a budget that previously yielded 60, zero carrying the motif twice.
What is unknown: how many reach the reference median ligand efficiency, and whether
anything survives.

### 2. The triage threshold is calibrated on a candidate population that no longer exists

**This is the most important open item, and it is easy to miss because the number
still looks valid.**

Triage keeps candidates reaching the reference actives' median ligand efficiency,
−0.260 kcal/mol/atom. That threshold was chosen while watching a candidate pool of
double-warhead molecules. Two amidines bury more polar surface than one, so the
constrained generator's candidates sit at a different place in the efficiency
distribution by construction.

A threshold inherited from a different population may now reject legitimate
candidates or wave weak ones through. Measure it on the new pool; do not assume it
transfers. The same class of error appears five times in the README — a number that
stays in place after what it measures has changed underneath it.

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
