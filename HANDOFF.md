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

Four commits went in after GitHub Actions stopped running (spending limit) and had
never been executed anywhere. They have now, on a 6-core Contabo box, and none of
them was broken:

| What | Status |
|---|---|
| `ruff` over everything | passes |
| full suite with RDKit present | passes — see the count below |
| `TestRunSettingsAreRecorded` in `test_dock_engine.py` | **ran, passes** |
| `scripts/probe_cpu_determinism.py` | **ran** — verdict REPRODUCIBLE, see below |
| `scripts/run_local.sh` | **ran** — every subcommand except `setup_self_hosted` |
| `scripts/setup_self_hosted.sh` | still **never run** — `bash -n` only, and not needed to run the pipeline |

Two defects the scripts did have, both found by running them rather than by a test:
output was block-buffered through `tee`, so a 101-minute run showed nothing for
~30 minutes at a stretch on a machine where the log is the only record; and the
`enrichment` branch never passed `--time-budget`, so it silently truncated a 54-ligand
screen to 16 and reported an `INCONCLUSIVE` 8-vs-8 AUC that looked like a measurement.
Both fixed.

Run `tests` first anyway. It is cheap, and it is the only thing that tells you the
tree you just pulled is the tree that was verified.

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

## How to work in this repository

### Detach anything long, and commit each result as it lands

**Anything expected to take more than ~10 minutes runs under `setsid`/`nohup` or
`tmux`. Commit each result the moment its file closes — not at the end of the
block of work.**

Not a style preference. This repository has lost work to session teardowns three
times, and the pattern was identical every time:

- `discover` was killed at 30 of 90 ligands having written no shard. Forty minutes
  of docking, with nothing to show it had run.
- Source edits to `scripts/discover.py` were lost mid-change, because they were
  being held for one tidy commit at the end.
- A watcher waiting to report a run's completion died with its session while the
  run itself, launched under `setsid`, carried on for another hour.

What survived every time was exactly what had been detached and what had been
committed. The 101-minute `discover` run that produced the current `runs/` went
straight through a session ending and finished normally.

Concretely:

```bash
setsid nohup bash -c 'EXHAUSTIVENESS=8 CPU=6 ./scripts/run_local.sh discover' \
  >> runs/discover.console.log 2>&1 < /dev/null &
```

Which things here cross the threshold: `discover` (~100 min), `control` (~100 min),
`enrichment` (~65 min at the raised time budget), and the full `tests` run (~6 min
for both passes, so borderline). `cpu-probe` and `redock` do not.

Two reasons the commit has to be *immediate* rather than at the end:

1. **A local, unpushed commit is already enough.** The work survives in `.git`
   even if nothing reaches the remote, so there is never a reason to hold an edit.
2. **"When its file closes" is not "when the work block finishes."** Those are
   different moments, and conflating them is how a truncated measurement gets into
   the record looking finished. That has already happened here: an `enrichment`
   screen stopped early on its default time budget at 16 of 54 ligands and
   reported an `INCONCLUSIVE` 8-vs-8 AUC of 0.641 — a number that looked like a
   measurement of enrichment and was a measurement of the time budget. It is kept
   as `runs/enrichment_truncated_1500s.log`, under a name that says what it is.

So: if a run is still writing, leave it out of the commit and say so explicitly.
A partial artefact is not a result, and `runs/` is the only record this machine
has — a GitHub Actions run had an immutable id attached to the commit, and here
the log is it.

## Pending work

### 1. The shortlist from run 37233542592 — DONE

Produced end to end on a 6-core machine (`./scripts/run_local.sh discover`, 101
minutes, one process), reproducing ligand signature `a8ae9075598df11d` and the
per-shard figures exactly: 26 amidine seeds, 55 motif-free reagents, 197 retained,
0 double-motif. Pooled outcome: 60 docked, 22 reach the reference median ligand
efficiency, 60/60 retain the amidine, 0 artefacts. The committed run is in `runs/`.
What that shortlist turned out to mean is item 2.

### 2. Triage withholds instead of filtering when it cannot discriminate — DONE

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

**Implemented.** Both triage paths withhold when nothing beats the threshold by more
than `VINA_ERROR_KCAL`, reporting every assessed candidate with its efficiency,
uncut and unsorted, with `triage_withheld` serialised explicitly and the list keyed
`candidates_unranked` instead of `shortlist`.

Two paths, which is the part that is easy to miss: `triage_candidates` in
`chemdisco/dock/screen.py` works on raw score at a percentile, and the triage that
actually produces the shortlist is inline in `combine()` in `scripts/discover.py`
and works on ligand efficiency. `discover.py` never called the library function. An
earlier attempt changed only the library function, left the pipeline output
unchanged, and so left the README describing behaviour the code did not execute —
`tests/test_discover_triage.py` exists to stop that recurring, driving `combine()`
through a shard fixture and asserting on the JSON it writes.

### 3. Exhaustiveness has never been raised

Fixed at 4 throughout, because two cores could not pay for more. Vina's default is 8
and published work commonly uses 16–32. On a machine with cores, raise it.

There is a specific question waiting: redocking 4FRS put the correct pose third, at
1.82 Å, with the top-ranked pose at 4.19 Å — and going from exhaustiveness 16 to 64
did not change that, which points at the scoring function rather than the sampling.
That was measured in isolated redocking. Whether it holds across a full pipeline run
at higher exhaustiveness has never been tested.

### 4. Cross-docking — DONE, and it came back negative

Measured: 5 structures, all 25 ordered pairs. Diagonal (redocking) 5/5 within
2 A with 4FRS at 1.78 A against the 1.82 A on record, so the arrangement is
sound. Off-diagonal: a pose within 2 A exists in 9/20, and the TOP-RANKED pose is
within 2 A in only 3/20.

Consequence, written up in the README: the shortlist's poses are unreliable, and
so are the scores read off them and the ligand efficiencies computed from those
scores. The enrichment AUC is untouched -- it measures separation between two
score populations, not individual pose correctness.

What is still open here:

- **Cross-docking with the generated candidates themselves.** This measured
  crystallographic ligands against crystallographic poses, because that is the
  only case where a right answer exists. The candidates have no crystal
  structures, so their pose error cannot be measured directly -- only bounded by
  this result. Any improvement would need a different kind of evidence
  (consensus across receptor conformations, say) rather than more docking.
- **Ensemble docking.** If one conformation places a different ligand correctly
  one time in four, docking into several conformations and keeping agreement is
  the standard response. Untried here.
- **Rescoring.** Three independent routes now say the limit is the scoring
  function. The obvious next move is a different scoring function on the same
  poses, not more sampling.

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
