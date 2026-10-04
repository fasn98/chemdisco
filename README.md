# chemdisco

An auditable computational drug-discovery pipeline: curated public bioactivity
data, QSAR models whose reported performance means something, and fragment-based
candidate generation that is honest about its reach.

## Why this exists

This is a rewrite of an earlier Streamlit application. That application had a
sound workflow and real integrations with PubChem, ChEMBL, UniProt and PubMed. It
also had a defect that made its scientific output unusable, and the defect is
worth stating plainly because avoiding it is this project's entire design
premise.

It reported invented numbers through the same interface it used for measured ones.

Concretely, from the previous codebase:

```python
# "docking" — a sum of property bonuses plus noise
final_score += random.uniform(-0.5, 0.5)

# "interaction energy", in kcal/mol
energy = -5.0 - (properties['molecular_weight'] / 100)
energy += random.uniform(-2.0, 2.0)

# activity labels, when no measurement existed
activity = (mw_score + logp_score) / 2 + np.random.normal(0, 0.1)

# unit conversion
return -np.log10(value * 1e-9)  # Assuming nM units
```

The last line is the most instructive. ChEMBL reports `standard_units` as nM, µM,
mM, M, µg·mL⁻¹, % and dozens of other things. Assuming nanomolar turns a
micromolar IC50 into a value three log units too potent, and turns a
percent-inhibition readout into numerical noise — and because the result still
looks like a plausible pIC50, nothing downstream can detect it.

The third line was worse. Synthetic activity labels were computed from molecular
weight and LogP, which were then supplied to the model as features, so the model
was fitting an arithmetic identity. On reconstruction that yields a held-out R² of
about **0.69** — an entirely ordinary QSAR result that invites no scrutiny
whatsoever. An absurd 0.99 would have been safer.

None of this was written in bad faith. It is drift: a plausible heuristic, then a
line of noise "to simulate experimental variation", then a synthetic label so a
model could train on thin data, then reference compounds injected to make a small
dataset trainable. Each step is locally reasonable. The destination is a platform
reporting fabricated binding affinities to two decimal places.

## How this codebase prevents that

**Every number carries its provenance.** The central type is `Quantity`, which
cannot hold a value without declaring where it came from — `MEASURED`, `DERIVED`,
`PREDICTED` or `HEURISTIC`. A measurement requires a citation. Provenance degrades
through a pipeline and never improves, so a prediction cannot be laundered into a
measurement by passing through a conversion.

**An unknown value is `None`, never a default.** No imputed molecular weight of
300, no fallback LogP of 2.0. A failed calculation returns an explicit absence
that renders as "not computed".

**Fabricated labels are unrepresentable.** A training target must consist of
quantities descending from measurements; `verify_label_provenance` refuses
anything else. This is the structural guarantee, and it exists because the
statistical alternative does not work — the correlation screen in this package
genuinely fails to detect the previous project's synthetic labels, and there is a
test asserting that it fails, so nobody mistakes a clean screen for a clean
dataset.

**No randomness in scoring paths.** `tests/test_no_fabrication.py` walks the
package's syntax tree and fails the build if any module that reports a scientific
value can reach a random number generator. Legitimate uses — shuffling a split,
bootstrap resampling, fragment recombination — are on an allowlist with written
justifications. Adding to that allowlist is a reviewable act; adding
`import random` to a scoring module is a build failure.

**Splits measure generalisation, not memorisation.** Scaffold splitting is the
default, and a random split is available only as the paired pessimistic baseline;
its own notes carry the warning. The gap between the two quantifies how much of a
model's apparent accuracy is analogue leakage.

**Models must beat the trivial alternatives.** Every evaluation reports the mean
predictor, a nearest-neighbour similarity lookup, and a label-permutation
distribution. A model that cannot beat a similarity lookup has added nothing over
looking up the most similar known compound. `Evaluation.is_defensible` encodes
that bar.

**Every metric has an error bar.** Bootstrap confidence intervals on R², RMSE, MAE
and Spearman. On a 200-compound test set the 95% interval on R² routinely spans
0.2, which means two models differing by 0.05 are indistinguishable — and the
report says so.

**Predictions outside the applicability domain cannot rank.** Three domain
definitions (k-NN distance, fingerprint similarity, Williams leverage) are
assessed and must agree. An extrapolated prediction is still reported, visibly
flagged, and excluded from any ranked candidate list.

## Architecture

The scientific logic is deliberately free of RDKit and of the network, so it is
testable in a bare environment and auditable without wading through API calls.

```
chemdisco/
  provenance.py        Quantity, Origin — the central primitive          [pure]
  units.py             concentration units → pActivity, with refusals    [pure]
  curate/
    records.py         ActivityRecord, Rejection, CurationReport         [pure]
    filters.py         confidence, assay type, endpoint, validity flags  [pure]
    aggregate.py       replicate combination; discards contradictions    [pure]
  split/
    scaffold.py        scaffold-disjoint partitioning                    [pure]
  qsar/
    dataset.py         training-set assembly; provenance enforcement     [pure]
    baselines.py       mean, 1-NN, permutation; leakage screens       [sklearn]
    applicability.py   three applicability-domain definitions          [numpy]
    evaluate.py        metrics with bootstrap intervals                [numpy]
    model.py           RF / GBM / Ridge returning Quantity objects    [sklearn]
  dock/
    pdb.py             PDB parsing, ligand vs additive vs peptide chain    [pure]
    box.py             search-box geometry and RMSD                       [pure]
    decoys.py          property-matched decoy selection and balancing     [pure]
    enrichment.py      AUC, EF, BEDROC; three-state verdict              [numpy]
    screen.py          batch docking, attrition accounting, sharding      [pure]
    engine.py          AutoDock Vina, with its error attached      [vina edge]
  chem/                                                            [RDKit edge]
    standardize.py     salt stripping, neutralisation, InChIKey identity
    descriptors.py     ECFP4 fingerprints and interpretable descriptors
    scaffold.py        Bemis-Murcko perception → feeds split/
    alerts.py          PAINS, Brenk, NIH, real SAscore, Lipinski
    similarity.py      Tanimoto, novelty assessment, diverse subsets
  generate/                                                        [RDKit edge]
    brics.py           fragment recombination with a filter cascade
  data/                                                          [network edge]
    chembl.py          cached, rate-limited client with the assay join
```

`split/scaffold.py` takes scaffolds as **strings**, not molecules. That is why
leakage bugs in the splitting logic are catchable without a chemistry install —
and three real bugs in it were caught that way during development.

## Installation

```bash
# Core: curation, units, splitting, QSAR evaluation. No chemistry toolkit needed.
pip install -e .

# With structure handling, generation and the full test suite.
pip install -e ".[chem,dev]"
```

## Validate before trusting

```bash
python scripts/validate_target.py                      # BACE1, UniProt P56817
python scripts/validate_target.py --accession P00533 --name EGFR
```

This walks the whole chain and prints what every stage discarded. Run it on a
well-populated target before pointing the pipeline anywhere harder.

## Running the tests

```bash
pytest                      # full suite
pytest -m "not slow"        # quick pass
```

The suite is written with `unittest.TestCase`, so it also runs under the standard
library alone:

```bash
python -m unittest discover -s tests -t .
```

CI runs it twice: once **without** RDKit, to prove the core logic really is
toolkit-free, and once with it. The RDKit job fails if any test is skipped,
because a skipped chemistry test in the job that installs RDKit means the guard
misfired and the suite is passing while testing nothing.

## About Creutzfeldt-Jakob disease

Prion disease is the motivating target for this work. It is also a genuinely hard
case, and the difficulties are worth knowing before aiming a pipeline at it:

- The pathogenic species, PrP<sup>Sc</sup>, is a misfolded conformer forming
  fibrillar aggregates. It offers no well-defined small-molecule pocket, and there
  is no high-resolution structure of the human form suitable for docking.
  PrP<sup>C</sup>, the normal conformer, does have solved structures and is a
  legitimate alternative target — stabilising it, or reducing its expression.
- Measured anti-prion data is largely cell-based (ScN2a and similar), reporting
  EC50 against a phenotype rather than binding affinity. Potency there is
  confounded by permeability, efflux and metabolism, and this package's curation
  policy treats those measurements accordingly rather than pooling them with
  binding constants.
- The small-molecule clinical record is a sequence of failures: quinacrine,
  pentosan polysulfate, doxycycline, flupirtine. Astemizole and anle138b showed
  effects in mouse models. The approach with the most current traction is not a
  small molecule but antisense reduction of *PRNP* expression.

So prion disease is an excellent motivation and a poor first validation target: a
pipeline validated only on scarce, cell-based data has no way to distinguish a
working model from a broken one. Validate on BACE1, then move.

## Docking, and what it was measured to be worth

The predecessor's "docking" was `-5.0 - molecular_weight/100 + random.uniform(-2, 2)`,
reported in kcal/mol. It was removed rather than ported. The replacement runs
AutoDock Vina against real PDB structures, and its reporting is shaped by what
redocking on this target actually showed.

**Redocking 4FRS** (BACE1 at 1.70 Å, 25-heavy-atom aminohydantoin inhibitor):

| | exhaustiveness 16 | exhaustiveness 64 |
|---|---|---|
| Top-ranked pose | 4.19 Å | 4.22 Å |
| Best pose found | 1.78 Å (rank 5) | 1.79 Å (rank 3) |
| Poses under 2 Å | 2 of 9 | 4 of 9 |
| Score spread | 1.23 kcal/mol | 0.86 kcal/mol |

Quadrupling the search effort found more correct poses and still ranked a 4.2 Å
pose first. That separates the two possible explanations: the **search** finds
the right answer, and the **scoring function** cannot pick it out. More compute
does not fix it.

The spread across all nine poses is well inside Vina's own ~2.5 kcal/mol error,
so the ranking between them is not determined by the score at all.
`DockingResult.pose_ranking_is_determined` returns False for exactly this case
and the report says so in words. The practical consequence is direct: on this
target, docking scores must not order candidates.

So docking answers "could this molecule occupy this pocket, in what orientation"
much better than "how tightly does it bind". Scores are `PREDICTED` quantities
carrying 2.5 kcal/mol as their uncertainty and `in_domain=None`, which keeps them
out of any ranked list by themselves.

**A bug worth recording**, because it is the kind that produces a believable
wrong answer. The first redocking attempt used 1FKN and returned a tidy PARTIAL
verdict that meant nothing. 1FKN's inhibitor is OM99-2, an octapeptide deposited
as *polymer chains* C and D — ATOM records, indistinguishable from receptor by a
HETATM filter — with only its non-standard hydroxyethylene isostere written as
the HETATM residue `1OL`. So the run docked into a pocket still occupied by its
own ligand, and measured RMSD against a 13-atom fragment of a 60-atom molecule.
Nearest-neighbour RMSD against a small reference set is lenient, so the error
flattered the result instead of exposing itself. The pipeline now detects
peptide-ligand chains, strips them from the receptor, and refuses to redock
against a fragment.

## Virtual screening: measured, then used accordingly

Before docking could triage the generated candidates the QSAR model could not
score, a prior question needed answering: **does docking separate actives from
inactives on this target at all?** If not, triage adds nothing and a ranked list
is noise.

**The experiment.** 40 potent BACE1 actives (pIC50 ≥ 8) against 50 decoys drawn
from the weakly-active end of the same curated ChEMBL set, matched on molecular
weight, logP, hydrogen bonding, flexibility and charge, and filtered to below
0.25 Tanimoto from every active. Docked into 4FRS, one receptor, one box,
38 CPU-minutes across six shards.

| | |
|---|---|
| AUC-ROC | **0.731** [0.617, 0.831] |
| BEDROC (α=20) | 0.359 |
| EF 1% | 0.00 (top 1% is one compound) |
| Mean score | actives −9.09, decoys −8.26 kcal/mol |

The lower bound sits well above random, so docking does carry information about
this site. But BEDROC of 0.36 and an empty top 1% say the *early* ranking is
poor: it separates the groups on average and does not concentrate actives at the
top, which is where a screen actually buys compounds. That is consistent with
the redocking result — the score discriminates weakly at fine resolution.

So `triage_candidates` filters against the actives' score distribution and
deliberately leaves the survivors unordered.

**Using weak binders as decoys** is a harder control than DUD-E's property-matched
library compounds: they are measured against this target, so there is no
contamination from untested binders, but they often share the actives' chemotype.
Enrichment here reads lower than a published DUD-E figure for the same target,
which is the honest direction for the bias to run.

**Three method errors this experiment produced,** recorded because each returned
a believable wrong answer:

1. The first run docked 8 actives against 8 decoys and reported AUC 0.641
   [0.317, 0.900] as *"docking does not separate the groups"*. An interval
   reaching from well below random to strong means the sample answered nothing.
   `separates == False` conflated "no signal" with "too small to see one", which
   is absence of evidence reported as evidence of absence. There is now a third
   verdict, `inconclusive`, and `compounds_needed` estimates the sample a
   conclusion would take — 68 per group, against the 8 used.
2. Per-pair property matching passed while the **group** means differed by
   34.7 Da, because the matcher could not fill every quota and the decoys it
   found skewed small. A gap that size is enough for Vina's size bias to
   manufacture enrichment. `balance_selection` now trims to the group-level gap.
3. Striding the interleaved ligand list across six shards starved half of them
   of actives — round-robin interleaving is periodic, so a six-way stride lands
   on a fixed phase. The pooled result survived only because every shard
   finished.

## What this pipeline cannot do

Stated here so it does not have to be inferred:

- **BRICS recombination cannot invent a fragment.** Output lives inside the
  chemical space spanned by the input actives. It will not discover a novel
  chemotype, and for a target with few known actives the reachable space is
  correspondingly small.
- **Cross-docking is not implemented.** Redocking puts a ligand back into the
  receptor conformation it induced, which is the easiest version of the problem.
  Whether a setup places a *different* ligand correctly is a harder question that
  cross-docking measures and this pipeline does not yet ask.
- **ADMET and toxicity prediction are absent.** The previous project's versions
  were invented formulas with no experimental basis. Honest versions require
  curated experimental datasets per endpoint, which is a substantial separate
  effort.
- **Novelty and model reliability are in genuine tension.** The more novel a
  generated candidate, the further it sits from the model's training data and the
  less its predicted activity means. This is intrinsic, not an engineering gap.
  The pipeline reports the domain verdict for every candidate and refuses to rank
  on extrapolations; it cannot resolve the conflict, and neither can anything
  else.

## Licence

MIT.
