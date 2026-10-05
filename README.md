# chemdisco

An auditable computational drug-discovery pipeline: curated public bioactivity
data, QSAR models whose reported performance means something, and fragment-based
candidate generation that is honest about its reach.

**Picking this up fresh? Read [HANDOFF.md](HANDOFF.md) first.** It holds what is
pending, what has not been verified, and the two commands to run before trusting
anything here on a new machine.

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
    pharmacophore.py   conserved-feature profiling, withheld when unusable
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
deliberately leaves the survivors unordered. Whether that filter discriminates at
all at Vina's precision is a separate question, measured once the pipeline ran end
to end — see "The triage threshold has never discriminated" below. It does not.

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

## The full pipeline, run end to end

`scripts/discover.py` assembles everything above on one target and produces a
shortlist — or, as often, a reasoned empty one. On BACE1 against 4FRS, roughly
50 CPU-minutes across six shards:

```
60 candidates docked alongside 30 reference actives, one receptor, one box
   all six shards reporting ligand signature f530b99d98df82f6
15 reach the reference median ligand efficiency (-0.260 kcal/mol/atom)
15 of those carry the anchoring motif TWICE: two inhibitors glued end to end
 0 on the shortlist
```

Run 37230653219, and the first run of this pipeline that is actually one
experiment — see below. The result is an empty shortlist, and that is the finding
rather than a failure to produce one. Every candidate that occupies the site as
efficiently as a median known inhibitor turns out to carry two amidines: in this
fragment space, joining two potent BACE1 inhibitors is how you get a compact,
efficient-looking molecule. The response is more fragments or a different
generator, not a looser filter.

### Constraining the generator to one warhead

Filtering the double-warhead products out does not help, because the generation cap
applies to what the builder emits: the budget goes to artefacts. So the constraint
moved into the fragment pool. Fragments carrying the measured anchoring motif become
BRICS *seeds*; only motif-free fragments are offered as reagents; every product grows
outward from exactly one warhead by construction.

Run 37233542592, same target and seed, all six shards agreeing on ligand signature
`a8ae9075598df11d`:

```
26 fragments carry the amidine -> build seeds
55 motif-free fragments        -> reagents
197 candidates retained (the unconstrained run carried 60 forward, from 96 retained)
 0 products carried the motif more than once
 0 products lost the motif during recombination
60 of 60 candidates carried forward retain a conserved feature; 0 retain none
```

The motif is measured, not declared: the profile is built against the
unrelated-target background *before* generation, so a kinase run would be
constrained on whatever separates kinase actives, and a target where nothing
discriminates is generated unconstrained and says so.

The pooled shortlist for this run existed only as per-shard measurements until it
was reproduced end to end on a single machine; the result, and what it showed about
the triage threshold, is in "The triage threshold has never discriminated" below.

### The shards were never docking the same list

Two runs with the same seed had docked 48 and 47 candidates, passing 12 and 22 of
them. Vina is seeded, so that should not happen. The ligand-signature check added
to find out produced the answer in one pair of runs: **identical curation
signatures** (`da70584f5064c941`, 5091 compounds, 572 actives) and **six different
ligand signatures within a single run.**

`BRICS.BRICSBuild(scrambleReagents=True)` shuffles its reagents using the `random`
module's *global* state, not any generator passed to it. The private
`random.Random(seed)` this package threaded through ordered the fragments and
nothing else, so the builder's enumeration order came from OS entropy — and with
the output capped, every process drew different products from the same fragments.

So every sharded `discover` run before this one pooled six independently generated
candidate lists and reported the total as one experiment. What survives: each
docked number, because the receptor, box and reference distribution were shared
and Vina was seeded. What does not: the pass-rate statistics, and the 12-versus-22
swing, which was never docking noise. Unaffected entirely: the QSAR results and
the docking-enrichment validation (AUC 0.731), neither of which calls the
generator.

The fix seeds the global generator for the enumeration and restores it afterwards.
The test that covers it perturbs the global state between two seeded calls — what a
fresh process does — and was confirmed to fail without the fix before being
trusted with it.

**A shortlist, when there is one, carries no potency prediction.** Generated
candidates sit outside the QSAR model's applicability domain almost by
construction: the reason to generate them is that they are new, which is exactly
where the model has no basis to predict. Attaching an IC50 would be inventing a
number. At best it is a shortlist worth a chemist's hour, not a result.

### The triage threshold has never discriminated

Run 37233542592 was finished end to end on a 6-core machine — 101 minutes, one
process, no sharding — reproducing the per-shard figures above and the ligand
signature `a8ae9075598df11d` exactly. For the first time the pipeline produced the
pooled shortlist the Actions spending limit had blocked:

```
60 candidates docked alongside 30 reference actives
22 reach the reference median ligand efficiency (-0.261 kcal/mol/atom)
60 of 60 retain the amidine; 0 artefacts
shortlist: 22
```

The threshold barely moved: the measured reference median is −0.261 against the
historic −0.260, despite exhaustiveness rising 4→8 and the machine changing. That
near-identity is a coincidence, and it hides the finding rather than being one.

**The one-warhead premise came out with the wrong sign.** The expectation on record
was that two buried amidines make the double-warhead pool *more* efficient, so the
constrained pool would sit worse and struggle to reach the bar. Measured, the
single-warhead pool is *more* efficient — median −0.250 against the double-warhead
control's −0.203, and higher at every matched heavy-atom band (30–34 atoms: −0.276
vs −0.213; 45–60: −0.219 vs −0.156). The threshold admits more of the new pool
(37%) than of the pool it was calibrated on (27%); it is not rejecting legitimate
candidates.

**But the population was never the point.** None of the 22 survivors clears the
threshold by more than Vina's own 2.5 kcal/mol error — the margins run a median of
0.49 and a maximum of 1.17 kcal/mol. Sixteen of the 60 sit within 0.010
kcal/mol/atom of the cut, and the whole shortlist spans 0.029. By this package's own
`scores_are_distinguishable`, no survivor is distinguishable from the threshold. The
unconstrained control gives the same answer — 0 of its 16 passers clear the cut by
the margin either — which is why the historic 15/60-all-artefacts result was never a
population-shift problem.

So the 22-vs-38 split is a cut through the densest part of a distribution whose width
is below the method's resolution. The threshold has never discriminated at Vina's
precision: not in the double-warhead pool it was calibrated on, not in the
single-warhead pool it was re-measured against. This is the conclusion the redocking
and enrichment sections reach by other routes — on this target the score separates
groups on average and cannot order individuals — arrived at a third time, from the
triage.

**Therefore the triage is withheld, not tightened.** A check that cannot discriminate
is reported as one and the candidates pass through unranked, exactly as the
conserved-feature check is withheld when no feature clears its floor. All 60
candidates are reported with their ligand efficiencies and no pass/fail; the
survivors are described as indistinguishable from a median known active at Vina's
precision — and so are most of the 38 the old cut rejected. The threshold stays as a
reported reference point, never as a filter that ranks. Implementing that in
`triage_candidates` is the one pending item; the measurement and the decision are
recorded here.

### Six method errors these runs produced

Each returned a believable wrong answer, and each was found by reading the output
rather than by a test failing.

**The threshold selected for molecular weight.** An earlier run returned zero
survivors because all 13 candidates clearing the raw-score threshold carried the
anchoring motif twice. Two things conspired: Vina's score grows close to linearly
with size, and the way BRICS makes a large molecule from inhibitor fragments is by
joining warheads. So "beats the median known active" selected for size, and size
in this fragment space means two drugs glued end to end. The correction — compare
ligand efficiency, not raw score — was already written in this package's own
engine module; the triage sorted by efficiency and thresholded on raw score.

**A shortlist of one molecule in four variations.** Five of fifteen entries shared
one aminothiazine core with different N-substituents. The diversity threshold
dropped from 0.85 to 0.7.

**A candidate with nothing to bind with.** The top entry by synthetic accessibility
was `N#Cc1cc(-c2cc(F)c(F)c(-c3ccc(F)cn3)c2)cc(Cl)c1F` — a polyfluorinated biaryl
nitrile with no basic nitrogen, scoring -9.19 against an aspartyl protease whose
inhibitors essentially all carry an amidine, guanidine or basic amine to engage
the Asp32/Asp228 dyad. It fills the pocket and cannot do the chemistry, and a
docking score cannot notice: scoring functions reward shape, not chemistry.

**A feature check that could not tell what it was measuring.** The fix for the
above measures which functional groups the known actives share and flags
candidates carrying none. Its first version compared prevalence only, so
`aromatic_ring` — in essentially every drug-like molecule — counted as conserved
and the warheadless biaryl passed. Adding a background and requiring enrichment
fixed that, and then exposed a deeper problem:

```
amidine                  90% of actives vs 48% of background  (1.9x)
halogen_on_aromatic      93% vs 51%                           (1.8x)
primary_aliphatic_amine  90% vs 47%                           (1.9x)
```

Three features indistinguishable. The background was weakly active BACE1
compounds — and a weak BACE1 binder is still a BACE1-series compound carrying the
same warhead. Filtering it to compounds below 0.4 Tanimoto from every active moved
amidine prevalence from 50% to 48%: essentially nothing, because two molecules can
share a small amidine and still sit below that threshold.

**A target's own data contains no "does not bind" population.** Every compound in
it was designed and tested against that target. So the check is *withheld* when no
feature reaches 3x enrichment, and the report says the survivors have not been
checked for binding chemistry — a stated gap rather than a silent one.

What it needed was a background from targets with nothing to do with this one.
`chemdisco/data/background.py` now fetches one: 1200 ligands of ten unrelated
targets — three kinases, two aminergic GPCRs, a peptide GPCR, two nuclear
receptors, carbonic anhydrase and COX-2 — balanced round-robin so that truncating
the set keeps the families even, with anything appearing in the target's own
dataset excluded by identifier. Measured on BACE1 (`probe-background`, run
37223022707):

```
amidine              90% of actives vs  4% of background  (22.5x)  <- conserved
primary_amine        90% vs 17%                            (5.4x)  <- conserved
halogen_on_aromatic  93% vs 33%                            (2.8x)  below the floor
basic_nitrogen_any   97% vs 53%                            (1.8x)
aromatic_ring       100% vs 98%                            (1.0x)  <- ubiquitous
```

The gap is the result, not the top number. The aromatic halogen — the feature that
let the warheadless biaryl through — now falls below the 3x floor, and `amidine` is
the only motif the check treats as anchoring. A background that enriched everything
would be as useless as none, so `probe_background.py` checks both sides and reports
either outcome; it exits 0 on a negative result, because a background that does not
work is a finding rather than a broken job.

These compounds are **presumed** non-binders, not measured ones: nobody tested them
against BACE1. Any that do bind dilute the enrichment rather than invent it, so the
error runs in the safe direction, and the output says "presumed" every time. A
background below 200 compounds or 3 protein families is refused rather than used —
an unusable background is worse than none, because it produces ratios that look
like measurements.

This is the mirror of a trade-off recorded above for enrichment. Measured weak
binders are the **right** control for docking enrichment, where shared chemistry
makes the test harder and therefore honest, and the **wrong** one for identifying
binding determinants, where shared chemistry erases the signal. Two correct
answers from one dataset, depending on the question.

**Withholding treated as failing.** When the check was withheld, every candidate
came back `retains_strong = False` and the pipeline dropped all 17 survivors for
failing a check already declared unusable. No verdict is not a failing verdict.

**A plausible cause accepted instead of measured — nearly.** Faced with the
12-versus-22 swing, the first explanation reached for was an unstable sort over
tied activity values, and work started on that before checking: curation already
sorts its output by compound identifier, so the ranking was never order-dependent.
The instrument built because the cause was *not* known — a signature per shard — found
the real one two runs later. A plausible story would have left it in place.

## What this pipeline cannot do

Stated here so it does not have to be inferred:

- **BRICS recombination cannot invent a fragment.** Output lives inside the
  chemical space spanned by the input actives. It will not discover a novel
  chemotype, and for a target with few known actives the reachable space is
  correspondingly small.
- **The conserved-feature background is presumed, not measured.** The compounds in
  it were tested against other targets, never against this one. The presumption
  dilutes enrichment rather than inflating it, but it is a presumption. And the
  check still withholds itself on any target where no feature clears 3x — which is
  the honest outcome, not a solved problem.
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
