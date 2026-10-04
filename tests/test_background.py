"""The background assembler: balance, exclusion, and when it refuses to be used.

No network and no RDKit here. What is tested is the logic that decides which
compounds end up in the comparison set, because that is where a silent error
would do the most damage: a background made of one chemotype still produces
confident-looking enrichment numbers.
"""

from __future__ import annotations

from chemdisco.data.background import (
    UNRELATED_TARGETS,
    BackgroundSet,
    assemble_background,
    summarise,
)


def contribution(prefix: str, n: int) -> list[tuple[str, str]]:
    return [(f"{prefix}{i}", f"C{'C' * (i % 5)}O{prefix}") for i in range(n)]


class TestBalance:
    def test_truncation_does_not_favour_the_first_target(self) -> None:
        """The failure this function exists to prevent.

        Concatenating per-target lists and cutting at the ceiling gives a
        background drawn entirely from whichever target came first. Enrichment
        against it would mean "rarer than in kinase inhibitors", not "rarer than
        in drug-like compounds".
        """
        result = assemble_background(
            {"EGFR": contribution("a", 500), "ESR1": contribution("b", 500)},
            max_total=100,
            families={"EGFR": "protein kinase", "ESR1": "nuclear receptor"},
        )
        assert len(result.smiles) == 100
        assert result.per_target == {"EGFR": 50, "ESR1": 50}

    def test_a_short_target_does_not_block_the_others(self) -> None:
        result = assemble_background(
            {
                "EGFR": contribution("a", 3),
                "ESR1": contribution("b", 100),
                "CA2": contribution("c", 100),
            },
            max_total=60,
            families={
                "EGFR": "protein kinase",
                "ESR1": "nuclear receptor",
                "CA2": "carbonic anhydrase",
            },
        )
        assert len(result.smiles) == 60
        assert result.per_target["EGFR"] == 3
        # The remaining 57 split as evenly as the rotation allows.
        assert abs(result.per_target["ESR1"] - result.per_target["CA2"]) <= 1

    def test_terminates_when_every_queue_is_exhausted(self) -> None:
        result = assemble_background(
            {"EGFR": contribution("a", 5)}, max_total=10_000
        )
        assert len(result.smiles) == 5

    def test_order_is_independent_of_dict_ordering(self) -> None:
        """Every docking shard rebuilds this; they must agree exactly."""
        forward = assemble_background(
            {"AR": contribution("a", 10), "CA2": contribution("b", 10)},
            max_total=12,
        )
        reversed_input = assemble_background(
            {"CA2": contribution("b", 10), "AR": contribution("a", 10)},
            max_total=12,
        )
        assert forward.smiles == reversed_input.smiles


class TestExclusion:
    def test_compounds_from_the_target_itself_are_removed(self) -> None:
        """Leaving them in puts actives in the background.

        A compound tested against this target carries its warhead, so including
        it raises the background prevalence of exactly the feature the check is
        trying to identify -- suppressing the signal rather than adding noise.
        """
        result = assemble_background(
            {"EGFR": contribution("a", 10)},
            exclude_molecule_ids=["a0", "a1", "a2"],
            max_total=100,
        )
        assert len(result.smiles) == 7
        assert result.n_excluded_overlap == 3

    def test_exclusion_does_not_cost_a_target_its_turn(self) -> None:
        """Otherwise a heavily-overlapping target silently contributes nothing."""
        result = assemble_background(
            {
                "EGFR": contribution("a", 20),
                "ESR1": contribution("b", 20),
            },
            exclude_molecule_ids=[f"a{i}" for i in range(10)],
            max_total=20,
        )
        assert result.per_target["EGFR"] == 10
        assert result.per_target["ESR1"] == 10

    def test_promiscuous_compounds_are_counted_once(self) -> None:
        shared = [("shared1", "CCO"), ("shared2", "CCN")]
        result = assemble_background(
            {
                "EGFR": [*shared, ("a1", "CCC")],
                "ESR1": [*shared, ("b1", "CCCC")],
            },
            max_total=100,
        )
        assert len(result.smiles) == 4
        assert result.n_duplicates == 2

    def test_blank_structures_are_skipped_not_counted(self) -> None:
        result = assemble_background(
            {"EGFR": [("a1", ""), ("a2", "   "), ("a3", "CCO")]},
            max_total=100,
        )
        assert result.smiles == ("CCO",)
        assert result.n_duplicates == 0
        assert result.n_excluded_overlap == 0


class TestUsability:
    def test_a_small_set_is_not_usable(self) -> None:
        result = assemble_background(
            {
                "EGFR": contribution("a", 20),
                "ESR1": contribution("b", 20),
                "CA2": contribution("c", 20),
            },
            max_total=60,
            families={
                "EGFR": "protein kinase",
                "ESR1": "nuclear receptor",
                "CA2": "carbonic anhydrase",
            },
        )
        assert len(result.smiles) == 60
        assert not result.is_usable
        assert "TOO SMALL OR TOO NARROW" in result.describe()

    def test_one_family_is_not_usable_however_large(self) -> None:
        result = assemble_background(
            {
                "EGFR": contribution("a", 500),
                "CDK2": contribution("b", 500),
                "MAPK14": contribution("c", 500),
            },
            max_total=1200,
            families=dict.fromkeys(["EGFR", "CDK2", "MAPK14"], "protein kinase"),
        )
        assert len(result.smiles) == 1200
        assert len(result.families) == 1
        assert not result.is_usable

    def test_large_and_broad_is_usable(self) -> None:
        result = assemble_background(
            {
                "EGFR": contribution("a", 200),
                "ESR1": contribution("b", 200),
                "CA2": contribution("c", 200),
            },
            max_total=600,
            families={
                "EGFR": "protein kinase",
                "ESR1": "nuclear receptor",
                "CA2": "carbonic anhydrase",
            },
        )
        assert result.is_usable
        assert "TOO SMALL" not in result.describe()


class TestReporting:
    def test_presumption_is_stated_not_implied(self) -> None:
        """These compounds were never tested against the target.

        The docking decoys are measured weak binders; these are not. A reader has
        to be able to tell the two apart from the output alone.
        """
        result = assemble_background(
            {"EGFR": contribution("a", 10)}, max_total=10
        )
        assert "PRESUMED non-binders" in result.describe()

    def test_failures_are_reported(self) -> None:
        result = assemble_background(
            {"EGFR": contribution("a", 10)},
            max_total=10,
            failures={"ESR1": "no SINGLE PROTEIN target"},
        )
        description = result.describe()
        assert "not retrieved" in description
        assert "ESR1" in description

    def test_empty_set_describes_itself(self) -> None:
        result = assemble_background({}, failures={"EGFR": "HTTP 500"})
        assert not result.smiles
        assert not result.is_usable
        assert "empty" in result.describe()
        assert "EGFR" in result.describe()

    def test_summary_is_json_serialisable(self) -> None:
        import json

        result = assemble_background(
            {"EGFR": contribution("a", 10)}, max_total=10
        )
        payload = summarise(result)
        assert json.loads(json.dumps(payload))["n_compounds"] == 10
        assert "presumed non-binders" in payload["source"]


class TestTargetList:
    def test_accessions_are_unique(self) -> None:
        accessions = [target.accession for target in UNRELATED_TARGETS]
        assert len(accessions) == len(set(accessions))

    def test_enough_families_to_be_usable_at_all(self) -> None:
        """A target list below the family floor could never produce a usable set."""
        families = {target.family for target in UNRELATED_TARGETS}
        assert len(families) >= BackgroundSet.MIN_FAMILIES

    def test_no_protease_in_the_background(self) -> None:
        """The point is unrelated chemistry.

        A protease in this list would share the transition-state motifs the check
        is meant to identify, which is the same mistake as using the target's own
        weak binders -- just harder to see.
        """
        families = " ".join(target.family for target in UNRELATED_TARGETS).lower()
        assert "protease" not in families
        assert "peptidase" not in families
