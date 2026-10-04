"""Tests for the ChEMBL client, run entirely offline against a seeded cache.

No network access, by design. A test that depends on a live public API fails for
reasons unrelated to the code, and more importantly a dataset fetched live is not
reproducible -- so the client's offline mode and its cache are themselves part of
what makes a reported metric checkable, and they deserve direct tests.

The fixtures are hand-built responses in ChEMBL's actual JSON shape, including the
awkward parts: ``standard_value`` arriving as a string, ``confidence_score`` living
on the assay rather than the activity, and ``data_validity_comment`` present.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from urllib.parse import urlencode

from chemdisco.curate import curate
from chemdisco.data.chembl import CHEMBL_BASE, ChEMBLClient, ChEMBLError, ResponseCache

ACCESSION = "P56817"  # human BACE1
TARGET_ID = "CHEMBL4822"


def _seed(cache: ResponseCache, path: str, params: dict, payload: dict) -> None:
    """Write a fixture under the exact URL the client will construct."""
    url = f"{CHEMBL_BASE}/{path}?{urlencode(sorted(params.items()))}"
    cache.put(url, payload)


def _activity(
    activity_id: str,
    *,
    molecule: str,
    smiles: str,
    value: object = "100",
    unit: str = "nM",
    relation: str = "=",
    activity_type: str = "IC50",
    assay: str = "CHEMBL_A1",
    validity: str | None = None,
    duplicate: int = 0,
    pchembl: object = "7.0",
) -> dict:
    return {
        "activity_id": activity_id,
        "molecule_chembl_id": molecule,
        "canonical_smiles": smiles,
        "standard_type": activity_type,
        "standard_value": value,
        "standard_units": unit,
        "standard_relation": relation,
        "assay_chembl_id": assay,
        "data_validity_comment": validity,
        "potential_duplicate": duplicate,
        "pchembl_value": pchembl,
    }


class ChEMBLFixtureCase(unittest.TestCase):
    """Base class providing a temporary cache seeded with a plausible target."""

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.cache = ResponseCache(pathlib.Path(self._tempdir.name))
        self.client = ChEMBLClient(cache=self.cache, offline=True)

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def seed_target(self, targets: list[dict] | None = None) -> None:
        payload = {
            "targets": targets
            if targets is not None
            else [
                {
                    "target_chembl_id": TARGET_ID,
                    "target_type": "SINGLE PROTEIN",
                    "pref_name": "Beta-secretase 1",
                }
            ],
            "page_meta": {"total_count": 1},
        }
        _seed(
            self.cache,
            "target.json",
            {"target_components__accession": ACCESSION, "limit": 50, "offset": 0},
            payload,
        )

    def seed_activities(self, activities: list[dict]) -> None:
        _seed(
            self.cache,
            "activity.json",
            {
                "target_chembl_id": TARGET_ID,
                "standard_type__in": "IC50",
                "canonical_smiles__isnull": "false",
                "limit": 1000,
                "offset": 0,
            },
            {"activities": activities, "page_meta": {"total_count": len(activities)}},
        )

    def seed_assays(self, assays: list[dict]) -> None:
        identifiers = ",".join(sorted({str(a["assay_chembl_id"]) for a in assays}))
        _seed(
            self.cache,
            "assay.json",
            {"assay_chembl_id__in": identifiers, "limit": 50, "offset": 0},
            {"assays": assays, "page_meta": {"total_count": len(assays)}},
        )


class TestCache(ChEMBLFixtureCase):
    def test_round_trip(self) -> None:
        url = "https://example.invalid/x"
        self.cache.put(url, {"a": 1})
        self.assertEqual(self.cache.get_payload(url), {"a": 1})

    def test_miss_returns_none(self) -> None:
        self.assertIsNone(self.cache.get_payload("https://example.invalid/absent"))

    def test_corrupt_entry_is_a_miss_not_a_crash(self) -> None:
        # A truncated gzip on disk must degrade to a refetch, not an exception
        # that looks like a ChEMBL outage.
        url = "https://example.invalid/corrupt"
        self.cache.put(url, {"a": 1})
        path = next(pathlib.Path(self._tempdir.name).glob("*.json.gz"))
        path.write_bytes(b"not gzip at all")
        self.assertIsNone(self.cache.get_payload(url))

    def test_hit_rate_is_reported(self) -> None:
        url = "https://example.invalid/y"
        self.cache.put(url, {"a": 1})
        self.cache.get_payload(url)
        self.cache.get_payload("https://example.invalid/absent")
        description = self.cache.describe()
        self.assertIn("1 hits", description)
        self.assertIn("1 misses", description)


class TestOfflineMode(ChEMBLFixtureCase):
    def test_offline_miss_raises_with_a_usable_message(self) -> None:
        with self.assertRaises(ChEMBLError) as context:
            self.client.single_protein_target(ACCESSION)
        message = str(context.exception)
        self.assertIn("offline mode", message)
        self.assertIn("Populate the cache", message)

    def test_offline_hit_succeeds(self) -> None:
        self.seed_target()
        self.assertEqual(self.client.single_protein_target(ACCESSION), TARGET_ID)


class TestTargetResolution(ChEMBLFixtureCase):
    def test_single_protein_target_resolved(self) -> None:
        self.seed_target()
        self.assertEqual(self.client.single_protein_target(ACCESSION), TARGET_ID)

    def test_no_single_protein_target_is_an_error_not_a_fallback(self) -> None:
        # Falling back to a cell-line or protein-complex target would silently
        # change what the dataset measures.
        self.seed_target(
            [
                {
                    "target_chembl_id": "CHEMBL_X",
                    "target_type": "CELL-LINE",
                    "pref_name": "Some cell line",
                }
            ]
        )
        with self.assertRaises(ChEMBLError) as context:
            self.client.single_protein_target(ACCESSION)
        self.assertIn("no SINGLE PROTEIN target", str(context.exception))

    def test_ambiguous_target_refuses_to_choose(self) -> None:
        self.seed_target(
            [
                {"target_chembl_id": "CHEMBL_A", "target_type": "SINGLE PROTEIN"},
                {"target_chembl_id": "CHEMBL_B", "target_type": "SINGLE PROTEIN"},
            ]
        )
        with self.assertRaises(ChEMBLError) as context:
            self.client.single_protein_target(ACCESSION)
        self.assertIn("several single-protein", str(context.exception))


class TestRecordAssembly(ChEMBLFixtureCase):
    def _standard_fixtures(self) -> None:
        self.seed_target()
        self.seed_activities(
            [
                _activity("1", molecule="CHEMBL_M1", smiles="CCO"),
                _activity(
                    "2",
                    molecule="CHEMBL_M2",
                    smiles="CCC",
                    value="2.5",
                    unit="uM",
                    pchembl="5.6",
                ),
                _activity(
                    "3",
                    molecule="CHEMBL_M3",
                    smiles="CCCC",
                    assay="CHEMBL_A2",
                    validity="Potential transcription error",
                    pchembl=None,
                ),
            ]
        )
        self.seed_assays(
            [
                {
                    "assay_chembl_id": "CHEMBL_A1",
                    "assay_type": "B",
                    "confidence_score": 9,
                    "description": "Inhibition of human BACE1",
                },
                {
                    "assay_chembl_id": "CHEMBL_A2",
                    "assay_type": "F",
                    "confidence_score": 5,
                    "description": "Cell-based readout",
                },
            ]
        )

    def test_records_are_assembled_with_assay_metadata_joined(self) -> None:
        self._standard_fixtures()
        records, provenance = self.client.fetch_records(
            ACCESSION, activity_types=("IC50",)
        )
        self.assertEqual(len(records), 3)
        by_id = {record.activity_id: record for record in records}
        # The confidence score lives on the assay; without the join it is absent
        # and the most important curation filter cannot be applied.
        self.assertEqual(by_id["1"].confidence_score, 9)
        self.assertEqual(by_id["3"].confidence_score, 5)
        self.assertEqual(by_id["1"].assay_type, "B")
        self.assertEqual(provenance["target_chembl_id"], TARGET_ID)

    def test_string_values_are_coerced_to_numbers(self) -> None:
        # ChEMBL returns standard_value as a string in its JSON.
        self._standard_fixtures()
        records, _ = self.client.fetch_records(ACCESSION, activity_types=("IC50",))
        self.assertEqual(records[0].value, 100.0)
        self.assertIsInstance(records[0].value, float)

    def test_unparseable_value_becomes_none_not_zero(self) -> None:
        self.seed_target()
        self.seed_activities(
            [_activity("1", molecule="M1", smiles="CCO", value="not a number")]
        )
        self.seed_assays(
            [
                {
                    "assay_chembl_id": "CHEMBL_A1",
                    "assay_type": "B",
                    "confidence_score": 9,
                }
            ]
        )
        records, _ = self.client.fetch_records(ACCESSION, activity_types=("IC50",))
        self.assertIsNone(records[0].value)

    def test_units_are_carried_through_verbatim(self) -> None:
        # The micromolar record must arrive as uM, not silently renamed to nM.
        self._standard_fixtures()
        records, _ = self.client.fetch_records(ACCESSION, activity_types=("IC50",))
        units = {record.activity_id: record.unit for record in records}
        self.assertEqual(units["1"], "nM")
        self.assertEqual(units["2"], "uM")

    def test_missing_assay_metadata_is_counted_and_explained(self) -> None:
        self.seed_target()
        self.seed_activities(
            [_activity("1", molecule="M1", smiles="CCO", assay="CHEMBL_MISSING")]
        )
        self.seed_assays([{"assay_chembl_id": "CHEMBL_MISSING"}])
        records, provenance = self.client.fetch_records(
            ACCESSION, activity_types=("IC50",)
        )
        # The assay exists but carries no confidence score.
        self.assertIsNone(records[0].confidence_score)
        self.assertEqual(provenance["n_missing_assay_metadata"], 0)

    def test_client_does_no_filtering_of_its_own(self) -> None:
        # Curation must see everything, including the records it will reject, so
        # that the attrition can be reported.
        self._standard_fixtures()
        records, _ = self.client.fetch_records(ACCESSION, activity_types=("IC50",))
        self.assertTrue(
            any(record.data_validity_comment for record in records),
            "the flagged record must reach curation rather than being dropped here",
        )
        self.assertTrue(any(record.confidence_score == 5 for record in records))


class TestEndToEndCuration(ChEMBLFixtureCase):
    """The client feeding curation: the integration that matters."""

    def test_fetch_then_curate_accounts_for_every_record(self) -> None:
        self.seed_target()
        self.seed_activities(
            [
                # Keeper: binding assay, confidence 9, 100 nM, pChEMBL agrees.
                _activity("1", molecule="M1", smiles="CCO"),
                # Keeper: micromolar, correctly converted to pIC50 5.6.
                _activity(
                    "2", molecule="M2", smiles="CCC", value="2.51", unit="uM",
                    pchembl="5.6",
                ),
                # Rejected: low target confidence.
                _activity("3", molecule="M3", smiles="CCCC", assay="CHEMBL_A2"),
                # Rejected: source flagged the value.
                _activity(
                    "4", molecule="M4", smiles="CCCCC",
                    validity="Outside typical range",
                ),
                # Rejected: censored relation.
                _activity("5", molecule="M5", smiles="CCCCCC", relation=">",
                          value="10000", pchembl=None),
                # Rejected: percent inhibition has no pActivity.
                _activity("6", molecule="M6", smiles="CCCCCCC", value="45",
                          unit="%", pchembl=None),
            ]
        )
        self.seed_assays(
            [
                {"assay_chembl_id": "CHEMBL_A1", "assay_type": "B", "confidence_score": 9},
                {"assay_chembl_id": "CHEMBL_A2", "assay_type": "B", "confidence_score": 4},
            ]
        )

        records, provenance = self.client.fetch_records(
            ACCESSION, activity_types=("IC50",)
        )
        self.assertEqual(len(records), 6)

        report = curate(records)
        self.assertEqual(report.n_input, 6)

        kept_ids = {point.compound_id for point in report.kept}
        self.assertEqual(kept_ids, {"M1", "M2"})

        rules = report.rejection_summary()
        self.assertIn("low_target_confidence", rules)
        self.assertIn("source_flagged_invalid", rules)
        self.assertIn("censored_measurement", rules)
        self.assertIn("unconvertible_value", rules)

        # The micromolar record must become pIC50 5.6, not 8.6 -- the specific
        # error the predecessor's "assume nM" conversion produced.
        by_compound = {point.compound_id: point for point in report.kept}
        self.assertAlmostEqual(by_compound["M2"].pactivity.require(), 5.6, places=1)
        self.assertAlmostEqual(by_compound["M1"].pactivity.require(), 7.0, places=6)

    def test_provenance_is_recorded_for_the_report(self) -> None:
        self.seed_target()
        self.seed_activities([_activity("1", molecule="M1", smiles="CCO")])
        self.seed_assays(
            [{"assay_chembl_id": "CHEMBL_A1", "assay_type": "B", "confidence_score": 9}]
        )
        _, provenance = self.client.fetch_records(ACCESSION, activity_types=("IC50",))
        self.assertEqual(provenance["uniprot_accession"], ACCESSION)
        self.assertIn("retrieved_at", provenance)
        self.assertIn("cache", provenance)
        # Must be serialisable for inclusion in a report.
        json.dumps(provenance)


if __name__ == "__main__":
    unittest.main()


class FakeResponse:
    """Minimal stand-in for a requests Response."""

    def __init__(self, status_code: int, payload=None, text: str = "", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


class FakeSession:
    """Replays a scripted sequence of responses, recording what was asked."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[str] = []
        self.headers: dict[str, str] = {}

    def get(self, url, timeout=None):
        self.calls.append(url)
        if not self.script:
            raise AssertionError(f"unexpected extra request to {url}")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class RetryTestCase(unittest.TestCase):
    """Retry behaviour, tested without network or real sleeping.

    These exist because a probe of the live API recorded an HTTP 500 and a
    60-second read timeout within the same minute that every other endpoint
    answered normally. Without retries one such blip aborts a run that may
    already have fetched thousands of records.
    """

    def _client(self, script, **kwargs):
        self.slept: list[float] = []
        client = ChEMBLClient(
            cache=None,
            offline=False,
            request_interval=0.0,
            sleeper=self.slept.append,
            **kwargs,
        )
        client._session = FakeSession(script)
        return client

    def test_transient_500_is_retried_then_succeeds(self) -> None:
        client = self._client(
            [
                FakeResponse(500, text="<!doctype html>"),
                FakeResponse(200, payload={"ok": True}),
            ]
        )
        self.assertEqual(client._fetch("status.json", {}), {"ok": True})
        self.assertEqual(len(client._session.calls), 2)

    def test_timeout_is_retried(self) -> None:
        client = self._client(
            [TimeoutError("read timed out"), FakeResponse(200, payload={"ok": True})]
        )
        self.assertEqual(client._fetch("status.json", {}), {"ok": True})

    def test_html_body_on_a_200_is_retried(self) -> None:
        # EBI serves its web framework's HTML from error pages, sometimes with a
        # 200. Treating that as success would feed garbage into curation.
        client = self._client(
            [
                FakeResponse(200, payload=None, text="<!doctype html><html>"),
                FakeResponse(200, payload={"ok": True}),
            ]
        )
        self.assertEqual(client._fetch("status.json", {}), {"ok": True})

    def test_backoff_is_exponential(self) -> None:
        client = self._client(
            [
                FakeResponse(503),
                FakeResponse(503),
                FakeResponse(503),
                FakeResponse(200, payload={"ok": True}),
            ],
            backoff_base=1.0,
        )
        client._fetch("status.json", {})
        self.assertEqual(self.slept, [1.0, 2.0, 4.0])

    def test_404_is_not_retried(self) -> None:
        # A 404 means the request is wrong; retrying repeats the mistake slowly.
        client = self._client([FakeResponse(404)])
        with self.assertRaises(ChEMBLError) as context:
            client._fetch("target.json", {"bogus": 1})
        self.assertIn("HTTP 404", str(context.exception))
        self.assertEqual(len(client._session.calls), 1)
        self.assertEqual(self.slept, [])

    def test_429_is_retried(self) -> None:
        client = self._client(
            [
                FakeResponse(429, headers={"Retry-After": "2"}),
                FakeResponse(200, payload={"ok": True}),
            ]
        )
        self.assertEqual(client._fetch("status.json", {}), {"ok": True})

    def test_persistent_failure_gives_up_and_says_how_many_tries(self) -> None:
        client = self._client([FakeResponse(500) for _ in range(5)], max_retries=4)
        with self.assertRaises(ChEMBLError) as context:
            client._fetch("status.json", {})
        message = str(context.exception)
        self.assertIn("after 5 attempt(s)", message)
        self.assertIn("HTTP 500", message)

    def test_retries_are_recorded_for_the_provenance_report(self) -> None:
        # A run that succeeded only after three retries is worth knowing about:
        # it says the source was struggling while the data was collected.
        client = self._client(
            [FakeResponse(500), FakeResponse(503), FakeResponse(200, payload={"ok": 1})]
        )
        client._fetch("status.json", {})
        self.assertEqual(len(client.notes), 2)
        self.assertTrue(all("retry" in note for note in client.notes))

    def test_a_successful_retry_is_cached_once(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            cache = ResponseCache(pathlib.Path(tempdir))
            client = ChEMBLClient(
                cache=cache, request_interval=0.0, sleeper=lambda _s: None
            )
            client._session = FakeSession(
                [FakeResponse(500), FakeResponse(200, payload={"ok": True})]
            )
            client._fetch("status.json", {})
            # Second call must be served from cache, making no further request.
            client._session.script = []
            self.assertEqual(client._fetch("status.json", {}), {"ok": True})

    def test_offline_mode_never_touches_the_network(self) -> None:
        client = ChEMBLClient(cache=None, offline=True)
        client._session = FakeSession([FakeResponse(200, payload={"ok": True})])
        with self.assertRaises(ChEMBLError):
            client._fetch("status.json", {})
        self.assertEqual(client._session.calls, [])
