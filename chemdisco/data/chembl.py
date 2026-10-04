"""ChEMBL client: retrieving bioactivity data with the fields curation needs.

Two design decisions worth explaining.

**Everything is cached to disk.** Not an optimisation. A reported model metric must
be reproducible, and ChEMBL changes between releases -- activities are added,
values corrected, assays reassigned. A result computed against "ChEMBL, sometime
last March" cannot be checked. The cache records the release and the exact
response, so a run can be repeated and a difference traced to the data rather than
the code. It also makes the test suite runnable without network access.

**Assay metadata is fetched separately and joined.** The ``activity`` endpoint does
not reliably carry ``confidence_score``, which lives on the ``assay`` resource, and
confidence score is the single most important curation field: it says whether the
measurement is against the protein you asked for or against a cell, a complex or a
family. Fetching activities alone and skipping the join -- which the predecessor
project did -- means that filter cannot be applied at all.

Target resolution goes through UniProt accession rather than target name. Searching
ChEMBL for the string "BACE1" returns the human enzyme, the mouse orthologue,
several cell-based assays and a handful of unrelated entries whose description
happens to contain the term. An accession resolves to exactly one protein.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
from urllib.parse import urlencode

from ..curate.records import ActivityRecord

logger = logging.getLogger(__name__)

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

#: Page size. ChEMBL caps this at 1000 and rejects larger values.
PAGE_LIMIT = 1000

#: Seconds between requests. ChEMBL asks for considerate use and does not publish
#: a hard rate limit; this is deliberately unhurried.
REQUEST_INTERVAL = 0.34


class ChEMBLError(RuntimeError):
    """Raised when ChEMBL cannot be reached or returns something unusable."""


@dataclass(slots=True)
class ResponseCache:
    """On-disk cache of raw API responses, keyed by request URL.

    Stores gzipped JSON. An entry is immutable once written: a changed response
    from ChEMBL produces a new key only if the URL changed, so to pick up a new
    ChEMBL release you clear the cache deliberately rather than having results
    shift under you mid-analysis.
    """

    directory: pathlib.Path
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        self.directory = pathlib.Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, url: str) -> pathlib.Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        return self.directory / f"{digest}.json.gz"

    def get(self, url: str) -> dict[str, Any] | None:
        path = self._path(url)
        if not path.exists():
            self.misses += 1
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.hits += 1
            return payload
        except Exception as error:
            # A corrupt entry is a cache miss, not a failure. Report it so a
            # disk problem does not masquerade as a ChEMBL problem.
            logger.warning("cache entry %s unreadable (%s); refetching", path.name, error)
            self.misses += 1
            return None

    def put(self, url: str, payload: dict[str, Any]) -> None:
        path = self._path(url)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump({"_url": url, "_payload": payload}, handle)

    def get_payload(self, url: str) -> dict[str, Any] | None:
        entry = self.get(url)
        if entry is None:
            return None
        return entry.get("_payload", entry)

    def describe(self) -> str:
        total = self.hits + self.misses
        rate = f"{self.hits / total:.0%}" if total else "n/a"
        return (
            f"cache at {self.directory}: {self.hits} hits, {self.misses} misses "
            f"({rate} hit rate)"
        )


@dataclass(slots=True)
class ChEMBLClient:
    """Minimal ChEMBL REST client, cached and rate-limited.

    Attributes:
        cache: Response cache. Strongly recommended; passing ``None`` makes runs
            unreproducible.
        offline: Refuse to make network requests and serve only from cache.
            Raises on a miss, which is what makes a test suite deterministic
            instead of quietly skipping data.
        request_interval: Seconds between network requests.
    """

    cache: ResponseCache | None = None
    offline: bool = False
    request_interval: float = REQUEST_INTERVAL
    _last_request: float = 0.0
    _session: Any = None
    notes: list[str] = field(default_factory=list)

    def _get_session(self):
        if self._session is None:
            try:
                import requests
            except ImportError as error:  # pragma: no cover
                raise ChEMBLError(
                    "the 'requests' package is required for network access; "
                    "install it, or run with offline=True against a populated cache"
                ) from error
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "Accept": "application/json",
                    # A descriptive agent is courtesy to a free public service and
                    # lets EBI contact a misbehaving client rather than block it.
                    "User-Agent": "chemdisco/0.1 (research use; +https://github.com/fasn98/chemdisco)",
                }
            )
        return self._session

    def _fetch(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{CHEMBL_BASE}/{path}?{urlencode(sorted(params.items()))}"

        if self.cache is not None:
            cached = self.cache.get_payload(url)
            if cached is not None:
                return cached

        if self.offline:
            raise ChEMBLError(
                f"offline mode and no cached response for {url}. Populate the "
                "cache with a network run first, or commit the fixture."
            )

        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)

        session = self._get_session()
        try:
            response = session.get(url, timeout=60)
        except Exception as error:
            raise ChEMBLError(f"request to {url} failed: {error}") from error
        self._last_request = time.monotonic()

        if response.status_code != 200:
            raise ChEMBLError(
                f"ChEMBL returned HTTP {response.status_code} for {url}"
            )
        try:
            payload = response.json()
        except Exception as error:
            raise ChEMBLError(f"ChEMBL returned unparseable JSON for {url}") from error

        if self.cache is not None:
            self.cache.put(url, payload)
        return payload

    def _paginate(
        self, path: str, params: dict[str, Any], collection: str, *, max_records: int
    ) -> list[dict[str, Any]]:
        """Walk ChEMBL's paginated responses.

        Pagination is driven by an explicit offset rather than by following
        ``page_meta.next``, because the cache keys on the full URL and a
        server-supplied next link can carry a session-dependent form that would
        defeat caching.
        """
        collected: list[dict[str, Any]] = []
        offset = 0
        while len(collected) < max_records:
            page_params = dict(params)
            page_params["limit"] = min(PAGE_LIMIT, max_records - len(collected))
            page_params["offset"] = offset
            payload = self._fetch(path, page_params)

            items = payload.get(collection, [])
            if not items:
                break
            collected.extend(items)

            meta = payload.get("page_meta", {})
            total = meta.get("total_count")
            offset += len(items)
            if total is not None and offset >= total:
                break
            if len(items) < page_params["limit"]:
                break
        return collected[:max_records]

    # -- target resolution ------------------------------------------------

    def targets_for_uniprot(self, accession: str) -> list[dict[str, Any]]:
        """ChEMBL targets whose components include ``accession``.

        Using an accession rather than a name is what makes this unambiguous. A
        name search for "BACE1" also returns the mouse orthologue and assorted
        cell-based assays, and pooling those into one dataset mixes species and
        assay systems.
        """
        return self._paginate(
            "target.json",
            {"target_components__accession": accession},
            "targets",
            max_records=50,
        )

    def single_protein_target(self, accession: str) -> str:
        """The single-protein ChEMBL target id for ``accession``.

        Raises:
            ChEMBLError: when no single-protein target exists, or when several do
                and the choice is therefore not ours to make silently.
        """
        targets = self.targets_for_uniprot(accession)
        single = [
            target
            for target in targets
            if target.get("target_type") == "SINGLE PROTEIN"
        ]
        if not single:
            raise ChEMBLError(
                f"no SINGLE PROTEIN target in ChEMBL for UniProt {accession}. "
                f"Found {len(targets)} target(s) of other types; a dataset built "
                "from those would mix protein complexes or cell lines with the "
                "intended protein."
            )
        if len(single) > 1:
            identifiers = ", ".join(
                str(target.get("target_chembl_id")) for target in single
            )
            raise ChEMBLError(
                f"UniProt {accession} maps to several single-protein ChEMBL "
                f"targets ({identifiers}). Pick one explicitly -- merging them "
                "without checking what they represent risks pooling unrelated "
                "assay sets."
            )
        return str(single[0]["target_chembl_id"])

    # -- activities -------------------------------------------------------

    def activities_for_target(
        self,
        target_chembl_id: str,
        *,
        activity_types: Sequence[str] = ("IC50", "EC50", "Ki", "Kd"),
        max_records: int = 20_000,
    ) -> list[dict[str, Any]]:
        """Raw activity records for a target.

        No filtering beyond the activity type and the presence of a structure.
        Curation is :mod:`chemdisco.curate`'s job and it reports what it removed;
        filtering here would hide that accounting.
        """
        return self._paginate(
            "activity.json",
            {
                "target_chembl_id": target_chembl_id,
                "standard_type__in": ",".join(activity_types),
                "canonical_smiles__isnull": "false",
            },
            "activities",
            max_records=max_records,
        )

    def assay_metadata(
        self, assay_ids: Iterable[str], *, batch_size: int = 50
    ) -> dict[str, dict[str, Any]]:
        """Assay records keyed by id, for the confidence-score join.

        Fetched in batches via ``assay_chembl_id__in``. Without this, the
        target-confidence filter cannot be applied, and every measurement has to
        be treated as if it were a direct single-protein assay.
        """
        unique = sorted({str(identifier) for identifier in assay_ids if identifier})
        metadata: dict[str, dict[str, Any]] = {}

        for start in range(0, len(unique), batch_size):
            batch = unique[start : start + batch_size]
            records = self._paginate(
                "assay.json",
                {"assay_chembl_id__in": ",".join(batch)},
                "assays",
                max_records=batch_size,
            )
            for record in records:
                identifier = record.get("assay_chembl_id")
                if identifier:
                    metadata[str(identifier)] = record
        return metadata

    def fetch_records(
        self,
        accession: str,
        *,
        activity_types: Sequence[str] = ("IC50", "EC50"),
        max_records: int = 20_000,
    ) -> tuple[list[ActivityRecord], dict[str, Any]]:
        """Fetch and assemble activity records ready for curation.

        Args:
            accession: UniProt accession of the target protein.
            activity_types: ChEMBL ``standard_type`` values to retrieve. The
                default keeps the functional family together; see
                :data:`chemdisco.curate.filters.ACTIVITY_FAMILIES` for why
                binding constants are not pooled with it by default.
            max_records: Ceiling on activities retrieved.

        Returns:
            ``(records, provenance)``. The provenance dictionary holds the
            resolved target id, counts and the ChEMBL release, and belongs in any
            report built from this data.
        """
        target_chembl_id = self.single_protein_target(accession)
        activities = self.activities_for_target(
            target_chembl_id,
            activity_types=activity_types,
            max_records=max_records,
        )

        assay_ids = [activity.get("assay_chembl_id") for activity in activities]
        assays = self.assay_metadata(assay_ids)

        missing_assay_metadata = 0
        records: list[ActivityRecord] = []

        for activity in activities:
            assay_id = str(activity.get("assay_chembl_id") or "")
            assay = assays.get(assay_id, {})
            if not assay:
                missing_assay_metadata += 1

            value = activity.get("standard_value")
            try:
                numeric_value = float(value) if value is not None else None
            except (TypeError, ValueError):
                numeric_value = None

            pchembl = activity.get("pchembl_value")
            try:
                pchembl_numeric = float(pchembl) if pchembl is not None else None
            except (TypeError, ValueError):
                pchembl_numeric = None

            records.append(
                ActivityRecord(
                    activity_id=str(activity.get("activity_id") or ""),
                    compound_id=str(activity.get("molecule_chembl_id") or ""),
                    smiles=str(activity.get("canonical_smiles") or ""),
                    target_id=target_chembl_id,
                    activity_type=str(activity.get("standard_type") or ""),
                    value=numeric_value,
                    unit=activity.get("standard_units"),
                    relation=str(activity.get("standard_relation") or "="),
                    assay_id=assay_id,
                    assay_type=str(
                        assay.get("assay_type") or activity.get("assay_type") or ""
                    ),
                    confidence_score=assay.get("confidence_score"),
                    data_validity_comment=activity.get("data_validity_comment"),
                    potential_duplicate=bool(activity.get("potential_duplicate")),
                    pchembl_value=pchembl_numeric,
                    extra={
                        "assay_description": assay.get("description"),
                        "target_organism": assay.get("assay_organism"),
                    },
                )
            )

        provenance: dict[str, Any] = {
            "source": "ChEMBL REST API",
            "uniprot_accession": accession,
            "target_chembl_id": target_chembl_id,
            "activity_types": list(activity_types),
            "n_activities_retrieved": len(records),
            "n_distinct_assays": len(assays),
            "n_missing_assay_metadata": missing_assay_metadata,
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if missing_assay_metadata:
            provenance["warning"] = (
                f"{missing_assay_metadata} activities have no assay metadata, so "
                "their target-confidence score is unknown and the default "
                "curation policy will reject them. That is the intended "
                "behaviour, but it explains part of the attrition."
            )
        if self.cache is not None:
            provenance["cache"] = self.cache.describe()
        return records, provenance

    def release(self) -> str | None:
        """The ChEMBL release this data came from, for the provenance record."""
        try:
            payload = self._fetch("status.json", {})
        except ChEMBLError:
            return None
        return payload.get("chembl_db_version") or payload.get("chembl_release")
