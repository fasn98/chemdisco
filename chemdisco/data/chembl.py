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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from ..curate.records import ActivityRecord

logger = logging.getLogger(__name__)

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

#: Page size. ChEMBL caps this at 1000 and rejects larger values.
PAGE_LIMIT = 1000

#: Seconds between requests. ChEMBL asks for considerate use and does not publish
#: a hard rate limit; this is deliberately unhurried.
REQUEST_INTERVAL = 0.34

#: HTTP statuses worth retrying. 5xx are server-side faults and 429 is explicit
#: rate limiting; both are transient. Everything else in 4xx means the request is
#: wrong and retrying only repeats the mistake.
RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

#: Attempts after the first before giving up.
MAX_RETRIES = 4

#: First backoff delay in seconds; doubles each attempt (1, 2, 4, 8).
BACKOFF_BASE = 1.0

#: Per-request timeout. Generous: ChEMBL activity pages of 1000 records are slow
#: under load, and a short timeout turns a slow success into a failure.
REQUEST_TIMEOUT = 90


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
        max_retries: Attempts after the first before giving up on a transient
            failure.
        backoff_base: First retry delay in seconds; doubles each attempt.
        timeout: Per-request timeout in seconds.
        sleeper: How the client waits. Injected rather than calling
            ``time.sleep`` directly so the retry and rate-limit behaviour can be
            tested without the suite actually sleeping through the backoff --
            which would make a correct implementation take fifteen seconds to
            verify and tempt someone into not verifying it.
        notes: Retries and other events worth reporting alongside the data.
    """

    cache: ResponseCache | None = None
    offline: bool = False
    request_interval: float = REQUEST_INTERVAL
    max_retries: int = MAX_RETRIES
    backoff_base: float = BACKOFF_BASE
    timeout: float = REQUEST_TIMEOUT
    sleeper: Callable[[float], None] = time.sleep
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

    def _request_once(self, url: str) -> tuple[dict[str, Any] | None, str | None, bool]:
        """One attempt. Returns ``(payload, error, retryable)``."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_interval:
            self.sleeper(self.request_interval - elapsed)

        session = self._get_session()
        try:
            response = session.get(url, timeout=self.timeout)
        except Exception as error:
            # Timeouts and connection resets are the common transient failures
            # against a busy public service, so they are retryable.
            self._last_request = time.monotonic()
            return None, f"{type(error).__name__}: {error}", True
        self._last_request = time.monotonic()

        status = response.status_code
        if status == 200:
            try:
                return response.json(), None, False
            except Exception:
                # A 200 carrying HTML is what EBI serves from an error page or a
                # maintenance window. Retryable: the next attempt often succeeds.
                snippet = response.text[:200].replace("\n", " ")
                return None, f"HTTP 200 but the body was not JSON: {snippet}", True

        if status in RETRYABLE_STATUS:
            retry_after = response.headers.get("Retry-After")
            hint = f"; Retry-After: {retry_after}" if retry_after else ""
            return None, f"HTTP {status}{hint}", True

        # 4xx other than 429 means the request itself is wrong. Retrying would
        # only repeat the mistake more slowly.
        return None, f"HTTP {status}", False

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

        # Retry with exponential backoff.
        #
        # Not defensive padding: a probe of this API recorded HTTP 500 on
        # /status.json and a 60-second read timeout on a molecule lookup within
        # the same minute that every other endpoint answered normally. Without
        # retries a single such blip aborts a run that may already have fetched
        # thousands of records, and the whole dataset is lost to a hiccup in a
        # free public service that owes nobody an uptime guarantee.
        last_error = ""
        for attempt in range(self.max_retries + 1):
            payload, error, retryable = self._request_once(url)
            if payload is not None:
                if self.cache is not None:
                    self.cache.put(url, payload)
                return payload

            last_error = error or "unknown failure"
            if not retryable or attempt == self.max_retries:
                break

            delay = self.backoff_base * (2**attempt)
            self.notes.append(
                f"retry {attempt + 1}/{self.max_retries} after {last_error} "
                f"(waiting {delay:.1f}s): {url}"
            )
            self.sleeper(delay)

        raise ChEMBLError(
            f"ChEMBL request failed after {self.max_retries + 1} attempt(s): "
            f"{last_error} for {url}"
        )

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
