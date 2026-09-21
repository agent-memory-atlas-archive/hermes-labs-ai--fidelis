"""Deterministic, raw-source relation envelopes for comparative evidence.

The envelope is additive ingestion metadata.  It does not rewrite source text,
assert truth, retrieve records, or resolve ambiguous relations.  Chroma payload
metadata is scalar-only in supported deployments, so the versioned envelope is
transported as canonical JSON under :data:`PAYLOAD_FIELD`.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


FEATURE_ENV = "COGITO_RELATION_ENVELOPES_V1"
SCHEMA_VERSION = "fidelis.relation-envelope/v2"
LEGACY_SCHEMA_VERSION = "fidelis.relation-envelope/v1"
EXTRACTOR_VERSION = "deterministic-raw-v2"
RELATION_SEMANTICS_VERSION = "directional-relations/v2"
PAYLOAD_FIELD = "relation_envelope_v2_json"
LEGACY_PAYLOAD_FIELD = "relation_envelope_v1_json"
DECLARATIONS_FIELD = "relation_declarations_v2"
LEGACY_DECLARATIONS_FIELD = "relation_declarations_v1"
TRANSPORT_FIELDS_FIELD = "relation_envelope_transport_fields_json"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}(?:-\d{2}-\d{2})?\b")
_LINK_RE = re.compile(
    r"\b(?P<verb>supersede(?:s)?|replace(?:s)?|follow(?:s)?)\s+"
    r"(?:raw\s+)?record\s+"
    r"(?P<record>[A-Za-z0-9][A-Za-z0-9_.:-]{2,127})\b",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9_.-]*)(?:\s+[A-Z][A-Za-z0-9_.-]*){0,3}\b"
)
_AMBIGUITY_RE = re.compile(r"\b(?:may|might|possibly|perhaps|unclear|unknown)\b", re.I)
_PRIOR_RE = re.compile(
    r"\b(?:previously|formerly|before|earlier|prior|used to|was running|had used|"
    r"historical|predates?)\b",
    re.I,
)
_CURRENT_RE = re.compile(
    r"\b(?:now|currently|today|latest|is running|now uses|now runs|active locally)\b",
    re.I,
)
_CHANGE_RE = re.compile(
    r"\b(?:changed?|supersed(?:e|es|ed)|replac(?:e|es|ed)|migrat(?:e|es|ed)|"
    r"transition(?:ed)?|moved)\b",
    re.I,
)
_GENERIC_ENTITY_WORDS = {
    "After",
    "Before",
    "Currently",
    "Earlier",
    "Formerly",
    "Now",
    "On",
    "Previously",
    "Today",
}
_SOURCE_FIELDS = (
    "collection",
    "source",
    "source_kind",
    "source_pointer",
    "source_span",
)
SUPPORTED_RELATION_TYPES = (
    "changes_from",
    "follows",
    "precedes",
    "rationale_for",
    "supersedes",
)
_DECLARED_RELATION_TYPES = frozenset(SUPPORTED_RELATION_TYPES)
_RELATION_ROLE_ASSIGNMENTS: dict[str, dict[str, tuple[str, ...]]] = {
    # Direction is always source record -> target record.
    "changes_from": {
        "source": ("may_support_current", "may_support_change"),
        "target": ("may_support_prior",),
    },
    "follows": {
        "source": ("may_support_current",),
        "target": ("may_support_prior",),
    },
    "precedes": {
        "source": ("may_support_prior",),
        "target": ("may_support_current",),
    },
    "rationale_for": {
        "source": ("may_support_rationale",),
        "target": ("may_support_decision",),
    },
    "supersedes": {
        "source": ("may_support_current", "may_support_change"),
        "target": ("may_support_prior",),
    },
}


def relation_envelopes_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if environ is None else environ
    raw = str(source.get(FEATURE_ENV, "")).strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(
        f"{FEATURE_ENV} must be one of 1/0, true/false, yes/no, on/off"
    )


def _literal_entity_spans(
    text: str,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    declared = metadata.get("entity_literals")
    literals = (
        [str(value) for value in declared]
        if isinstance(declared, list)
        else []
    )
    if not literals:
        literals = [
            match.group(0)
            for match in _ENTITY_RE.finditer(text)
            if match.group(0) not in _GENERIC_ENTITY_WORDS
        ]
    spans: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for literal in literals:
        if not literal:
            continue
        for match in re.finditer(re.escape(literal), text):
            key = (literal, match.start(), match.end())
            if key in seen:
                continue
            seen.add(key)
            spans.append(
                {
                    "text": literal,
                    "start": match.start(),
                    "end": match.end(),
                    "derivation": "literal_raw_text_span",
                }
            )
    return sorted(spans, key=lambda item: (item["start"], item["end"], item["text"]))


def _roles(text: str) -> list[str]:
    # Record identifiers are transport syntax, not event language.  Mask
    # explicit link clauses so an id such as ``project-prior`` cannot create a
    # false prior-role tag.
    role_text = _LINK_RE.sub("", text)
    roles: list[str] = []
    if _PRIOR_RE.search(role_text):
        roles.append("may_support_prior")
    if _CURRENT_RE.search(role_text):
        roles.append("may_support_current")
    # The relation verb itself is valid change evidence even though the target
    # identifier following it is masked for prior/current role detection.
    if _CHANGE_RE.search(text):
        roles.append("may_support_change")
    return roles or ["unknown"]


def _scalar_or_none(value: Any) -> str | int | float | bool | None:
    return value if isinstance(value, (str, int, float, bool)) else None


def _relation_assignment(relation_type: str) -> dict[str, tuple[str, ...]]:
    return _RELATION_ROLE_ASSIGNMENTS.get(
        relation_type,
        {"source": (), "target": ()},
    )


def _relation_record(
    *,
    relation_type: str,
    source_record_id: str,
    target_record_id: str | None,
    source_span: dict[str, Any] | None,
    status: str,
    declaration_origin: str,
    declared_by: str,
) -> dict[str, Any]:
    assignment = _relation_assignment(relation_type)
    return {
        "type": relation_type,
        "semantics_version": RELATION_SEMANTICS_VERSION,
        "direction": "source_record_to_target_record",
        "source_record_id": source_record_id,
        "target_record_id": target_record_id,
        "facet_roles": {
            "source": list(assignment["source"]),
            "target": list(assignment["target"]),
        },
        "source_span": source_span,
        "status": status,
        "declaration": {
            "origin": declaration_origin,
            "declared_by": declared_by,
            "support": (
                "verbatim_raw_text_span"
                if source_span
                else "no_cross_record_support_span"
            ),
            "validation": "deterministic_structure_and_literal_span",
            "authority": "declaration_only_not_truth_verified",
        },
    }


def build_relation_envelope(
    *,
    record_id: str,
    text: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one deterministic envelope from raw text and declared metadata.

    Metadata may identify provenance or literal entity strings.  Entity strings
    are accepted only when they occur verbatim in ``text``.  Cross-record links
    are admitted only from explicit raw phrases such as
    ``"supersedes record <id>"``.  An ambiguity marker keeps links unresolved.
    """

    source = dict(metadata or {})
    entity_spans = _literal_entity_spans(text, source)
    time_expressions = [
        {
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
            "resolution": "literal_uninterpreted",
        }
        for match in _DATE_RE.finditer(text)
    ]
    ambiguous = bool(_AMBIGUITY_RE.search(text))
    relations: list[dict[str, Any]] = []
    linked_ids: list[str] = []
    for match in _LINK_RE.finditer(text):
        target = match.group("record")
        relation_type = {
            "supersede": "supersedes",
            "supersedes": "supersedes",
            "replace": "supersedes",
            "replaces": "supersedes",
            "follow": "follows",
            "follows": "follows",
        }[match.group("verb").lower()]
        relation = _relation_record(
            relation_type=relation_type,
            source_record_id=str(record_id),
            target_record_id=target,
            source_span={
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
            },
            status=(
                "unresolved_ambiguous"
                if ambiguous
                else "declared_in_raw_text_not_truth_verified"
            ),
            declaration_origin="raw_text_explicit_relation",
            declared_by="source_record",
        )
        relations.append(relation)
        if not ambiguous and target not in linked_ids:
            linked_ids.append(target)

    declaration_sets = (
        (source.get(DECLARATIONS_FIELD), "v2"),
        (source.get(LEGACY_DECLARATIONS_FIELD), "v1_legacy"),
    )
    for declarations, declaration_version in declaration_sets:
        if not isinstance(declarations, list):
            continue
        for declaration in declarations:
            if not isinstance(declaration, Mapping):
                continue
            relation_type = str(declaration.get("type") or "")
            target = str(declaration.get("target_record_id") or "")
            support_text = str(declaration.get("support_text") or "")
            declared_by = str(declaration.get("declared_by") or "caller")
            if (
                relation_type not in _DECLARED_RELATION_TYPES
                or not target
                or not support_text
                or declared_by not in {"caller", "source"}
            ):
                continue
            start = text.find(support_text)
            if start < 0:
                continue
            relation = _relation_record(
                relation_type=relation_type,
                source_record_id=str(record_id),
                target_record_id=target,
                source_span={
                    "text": support_text,
                    "start": start,
                    "end": start + len(support_text),
                },
                status=(
                    "unresolved_ambiguous"
                    if ambiguous
                    else (
                        "declared_in_source_metadata_with_literal_raw_span_"
                        "not_truth_verified"
                    )
                ),
                declaration_origin=(
                    "source_metadata_explicit_relation"
                    if declaration_version == "v2"
                    else "legacy_source_metadata_explicit_relation"
                ),
                declared_by=declared_by,
            )
            if relation not in relations:
                relations.append(relation)
            if not ambiguous and target not in linked_ids:
                linked_ids.append(target)

    # A raw record may explicitly describe an in-record transition without
    # naming another stored record.  Preserve the typed observation while
    # leaving its cross-record target unresolved.
    if not relations and _PRIOR_RE.search(text) and _CURRENT_RE.search(text):
        relations.append(
            _relation_record(
                relation_type="changes_from",
                source_record_id=str(record_id),
                target_record_id=None,
                source_span=None,
                status=(
                    "unresolved_ambiguous"
                    if ambiguous
                    else "unresolved_target_not_cross_record_link"
                ),
                declaration_origin="raw_text_in_record_transition",
                declared_by="source_record",
            )
        )

    provenance = {
        field: _scalar_or_none(source.get(field))
        for field in _SOURCE_FIELDS
    }
    provenance["identity_strength"] = (
        "record_and_source_linked"
        if record_id and any(provenance.values())
        else "record_only"
        if record_id
        else "unlinked"
    )
    unresolved: list[str] = []
    if not entity_spans:
        unresolved.append("entity")
    if not time_expressions:
        unresolved.append("event_time")
    if not relations:
        unresolved.append("typed_relation")
    if ambiguous:
        unresolved.append("ambiguity")

    candidate_roles = _roles(text)
    for relation in relations:
        if (
            relation.get("target_record_id")
            and relation.get("status") != "unresolved_ambiguous"
        ):
            facet_roles = relation.get("facet_roles")
            source_roles = (
                facet_roles.get("source", [])
                if isinstance(facet_roles, Mapping)
                else []
            )
        else:
            source_roles = []
        for role in source_roles:
            if role not in candidate_roles:
                candidate_roles.append(role)

    return {
        "schema_version": SCHEMA_VERSION,
        "relation_semantics_version": RELATION_SEMANTICS_VERSION,
        "extractor": {
            "name": "fidelis-deterministic-relation-extractor",
            "version": EXTRACTOR_VERSION,
            "runtime_llm_used": False,
        },
        "record": {
            "id": str(record_id),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "raw_text_preserved": True,
        },
        "provenance": provenance,
        "entity_spans": entity_spans,
        "event_time": {
            "expressions": time_expressions,
            "status": "literal_uninterpreted" if time_expressions else "unresolved",
        },
        "candidate_roles": candidate_roles,
        "relations": relations,
        "linked_raw_record_ids": linked_ids,
        "ambiguity_status": "unresolved" if ambiguous else "not_detected",
        "unresolved_fields": sorted(set(unresolved)),
        "authority": (
            "search_metadata_only_not_truth_or_completeness_authority"
        ),
    }


def canonical_envelope_json(envelope: Mapping[str, Any]) -> str:
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"))


def envelope_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get(PAYLOAD_FIELD)
    if not isinstance(raw, str):
        raw = metadata.get(LEGACY_PAYLOAD_FIELD)
    if not isinstance(raw, str):
        return None
    try:
        envelope = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if (
        not isinstance(envelope, dict)
        or envelope.get("schema_version")
        not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}
    ):
        return None
    return envelope


def validated_envelope_for_candidate(
    candidate: Mapping[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Return a relation envelope only when ID and raw-text hash still bind."""

    metadata = candidate.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    record_id = (
        candidate.get("record_id")
        or candidate.get("id")
        or metadata.get("record_id")
    )
    envelope = envelope_from_metadata(metadata)
    text = str(candidate.get("text") or candidate.get("memory") or "")
    record = envelope.get("record", {}) if envelope else {}
    if (
        record_id
        and envelope
        and str(record.get("id") or "") == str(record_id)
        and record.get("text_sha256")
        == hashlib.sha256(text.encode("utf-8")).hexdigest()
    ):
        return str(record_id), envelope
    return None


def declared_pair_roles(
    source_envelope: Mapping[str, Any],
    target_id: str,
    target_envelope: Mapping[str, Any],
) -> set[str]:
    """Return candidate roles licensed by one explicit declared relation."""

    # Target validation is performed by the caller.  Do not infer pair roles
    # from free lexical candidate tags: only the declared relation type and
    # explicit source->target direction license facet roles.
    del target_envelope
    roles: set[str] = set()
    for relation in source_envelope.get("relations", []):
        if (
            isinstance(relation, Mapping)
            and str(relation.get("target_record_id") or "") == target_id
            and relation.get("status") != "unresolved_ambiguous"
            and relation.get("type") in _DECLARED_RELATION_TYPES
            and relation.get("direction", "source_record_to_target_record")
            == "source_record_to_target_record"
        ):
            relation_type = str(relation["type"])
            assignment = _relation_assignment(relation_type)
            roles.update(assignment["source"])
            roles.update(assignment["target"])
    return roles


def ingestion_payload(
    *,
    text: str,
    user_id: str,
    record_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the scalar vector payload and its decoded relation envelope."""

    declared = dict(metadata or {})
    envelope = build_relation_envelope(
        record_id=record_id,
        text=text,
        metadata=declared,
    )
    payload: dict[str, Any] = {"data": text, "user_id": user_id}
    transport_fields: list[str] = []
    for field in _SOURCE_FIELDS:
        value = declared.get(field)
        if isinstance(value, (str, int, float, bool)) and value != "":
            payload[field] = value
            transport_fields.append(field)
    payload.update(
        {
            "record_id": record_id,
            "relation_envelope_schema": SCHEMA_VERSION,
            "relation_envelope_extractor": EXTRACTOR_VERSION,
            PAYLOAD_FIELD: canonical_envelope_json(envelope),
        }
    )
    transport_fields.extend(
        [
            "record_id",
            "relation_envelope_schema",
            "relation_envelope_extractor",
            PAYLOAD_FIELD,
            TRANSPORT_FIELDS_FIELD,
        ]
    )
    payload[TRANSPORT_FIELDS_FIELD] = json.dumps(
        sorted(set(transport_fields)),
        separators=(",", ":"),
    )
    return payload, envelope


def visible_payload_metadata(
    payload: Mapping[str, Any],
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Return retrieval metadata with exact relation-transport rollback."""

    visible = {
        key: value for key, value in payload.items() if key != "data"
    }
    stored_schema = payload.get("relation_envelope_schema")
    if enabled or stored_schema not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        return visible
    raw_fields = payload.get(TRANSPORT_FIELDS_FIELD)
    try:
        fields = json.loads(raw_fields) if isinstance(raw_fields, str) else []
    except (TypeError, ValueError):
        fields = []
    if not isinstance(fields, list):
        fields = []
    # Version-one records written before the transport manifest still roll
    # back safely; source fields are intentionally included because /store
    # only persisted them as part of this feature path.
    fields.extend(
        [
            "record_id",
            "relation_envelope_schema",
            "relation_envelope_extractor",
            PAYLOAD_FIELD,
            LEGACY_PAYLOAD_FIELD,
            TRANSPORT_FIELDS_FIELD,
            *_SOURCE_FIELDS,
        ]
    )
    for field in fields:
        visible.pop(str(field), None)
    return visible


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_project_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    return posixpath.normpath(raw) if raw else ""


def _parse_timestamp(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def dry_run_backfill_relation_envelopes(
    records: list[Mapping[str, Any]],
    *,
    session_sidecar: Mapping[str, Any] | None = None,
    snapshot_hashes: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Classify legacy records and propose v2 envelopes without writing.

    Existing explicit relations retain the original evidence-bound eligibility
    rules.  A hash-verified session sidecar may additionally license
    provenance-only envelope proposals.  When the existing relation feature is
    enabled, non-overlapping same-project/session records may be reported as
    metadata chronology context.  Chronology never becomes a semantic
    relation, a linked admission, or current-truth evidence.
    """

    snapshots = [dict(record) for record in records]
    sidecar = {
        str(key): str(value)
        for key, value in dict(session_sidecar or {}).items()
    }
    hashes = {
        str(key): str(value)
        for key, value in dict(snapshot_hashes or {}).items()
        if value is not None
    }
    observed_sidecar_sha256 = _canonical_sha256(sidecar)
    expected_sidecar_sha256 = hashes.get("session_sidecar_sha256")
    sidecar_hash_status = (
        "not_provided"
        if not sidecar
        else "verified"
        if expected_sidecar_sha256 == observed_sidecar_sha256
        else "missing_expected_hash"
        if not expected_sidecar_sha256
        else "mismatch"
    )
    snapshot_valid = sidecar_hash_status in {"not_provided", "verified"}
    chronology_enabled = relation_envelopes_enabled(environ)
    by_id = {
        str(record.get("record_id") or record.get("id")): record
        for record in snapshots
        if record.get("record_id") or record.get("id")
    }
    proposals: list[dict[str, Any]] = []
    provenance_proposals: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    chronology_candidates: list[dict[str, Any]] = []
    for record in snapshots:
        record_id = str(record.get("record_id") or record.get("id") or "")
        text = str(record.get("text") or record.get("data") or "")
        metadata_value = record.get("metadata")
        metadata = (
            dict(metadata_value)
            if isinstance(metadata_value, Mapping)
            else {}
        )
        for field in (*_SOURCE_FIELDS, DECLARATIONS_FIELD, LEGACY_DECLARATIONS_FIELD):
            if field in record and field not in metadata:
                metadata[field] = record[field]
        reasons: list[str] = []
        if not record_id:
            reasons.append("missing_stable_record_id")
        if not text:
            reasons.append("missing_raw_text")
        observed_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        declared_text_sha256 = str(
            record.get("text_sha256")
            or metadata.get("text_sha256")
            or ""
        )
        if (
            declared_text_sha256
            and declared_text_sha256 != observed_text_sha256
        ):
            reasons.append("raw_text_sha256_mismatch")

        ingest_hash = str(
            record.get("ingest_hash")
            or metadata.get("ingest_hash")
            or ""
        )
        session_id = str(
            record.get("session_id")
            or metadata.get("session_id")
            or ""
        )
        project_path = _normalized_project_path(
            record.get("project_path")
            or metadata.get("project_path")
        )
        sidecar_record_id = sidecar.get(ingest_hash) if ingest_hash else None
        session_fields_present = bool(
            ingest_hash or session_id or project_path
        )
        source_user = str(
            record.get("user_id")
            or metadata.get("user_id")
            or ""
        )
        provenance_eligible = bool(
            record_id
            and text
            and source_user
            and (
                not declared_text_sha256
                or declared_text_sha256 == observed_text_sha256
            )
        )
        provenance_eligible = bool(
            provenance_eligible
            and sidecar
            and snapshot_valid
            and ingest_hash
            and session_id
            and project_path
            and sidecar_record_id == record_id
        )
        if session_fields_present and sidecar:
            if not snapshot_valid:
                reasons.append("session_sidecar_hash_not_verified")
            elif not ingest_hash:
                reasons.append("missing_ingest_hash")
            elif sidecar_record_id is None:
                reasons.append("ingest_hash_not_in_session_sidecar")
            elif sidecar_record_id != record_id:
                reasons.append("session_sidecar_record_id_mismatch")
            if not session_id:
                reasons.append("missing_session_id")
            if not project_path:
                reasons.append("missing_project_path")
            if not source_user:
                reasons.append("missing_user_namespace")

        enriched_metadata = dict(metadata)
        if provenance_eligible:
            project_sha256 = hashlib.sha256(
                project_path.encode("utf-8")
            ).hexdigest()
            enriched_metadata.update(
                {
                    "source_kind": "session_ingest",
                    "source_pointer": f"session:{session_id}",
                    "source": f"project-sha256:{project_sha256}",
                }
            )
        if not any(enriched_metadata.get(field) for field in _SOURCE_FIELDS):
            reasons.append("missing_source_provenance")
        envelope = (
            build_relation_envelope(
                record_id=record_id,
                text=text,
                metadata=enriched_metadata,
            )
            if record_id and text
            else None
        )
        explicit_relations = [
            relation
            for relation in (envelope or {}).get("relations", [])
            if (
                isinstance(relation, Mapping)
                and relation.get("target_record_id")
                and relation.get("source_span")
                and relation.get("status") != "unresolved_ambiguous"
            )
        ]
        if not explicit_relations:
            reasons.append("no_unambiguous_explicit_cross_record_relation")
        valid_relations: list[Mapping[str, Any]] = []
        for relation in explicit_relations:
            target_id = str(relation.get("target_record_id") or "")
            target = by_id.get(target_id)
            if target is None:
                reasons.append(f"target_not_in_supplied_records:{target_id}")
                continue
            target_metadata = target.get("metadata")
            if not isinstance(target_metadata, Mapping):
                target_metadata = {}
            target_user = str(
                target.get("user_id") or target_metadata.get("user_id") or ""
            )
            if source_user != target_user:
                reasons.append(f"user_namespace_mismatch:{target_id}")
                continue
            valid_relations.append(relation)
        eligible = bool(
            envelope
            and explicit_relations
            and len(valid_relations) == len(explicit_relations)
            and snapshot_valid
            and not {
                reason
                for reason in reasons
                if reason
                not in {
                    "session_sidecar_hash_not_verified",
                    "missing_ingest_hash",
                    "ingest_hash_not_in_session_sidecar",
                    "session_sidecar_record_id_mismatch",
                    "missing_session_id",
                    "missing_project_path",
                }
            }
        )
        ambiguous = bool(
            envelope
            and envelope.get("ambiguity_status") == "unresolved"
        )
        unresolved_target = bool(
            envelope
            and any(
                isinstance(relation, Mapping)
                and not relation.get("target_record_id")
                for relation in envelope.get("relations", [])
            )
        )
        invalid_identity_or_hash = any(
            reason
            in {
                "missing_stable_record_id",
                "missing_raw_text",
                "raw_text_sha256_mismatch",
                "session_sidecar_hash_not_verified",
                "session_sidecar_record_id_mismatch",
                "missing_user_namespace",
            }
            for reason in reasons
        )
        if invalid_identity_or_hash:
            classification = "invalid_identity_or_hash"
        elif ambiguous:
            classification = "unresolved_ambiguous"
        elif eligible:
            classification = "explicit_link"
        elif unresolved_target or explicit_relations:
            classification = "unresolved_no_exact_target"
        elif provenance_eligible:
            classification = "provenance_envelope"
        else:
            classification = "unlinked_source"
        result = {
            "record_id": record_id or None,
            "eligible": eligible,
            "provenance_eligible": provenance_eligible,
            "classification": classification,
            "text_sha256": observed_text_sha256 if text else None,
            "reasons": sorted(set(reasons)),
            "proposed_schema_version": (
                SCHEMA_VERSION
                if eligible or provenance_eligible
                else None
            ),
        }
        results.append(result)
        if eligible and envelope is not None:
            proposals.append(
                {
                    "record_id": record_id,
                    "text_sha256": envelope["record"]["text_sha256"],
                    "classification": "explicit_link",
                    "envelope_sha256": _canonical_sha256(envelope),
                    "linked_admission_eligible": True,
                }
            )
        if provenance_eligible and envelope is not None:
            provenance_proposals.append(
                {
                    "record_id": record_id,
                    "text_sha256": envelope["record"]["text_sha256"],
                    "classification": "provenance_envelope",
                    "envelope_sha256": _canonical_sha256(envelope),
                    "linked_admission_eligible": False,
                    "source_proof": {
                        "ingest_hash": ingest_hash,
                        "session_sidecar_record_id": sidecar_record_id,
                        "session_sidecar_sha256": observed_sidecar_sha256,
                    },
                }
            )
            start_ts = str(
                record.get("start_ts")
                or metadata.get("start_ts")
                or ""
            )
            end_ts = str(
                record.get("end_ts")
                or metadata.get("end_ts")
                or ""
            )
            parsed_start = _parse_timestamp(start_ts)
            parsed_end = _parse_timestamp(end_ts)
            if parsed_start and parsed_end and parsed_start <= parsed_end:
                chronology_candidates.append(
                    {
                        "record_id": record_id,
                        "user_id": source_user,
                        "project_source": enriched_metadata["source"],
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "start": parsed_start,
                        "end": parsed_end,
                    }
                )
            else:
                result["reasons"] = sorted(
                    {
                        *result["reasons"],
                        "invalid_or_missing_time_bounds",
                    }
                )

    chronology_context: list[dict[str, Any]] = []
    chronology_abstentions: list[dict[str, Any]] = []
    if chronology_enabled:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for candidate in chronology_candidates:
            key = (
                candidate["user_id"],
                candidate["project_source"],
            )
            grouped.setdefault(key, []).append(candidate)
        for group in grouped.values():
            ordered = sorted(
                group,
                key=lambda item: (
                    item["start"],
                    item["end"],
                    item["record_id"],
                ),
            )
            for earlier, later in zip(ordered, ordered[1:]):
                identifiers = {
                    "earlier_record_id": earlier["record_id"],
                    "later_record_id": later["record_id"],
                }
                if earlier["end"] >= later["start"]:
                    chronology_abstentions.append(
                        {
                            **identifiers,
                            "classification": "chronology_abstained",
                            "reasons": ["overlap_or_tie"],
                            "linked_admission_eligible": False,
                        }
                    )
                    continue
                chronology_context.append(
                    {
                        **identifiers,
                        "classification": "metadata_chronology_only",
                        "earlier_time": {
                            "field": "end_ts",
                            "value": earlier["end_ts"],
                        },
                        "later_time": {
                            "field": "start_ts",
                            "value": later["start_ts"],
                        },
                        "reasons": [
                            "same_user",
                            "same_project_source",
                            "strictly_non_overlapping_timestamps",
                        ],
                        "semantic_relation": None,
                        "linked_admission_eligible": False,
                        "authority": (
                            "ordering_only_not_semantic_relation_"
                            "current_truth_or_completeness"
                        ),
                    }
                )
    return {
        "schema_version": "fidelis.relation-backfill-dry-run/v2",
        "mode": "dry_run_no_writes",
        "runtime_llm_used": False,
        "feature_env": FEATURE_ENV,
        "feature_enabled": chronology_enabled,
        "terminal_status": (
            "NOT_ESTIMABLE" if not snapshot_valid else "PASS"
        ),
        "inspected_count": len(snapshots),
        "eligible_count": len(proposals),
        "provenance_envelope_count": len(provenance_proposals),
        "chronology_context_count": len(chronology_context),
        "chronology_abstention_count": len(chronology_abstentions),
        "snapshot_verification": {
            "session_sidecar_sha256": {
                "expected": expected_sidecar_sha256,
                "observed": observed_sidecar_sha256 if sidecar else None,
                "status": sidecar_hash_status,
            },
            "source_snapshot_sha256": hashes.get(
                "source_snapshot_sha256"
            ),
        },
        "results": results,
        "proposals": proposals,
        "provenance_proposals": provenance_proposals,
        "chronology_context": chronology_context,
        "chronology_abstentions": chronology_abstentions,
        "authority": (
            "dry_run_provenance_and_chronology_only_"
            "not_truth_completeness_or_live_mutation"
        ),
    }


def expose_linked_raw_evidence(
    *,
    evidence_needs: tuple[str, ...] | list[str],
    candidates: list[dict[str, Any]],
    limit: int = 8,
) -> dict[str, Any]:
    """Expose already-admitted linked raw records without retrieving or reranking.

    A packet is emitted only when a source envelope declares a non-ambiguous
    raw-text link and the linked record is present in ``candidates``.  The
    candidate objects are copied in their existing order and their text is not
    modified.
    """

    needs = tuple(str(value) for value in evidence_needs)
    if not {"prior", "current"}.issubset(needs):
        return {
            "status": "not_needed",
            "required_facets": list(needs),
            "records": [],
        }
    by_id: dict[str, dict[str, Any]] = {}
    envelopes: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        metadata = candidate.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        record_id = (
            candidate.get("record_id")
            or candidate.get("id")
            or metadata.get("record_id")
        )
        if record_id:
            stable_id = str(record_id)
            by_id[stable_id] = candidate
            validated = validated_envelope_for_candidate(candidate)
            if validated:
                envelopes[validated[0]] = validated[1]

    selected_ids: set[str] = set()
    for source_id, candidate in by_id.items():
        envelope = envelopes.get(source_id)
        if not envelope or envelope.get("ambiguity_status") == "unresolved":
            continue
        for target_id in envelope.get("linked_raw_record_ids", []):
            target = str(target_id)
            target_envelope = envelopes.get(target)
            roles = (
                declared_pair_roles(
                    envelope,
                    target,
                    target_envelope,
                )
                if target_envelope
                else set(envelope.get("candidate_roles", []))
            )
            if (
                target in by_id
                and target_envelope
                and {
                    "may_support_prior",
                    "may_support_current",
                }.issubset(roles)
            ):
                selected_ids.update((source_id, target))

    records = [
        dict(candidate)
        for candidate in candidates
        if str(
            candidate.get("record_id")
            or candidate.get("id")
            or (
                candidate.get("metadata", {}).get("record_id")
                if isinstance(candidate.get("metadata"), Mapping)
                else ""
            )
        )
        in selected_ids
    ][: max(0, limit)]
    return {
        "status": (
            "declared_linked_evidence"
            if records
            else "no_declared_relation_in_admitted_candidates"
        ),
        "required_facets": list(needs),
        "records": records,
        "raw_source_text_preserved": all(
            record.get("text") is not None for record in records
        ),
        "authority": (
            "declared_relation_candidates_only_not_truth_or_completeness"
        ),
    }
