"""Offline-first DAgger corrective demonstrations for the human policy.

This module is deliberately outside the production control path.  It consumes
an enriched ``SHADOW_HUMAN`` decision log, selects high-value bot-induced
states, and provides a small manual-labeling queue.  A state is eligible only
when the shadow log contains the complete state and temporal history; summary
records are counted and skipped rather than being reconstructed or assigned a
handcrafted label.

The workflow is:

    python v3/dagger_corrective.py select ...
    python v3/dagger_corrective.py review ...
    python v3/dagger_corrective.py label ...
    python v3/dagger_corrective.py merge ...
    python v3/dagger_corrective.py evaluate ...

``merge`` includes only rows explicitly labeled by a human.  It excludes the
deterministic corrective holdout by default, so the holdout remains unseen by
training.  None of these commands connects to the game, changes a policy
mode, or sends actions to a controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

# ``python v3/dagger_corrective.py`` sets sys.path[0] to the v3 directory,
# while ``python -m v3.dagger_corrective`` starts at the repository root.
# Normalize both entry points so the selector always resolves the exact
# training vectorizer instead of silently treating feature extraction as
# unavailable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from brotato_ai.data.human_demo import _from_blob, _json_bytes


ACTION_NAMES = (
    "IDLE", "UP", "DOWN", "LEFT", "RIGHT",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
)
ACTION_BY_NAME = {name: index for index, name in enumerate(ACTION_NAMES)}
QUEUE_SCHEMA_VERSION = 1
CORRECTION_SCHEMA_VERSION = 1
DEFAULT_MIN_GAP_MS = 500.0
DEFAULT_HOLDOUT_FRACTION = 0.25
HIGH_CONFIDENCE = 0.70
HARD_RISK = 0.65


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _optional_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _action(value: Any) -> int | None:
    if isinstance(value, str):
        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        if normalized in ACTION_BY_NAME:
            return ACTION_BY_NAME[normalized]
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if 0 <= result < len(ACTION_NAMES) else None


def _action_name(value: Any) -> str | None:
    action = _action(value)
    return ACTION_NAMES[action] if action is not None else None


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    errors = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                errors["invalid_json"] += 1
                continue
            if not isinstance(value, dict):
                errors["non_object"] += 1
                continue
            value["_source_line"] = line_number
            records.append(value)
    return records, dict(errors)


def _full_state(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("state", "full_state", "game_state"):
        value = record.get(key)
        if isinstance(value, Mapping) and value:
            return value
    return None


def _temporal_history(record: Mapping[str, Any]) -> list[Any] | None:
    for key in ("temporal_history", "temporal_history_refs", "state_history", "full_state_history"):
        value = record.get(key)
        if isinstance(value, list) and value:
            return value
    return None


def _load_state_sidecar(path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Load the shadow full-state sidecar keyed by ``state_ref``."""

    if path is None:
        return {}, {}
    if not path.is_file():
        return {}, {"sidecar_not_found": 1}
    values: dict[str, dict[str, Any]] = {}
    errors = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                errors["invalid_json"] += 1
                continue
            if not isinstance(value, Mapping) or not value.get("state_ref") or not isinstance(value.get("state"), Mapping):
                errors["invalid_state_record"] += 1
                continue
            values[str(value["state_ref"])] = dict(value)
    return values, dict(errors)


def _resolve_captured_state(
    record: Mapping[str, Any],
    state_lookup: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, list[Any] | None]:
    state = _full_state(record)
    if state is None:
        state_ref = record.get("state_ref")
        sidecar = state_lookup.get(str(state_ref)) if state_ref else None
        if isinstance(sidecar, Mapping) and isinstance(sidecar.get("state"), Mapping):
            state = sidecar["state"]
    raw_history = _temporal_history(record)
    if raw_history is None:
        return state, None
    history: list[Any] = []
    for sample in raw_history:
        if isinstance(sample, Mapping) and sample.get("state_ref") and not isinstance(sample.get("state"), Mapping):
            sidecar = state_lookup.get(str(sample.get("state_ref")))
            if isinstance(sidecar, Mapping) and isinstance(sidecar.get("state"), Mapping):
                expanded = dict(sample)
                expanded["state"] = sidecar["state"]
                history.append(expanded)
                continue
        history.append(sample)
    # The bridge can publish duplicate or delayed timestamps.  The training
    # feature builder consumes chronological samples, so order the captured
    # values by their original timestamp while retaining the exact values for
    # drift auditing.
    history.sort(key=lambda sample: _number(_mapping(sample).get("timestamp_ms")))
    return state, history


def _context(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(record.get("context"))


def _state_hp(record: Mapping[str, Any]) -> float:
    context = _context(record)
    if _optional_number(context.get("health_fraction")) is not None:
        return max(0.0, min(1.0, _number(context.get("health_fraction"))))
    player = _mapping(_full_state(record) or {}).get("player")
    player = _mapping(player)
    return max(0.0, min(1.0, _number(player.get("health"), 0.0) / max(1.0, _number(player.get("max_health"), 1.0))))


def _build_key(record: Mapping[str, Any]) -> str:
    context = _context(record)
    value = context.get("build") or context.get("weapons")
    if isinstance(value, list):
        return ",".join(sorted(str(item) for item in value)) or "unknown"
    return str(value or "unknown")


def _wave(record: Mapping[str, Any]) -> int:
    context = _context(record)
    return max(0, int(_number(context.get("wave"), _number(record.get("wave"), 0))))


def _risk_total(record: Mapping[str, Any], field: str) -> float | None:
    value = _mapping(record.get(field))
    total = _optional_number(value.get("total"))
    return total


def _feature_vector(state: Mapping[str, Any], previous_action: int) -> np.ndarray | None:
    """Build the current vector using the same semantic vectorizer as training."""

    try:
        from v3.combat_policy import SemanticCombatVectorizer

        values = SemanticCombatVectorizer().build(state, int(previous_action))
        values = np.asarray(values, dtype=np.float32).ravel()
        if values.size == 0 or not np.isfinite(values).all():
            return None
        return values
    except Exception:
        return None


def _previous_action(record: Mapping[str, Any]) -> int:
    safety = _mapping(record.get("safety"))
    for value in (
        safety.get("actual_applied_action"),
        record.get("handcrafted_action"),
        record.get("current_action"),
    ):
        parsed = _action(value)
        if parsed is not None:
            return parsed
    return 0


def human_feature_stats(dataset: Path) -> dict[str, Any]:
    """Load mean/std/quantiles for the current stored feature representation."""

    if not dataset.is_file():
        return {"available": False, "error": f"dataset not found: {dataset}"}
    connection = sqlite3.connect(str(dataset))
    try:
        # Bot-induced intervention states are captured during combat.  Compare
        # them with the normal human combat distribution rather than allowing
        # shop/menu vectors to dilute the OOD signal.
        rows = connection.execute(
            "SELECT feature_blob FROM frames WHERE feature_blob IS NOT NULL AND phase='combat' ORDER BY frame_id"
        ).fetchall()
    finally:
        connection.close()
    values: list[np.ndarray] = []
    malformed = 0
    for (blob,) in rows:
        decoded = _from_blob(blob, [])
        try:
            array = np.asarray(decoded, dtype=np.float32).ravel()
        except (TypeError, ValueError):
            malformed += 1
            continue
        if array.size and np.isfinite(array).all():
            values.append(array)
        else:
            malformed += 1
    if not values:
        return {"available": False, "error": "human dataset has no valid feature vectors", "malformed": malformed}
    width = len(values[0])
    values = [value for value in values if len(value) == width]
    matrix = np.asarray(values, dtype=np.float64)
    mean = matrix.mean(axis=0)
    std = np.maximum(matrix.std(axis=0), 1e-5)
    quantiles = np.quantile(matrix, (0.10, 0.50, 0.90), axis=0)
    return {
        "available": True,
        "dataset": str(dataset),
        "phase_filter": "combat",
        "samples": int(len(matrix)),
        "feature_width": int(width),
        "malformed": int(malformed),
        "mean": mean,
        "std": std,
        "quantiles": quantiles,
    }


def _ood_metrics(vector: np.ndarray | None, stats: Mapping[str, Any] | None) -> dict[str, float | None]:
    if vector is None or not _mapping(stats).get("available"):
        return {"rms_z": None, "mean_abs_z": None, "p95_abs_z": None}
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != mean.shape:
        return {"rms_z": None, "mean_abs_z": None, "p95_abs_z": None}
    z = np.abs((value - mean) / std)
    return {
        "rms_z": float(np.sqrt(np.mean(z * z))),
        "mean_abs_z": float(np.mean(z)),
        "p95_abs_z": float(np.percentile(z, 95)),
    }


def feature_distribution_shift(
    human_stats: Mapping[str, Any],
    bot_vectors: Iterable[np.ndarray],
) -> dict[str, Any]:
    """Compare bot-state vectors with normal human-play vectors.

    This is a representation-level diagnostic, not a causal safety claim.  It
    reports standardized mean and quantile gaps plus per-state OOD scores.
    """

    vectors = [np.asarray(value, dtype=np.float64).ravel() for value in bot_vectors]
    vectors = [value for value in vectors if value.size and np.isfinite(value).all()]
    if not _mapping(human_stats).get("available") or not vectors:
        return {
            "available": False,
            "reason": "missing human feature stats or full-state bot vectors",
            "bot_samples": len(vectors),
        }
    human_mean = np.asarray(human_stats["mean"], dtype=np.float64)
    human_std = np.asarray(human_stats["std"], dtype=np.float64)
    human_quantiles = np.asarray(human_stats["quantiles"], dtype=np.float64)
    vectors = [value for value in vectors if value.shape == human_mean.shape]
    if not vectors:
        return {"available": False, "reason": "feature widths do not match", "bot_samples": 0}
    bot = np.asarray(vectors, dtype=np.float64)
    bot_mean = bot.mean(axis=0)
    bot_quantiles = np.quantile(bot, (0.10, 0.50, 0.90), axis=0)
    standardized_mean = np.abs(bot_mean - human_mean) / human_std
    quantile_gap = np.abs(bot_quantiles - human_quantiles) / human_std
    z = (bot - human_mean[None, :]) / human_std[None, :]
    rms = np.sqrt(np.mean(z * z, axis=1))
    return {
        "available": True,
        "human_samples": int(human_stats["samples"]),
        "bot_samples": int(len(bot)),
        "feature_width": int(bot.shape[1]),
        "standardized_mean_gap": float(np.mean(standardized_mean)),
        "feature_mean_fraction_over_2sd": float(np.mean(standardized_mean >= 2.0)),
        "mean_quantile_gap": float(np.mean(quantile_gap)),
        "median_quantile_gap": float(np.median(quantile_gap)),
        "bot_state_rms_z_mean": float(np.mean(rms)),
        "bot_state_rms_z_p90": float(np.percentile(rms, 90)),
        "bot_state_fraction_rms_z_over_2": float(np.mean(rms >= 2.0)),
        "interpretation": "Large standardized or quantile gaps indicate representation-level distribution shift; they do not prove the learned action caused the state.",
    }


def priority_score(
    record: Mapping[str, Any],
    *,
    ood: Mapping[str, Any] | None = None,
) -> tuple[float, list[str]]:
    """Score a shadow state for manual correction, with auditable reasons."""

    proposal = _action(record.get("human_model_proposal"))
    handcrafted = _action(record.get("handcrafted_action"))
    if proposal is None:
        return 0.0, ["no_learned_proposal"]
    score = 0.0
    reasons: list[str] = []
    if handcrafted is not None and proposal != handcrafted:
        score += 3.0
        reasons.append("learned_vs_handcrafted_disagreement")
    confidence = _number(record.get("model_confidence"))
    if confidence >= HIGH_CONFIDENCE:
        score += 2.5
        reasons.append("high_confidence")
    safety = _mapping(record.get("safety"))
    if bool(safety.get("human_would_override")):
        score += 4.0
        reasons.append("counterfactual_safety_override")
    human_risk = _risk_total(record, "human_risk")
    handcrafted_risk = _risk_total(record, "handcrafted_risk")
    delta = _optional_number(record.get("human_minus_handcrafted_risk"))
    if human_risk is not None and human_risk >= HARD_RISK:
        score += 3.0
        reasons.append("learned_proposal_hard_risk")
    if delta is not None and delta >= 0.20:
        score += 2.5
        reasons.append("learned_substantially_riskier")
        if delta >= 0.65:
            score += 1.5
            reasons.append("learned_much_riskier")
    hp = _state_hp(record)
    if hp < 0.35:
        score += 2.5
        reasons.append("low_hp")
        if hp < 0.20:
            score += 1.5
            reasons.append("very_low_hp")
    context = _context(record)
    enemy_count = _number(context.get("enemy_count"))
    projectile_count = _number(context.get("projectile_count"))
    nearest = _optional_number(context.get("nearest_enemy_distance"))
    if enemy_count >= 10 or projectile_count >= 7:
        score += 2.0
        reasons.append("dense_combat")
    if nearest is not None and nearest < 100.0:
        score += 1.5
        reasons.append("bad_positioning_pressure")
    if bool(record.get("dangerous_state")):
        score += 1.5
        reasons.append("dangerous_state")
    if bool(context.get("dead")) or bool(context.get("victory")):
        score += 1.0
        reasons.append("terminal_or_near_terminal_state")
    if ood:
        rms_z = _optional_number(ood.get("rms_z"))
        if rms_z is not None and rms_z >= 2.0:
            score += 2.5
            reasons.append("feature_ood")
        elif rms_z is not None and rms_z >= 1.5:
            score += 1.0
            reasons.append("feature_shifted")
    # Keep this useful in logs even when the risk payload is incomplete.
    if human_risk is None and handcrafted_risk is not None and delta is not None and delta > 0.0:
        reasons.append("risk_delta_without_learned_breakdown")
    return float(score), reasons


def _stable_holdout(queue_id: str, fraction: float) -> str:
    fraction = max(0.0, min(1.0, float(fraction)))
    value = int(hashlib.sha256(queue_id.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
    return "holdout" if value < fraction else "train"


def _candidate_item(
    record: Mapping[str, Any],
    *,
    source_log: Path,
    human_stats: Mapping[str, Any],
    source_line: int,
    state_lookup: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, str | None, np.ndarray | None]:
    proposal = _action(record.get("human_model_proposal"))
    if proposal is None:
        return None, "no_learned_proposal", None
    state, history = _resolve_captured_state(record, state_lookup or {})
    if state is None:
        return None, "missing_full_state", None
    if history is None:
        return None, "missing_temporal_history", None
    previous = _previous_action(record)
    vector = None
    raw_vector = record.get("feature_vector")
    if isinstance(raw_vector, list):
        try:
            vector = np.asarray(raw_vector, dtype=np.float32).ravel()
            if not vector.size or not np.isfinite(vector).all():
                vector = None
        except (TypeError, ValueError):
            vector = None
    if vector is None:
        vector = _feature_vector(state, previous)
    ood = _ood_metrics(vector, human_stats)
    score, reasons = priority_score(record, ood=ood)
    episode = str(record.get("episode", "unknown"))
    tick = int(_number(record.get("tick"), -1))
    timestamp_ms = _number(record.get("timestamp_ms"), 0.0)
    queue_id = f"{source_log.stem}:ep{episode}:tick{tick}:line{source_line}"
    item = {
        "queue_id": queue_id,
        "schema_version": QUEUE_SCHEMA_VERSION,
        "source_log": str(source_log),
        "source_line": int(source_line),
        "source_episode": episode,
        "source_tick": tick,
        "source_timestamp_ms": timestamp_ms,
        "split": _stable_holdout(queue_id, DEFAULT_HOLDOUT_FRACTION),
        "priority_score": score,
        "selection_reasons": reasons,
        "feature_ood": ood,
        "feature_vector": vector.tolist() if vector is not None else None,
        "state": dict(state),
        "temporal_history": history,
        "current_action": _action(record.get("handcrafted_action")),
        "current_action_name": _action_name(record.get("handcrafted_action")),
        "learned_proposal": proposal,
        "learned_proposal_name": _action_name(proposal),
        "model_confidence": _optional_number(record.get("model_confidence")),
        "handcrafted_recommendation": _action(record.get("handcrafted_action")),
        "handcrafted_recommendation_name": _action_name(record.get("handcrafted_action")),
        "risk": {
            "handcrafted": record.get("handcrafted_risk"),
            "learned": record.get("human_risk"),
            "learned_minus_handcrafted": record.get("human_minus_handcrafted_risk"),
        },
        "safety": dict(_mapping(record.get("safety"))),
        "build": _build_key(record),
        "wave": _wave(record),
        "shadow_record": dict(record),
        "status": "unlabeled",
    }
    return item, None, vector


def select_queue(
    shadow_log: Path,
    output: Path,
    *,
    human_dataset: Path | None = None,
    budget: int = 200,
    min_gap_ms: float = DEFAULT_MIN_GAP_MS,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    report_path: Path | None = None,
    state_log: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing queue: {output}")
    records, parse_errors = _read_jsonl(shadow_log)
    state_lookup, state_sidecar_errors = _load_state_sidecar(state_log)
    stats = human_feature_stats(human_dataset) if human_dataset else {"available": False, "reason": "not provided"}
    candidates: list[tuple[dict[str, Any], np.ndarray | None]] = []
    skipped = Counter()
    for record in records:
        item, reason, vector = _candidate_item(
            record,
            source_log=shadow_log,
            human_stats=stats,
            source_line=int(record.get("_source_line", 0)),
            state_lookup=state_lookup,
        )
        if item is None:
            skipped[reason or "not_eligible"] += 1
        else:
            item["split"] = _stable_holdout(item["queue_id"], holdout_fraction)
            candidates.append((item, vector))
    vectors = [vector for _item, vector in candidates if vector is not None]
    shift = feature_distribution_shift(stats, vectors)

    # Prefer high priority, then make the queue cover builds/waves/HP bands.
    buckets: dict[tuple[str, str, str], list[tuple[dict[str, Any], np.ndarray | None]]] = defaultdict(list)
    for item, vector in candidates:
        hp = _state_hp(item["shadow_record"])
        hp_band = "very_low" if hp < 0.20 else "low" if hp < 0.35 else "normal"
        wave_band = "1-4" if item["wave"] <= 4 else "5-7" if item["wave"] <= 7 else "8-12" if item["wave"] <= 12 else "13+"
        buckets[(item["build"], wave_band, hp_band)].append((item, vector))
    for values in buckets.values():
        values.sort(key=lambda pair: (-_number(pair[0].get("priority_score")), pair[0]["source_line"]))
    bucket_order = sorted(buckets, key=lambda key: -_number(buckets[key][0][0].get("priority_score")))
    selected: list[dict[str, Any]] = []
    selected_times: dict[str, list[float]] = defaultdict(list)
    cursors = {key: 0 for key in bucket_order}
    while len(selected) < max(0, int(budget)) and bucket_order:
        progressed = False
        for key in list(bucket_order):
            values = buckets[key]
            while cursors[key] < len(values):
                item, _vector = values[cursors[key]]
                cursors[key] += 1
                source_episode = str(item["source_episode"])
                timestamp = _number(item.get("source_timestamp_ms"))
                if any(abs(timestamp - prior) < max(0.0, float(min_gap_ms)) for prior in selected_times[source_episode]):
                    continue
                selected_times[source_episode].append(timestamp)
                selected.append(item)
                progressed = True
                break
            if cursors[key] >= len(values):
                bucket_order.remove(key)
            if len(selected) >= max(0, int(budget)):
                break
        if not progressed:
            break
    selected.sort(key=lambda item: (-_number(item.get("priority_score")), item["source_line"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n")
    report = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "purpose": "manual DAgger corrective queue; no synthetic labels",
        "shadow_log": str(shadow_log),
        "full_state_sidecar": str(state_log) if state_log else None,
        "full_state_sidecar_records": len(state_lookup),
        "full_state_sidecar_errors": state_sidecar_errors,
        "queue": str(output),
        "human_dataset": str(human_dataset) if human_dataset else None,
        "records_read": len(records),
        "parse_errors": parse_errors,
        "eligible_full_state_records": len(candidates),
        "selected_records": len(selected),
        "budget": int(budget),
        "min_gap_ms": float(min_gap_ms),
        "holdout_fraction": float(holdout_fraction),
        "skipped": dict(skipped),
        "selection": {
            "priority": "disagreement, high confidence, counterfactual safety override, hard risk, risk delta, low HP, dense/bad positioning, feature OOD",
            "stratification_buckets": len(buckets),
            "selected_builds": dict(Counter(item["build"] for item in selected)),
            "selected_wave_bands": dict(Counter("1-4" if item["wave"] <= 4 else "5-7" if item["wave"] <= 7 else "8-12" if item["wave"] <= 12 else "13+" for item in selected)),
            "selected_splits": dict(Counter(item["split"] for item in selected)),
        },
        "feature_distribution_shift": shift,
        "interpretation": "Existing summary-only shadow rows are intentionally not eligible; rerun SHADOW_HUMAN with full-state capture before labeling.",
    }
    if report_path is None:
        report_path = output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def _create_correction_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS queue_items (
            queue_id TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            source_log TEXT NOT NULL,
            source_line INTEGER NOT NULL,
            source_episode TEXT NOT NULL,
            source_tick INTEGER NOT NULL,
            source_timestamp_ms REAL NOT NULL,
            priority_score REAL NOT NULL,
            selection_reasons TEXT NOT NULL,
            feature_ood_json TEXT NOT NULL,
            current_action INTEGER,
            learned_proposal INTEGER NOT NULL,
            model_confidence REAL,
            state_blob BLOB NOT NULL,
            temporal_history_blob BLOB NOT NULL,
            shadow_blob BLOB NOT NULL,
            feature_blob BLOB,
            status TEXT NOT NULL DEFAULT 'unlabeled',
            human_action INTEGER,
            hold_duration_ms REAL,
            labeled_at_ns INTEGER,
            annotator TEXT
        );
        CREATE INDEX IF NOT EXISTS correction_status ON queue_items(status, split);
        CREATE INDEX IF NOT EXISTS correction_context ON queue_items(source_episode, source_tick);
        """
    )


def _load_queue(path: Path) -> list[dict[str, Any]]:
    records, errors = _read_jsonl(path)
    if errors:
        raise ValueError(f"queue contains malformed lines: {errors}")
    return records


def _seed_label_db(database: Path, queue: list[dict[str, Any]]) -> None:
    connection = sqlite3.connect(str(database))
    try:
        _create_correction_schema(connection)
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            ("schema_version", str(CORRECTION_SCHEMA_VERSION)),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            ("label_source", "manual human labels only; no synthetic labels"),
        )
        for item in queue:
            feature = item.get("feature_vector")
            feature_blob = _json_bytes(feature) if isinstance(feature, list) else None
            connection.execute(
                """INSERT OR IGNORE INTO queue_items(
                    queue_id,split,source_log,source_line,source_episode,source_tick,
                    source_timestamp_ms,priority_score,selection_reasons,feature_ood_json,
                    current_action,learned_proposal,model_confidence,state_blob,
                    temporal_history_blob,shadow_blob,feature_blob
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(item["queue_id"]), str(item.get("split", "train")), str(item["source_log"]),
                    int(item.get("source_line", 0)), str(item.get("source_episode", "unknown")),
                    int(item.get("source_tick", -1)), _number(item.get("source_timestamp_ms")),
                    _number(item.get("priority_score")), json.dumps(item.get("selection_reasons", []), separators=(",", ":")),
                    json.dumps(item.get("feature_ood", {}), separators=(",", ":")),
                    _action(item.get("current_action")), int(_action(item.get("learned_proposal")) or 0),
                    _optional_number(item.get("model_confidence")), _json_bytes(item.get("state", {})),
                    _json_bytes(item.get("temporal_history", [])), _json_bytes(item.get("shadow_record", {})),
                    feature_blob,
                ),
            )
        connection.commit()
    finally:
        connection.close()


def _label_rows_file(path: Path) -> dict[str, tuple[int, float | None]]:
    records, errors = _read_jsonl(path)
    if errors:
        raise ValueError(f"label file contains malformed lines: {errors}")
    result: dict[str, tuple[int, float | None]] = {}
    for row in records:
        queue_id = str(row.get("queue_id", ""))
        action = _action(row.get("human_corrective_action", row.get("action")))
        if not queue_id or action is None:
            raise ValueError("each label needs queue_id and a valid action")
        duration = _optional_number(row.get("hold_duration_ms"))
        if duration is not None and duration < 0.0:
            raise ValueError(f"negative hold duration for {queue_id}")
        result[queue_id] = (action, duration)
    return result


def _apply_label(connection: sqlite3.Connection, queue_id: str, action: int, duration: float | None, annotator: str) -> None:
    updated = connection.execute(
        """UPDATE queue_items SET status='labeled', human_action=?, hold_duration_ms=?,
           labeled_at_ns=?, annotator=? WHERE queue_id=? AND status IN ('unlabeled','skipped')""",
        (int(action), duration, time.time_ns(), annotator, queue_id),
    ).rowcount
    if updated != 1:
        raise ValueError(f"queue item is missing or already finalized: {queue_id}")


def label_queue(
    queue_path: Path,
    database: Path,
    *,
    labels_path: Path | None = None,
    annotator: str = "human",
    init_only: bool = False,
) -> dict[str, Any]:
    queue = _load_queue(queue_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    _seed_label_db(database, queue)
    connection = sqlite3.connect(str(database))
    try:
        if init_only and labels_path is not None:
            raise ValueError("--init-only cannot be combined with --labels-jsonl")
        if init_only:
            pass
        elif labels_path is not None:
            labels = _label_rows_file(labels_path)
            for queue_id, (action, duration) in labels.items():
                _apply_label(connection, queue_id, action, duration, annotator)
            connection.commit()
        else:
            rows = connection.execute(
                """SELECT queue_id,split,source_episode,source_tick,source_timestamp_ms,
                          priority_score,current_action,learned_proposal,model_confidence,
                          feature_ood_json,status FROM queue_items
                   WHERE status='unlabeled' ORDER BY priority_score DESC,source_line"""
            ).fetchall()
            for row in rows:
                queue_id, split, episode, tick, timestamp, priority, current, proposal, confidence, ood, _status = row
                print(
                    f"[{queue_id}] split={split} ep={episode} tick={tick} t={timestamp:.1f}ms "
                    f"priority={priority:.2f} current={_action_name(current)} learned={_action_name(proposal)} "
                    f"confidence={confidence if confidence is not None else 'n/a'} ood={ood}",
                    flush=True,
                )
                answer = input("Human action [0-8/name, s=skip, q=quit]: ").strip()
                if answer.lower() == "q":
                    break
                if answer.lower() == "s":
                    connection.execute(
                        "UPDATE queue_items SET status='skipped',labeled_at_ns=?,annotator=? WHERE queue_id=?",
                        (time.time_ns(), annotator, queue_id),
                    )
                    connection.commit()
                    continue
                action = _action(answer)
                if action is None:
                    print("Invalid action; leaving item unlabeled.", flush=True)
                    continue
                duration_text = input("Optional human hold duration in ms [blank=unknown]: ").strip()
                duration = _optional_number(duration_text) if duration_text else None
                if duration is not None and duration < 0.0:
                    print("Negative duration rejected; label not saved.", flush=True)
                    continue
                _apply_label(connection, queue_id, action, duration, annotator)
                connection.commit()
        counts = dict(connection.execute("SELECT status,COUNT(*) FROM queue_items GROUP BY status").fetchall())
        split_counts = connection.execute(
            "SELECT split,status,COUNT(*) FROM queue_items GROUP BY split,status"
        ).fetchall()
    finally:
        connection.close()
    report = {
        "schema_version": CORRECTION_SCHEMA_VERSION,
        "queue": str(queue_path),
        "database": str(database),
        "labels_file": str(labels_path) if labels_path else None,
        "counts_by_status": counts,
        "counts_by_split_status": {f"{split}:{status}": int(count) for split, status, count in split_counts},
        "label_source": "manual human labels only; skipped/unlabeled rows are excluded from training",
    }
    report_path = database.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_queue(queue_path: Path) -> dict[str, Any]:
    """Validate full-state joins before a human is asked to label anything."""

    queue = _load_queue(queue_path)
    errors: list[str] = []
    timestamp_errors = 0
    history_lengths: list[int] = []
    feature_widths: Counter[str] = Counter()
    for item in queue:
        queue_id = str(item.get("queue_id", "unknown"))
        state = item.get("state")
        history = item.get("temporal_history")
        if not isinstance(state, Mapping) or not state:
            errors.append(f"{queue_id}: missing full state")
        if not isinstance(history, list) or not history:
            errors.append(f"{queue_id}: missing temporal history")
            history = []
        history_lengths.append(len(history))
        times: list[float] = []
        for sample in history:
            if not isinstance(sample, Mapping) or not isinstance(sample.get("state"), Mapping):
                errors.append(f"{queue_id}: history sample lacks full state")
                continue
            value = _optional_number(sample.get("timestamp_ms"))
            if value is not None:
                times.append(value)
        current_time = _optional_number(item.get("source_timestamp_ms"))
        if times and any(right < left for left, right in zip(times, times[1:])):
            timestamp_errors += 1
            errors.append(f"{queue_id}: history timestamps are not monotonic")
        if times and current_time is not None and times[-1] > current_time + 1e-6:
            timestamp_errors += 1
            errors.append(f"{queue_id}: history extends beyond current state")
        vector = item.get("feature_vector")
        if vector is None:
            errors.append(f"{queue_id}: missing feature vector")
        else:
            try:
                array = np.asarray(vector, dtype=np.float32).ravel()
                feature_widths[str(len(array))] += 1
                if not len(array) or not np.isfinite(array).all():
                    errors.append(f"{queue_id}: invalid feature vector")
            except (TypeError, ValueError):
                errors.append(f"{queue_id}: malformed feature vector")
        if item.get("status") not in {None, "unlabeled"}:
            errors.append(f"{queue_id}: queue unexpectedly contains a finalized label")
    report = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "queue": str(queue_path),
        "queue_items": len(queue),
        "errors": errors,
        "ok": not errors,
        "full_state_rows": sum(isinstance(item.get("state"), Mapping) and bool(item.get("state")) for item in queue),
        "history_rows": sum(bool(item.get("temporal_history")) for item in queue),
        "history_length": {
            "min": min(history_lengths) if history_lengths else None,
            "max": max(history_lengths) if history_lengths else None,
            "mean": float(np.mean(history_lengths)) if history_lengths else None,
        },
        "feature_widths": dict(feature_widths),
        "timestamp_error_items": timestamp_errors,
        "build_coverage": dict(Counter(str(item.get("build", "unknown")) for item in queue)),
        "wave_coverage": dict(Counter(str(item.get("wave", "unknown")) for item in queue)),
        "split_coverage": dict(Counter(str(item.get("split", "unknown")) for item in queue)),
        "synthetic_labels": 0,
    }
    return report


def _create_training_episode(connection: sqlite3.Connection, episode_id: str, timestamp_ns: int, frame_id: int) -> None:
    connection.execute(
        """INSERT INTO episodes(
            episode_id,session_id,started_ns,ended_ns,outcome,start_phase,end_phase,
            first_frame_id,last_frame_id,first_timestamp_ns,last_timestamp_ns,
            first_bridge_timestamp_ms,last_bridge_timestamp_ms
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            episode_id, "dagger-corrective", timestamp_ns, timestamp_ns,
            "dagger_corrective_labeled", "combat", "combat", frame_id, frame_id,
            timestamp_ns, timestamp_ns, timestamp_ns / 1e6, timestamp_ns / 1e6,
        ),
    )


def merge_corrective_dataset(base_dataset: Path, corrections: Path, output: Path, *, include_holdout: bool = False) -> dict[str, Any]:
    """Merge manual corrective labels into a rich training SQLite dataset."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if not corrections.is_file():
        raise FileNotFoundError(corrections)
    from v3.merge_rich_human_demos import merge_sources

    base_summary = merge_sources([base_dataset], output)
    source = sqlite3.connect(str(corrections))
    destination = sqlite3.connect(str(output))
    inserted = 0
    skipped = Counter()
    try:
        rows = source.execute(
            """SELECT queue_id,split,source_log,source_line,source_episode,source_tick,
                      source_timestamp_ms,priority_score,selection_reasons,feature_ood_json,
                      current_action,learned_proposal,model_confidence,state_blob,
                      temporal_history_blob,shadow_blob,feature_blob,status,human_action,
                      hold_duration_ms,labeled_at_ns,annotator
               FROM queue_items ORDER BY source_log,source_episode,source_timestamp_ms,queue_id"""
        ).fetchall()
        for row in rows:
            (
                queue_id, split, source_log, source_line, source_episode, source_tick,
                timestamp_ms, priority, reasons_json, ood_json, current_action,
                learned_proposal, confidence, state_blob, history_blob, shadow_blob,
                feature_blob, status, human_action, hold_duration_ms, labeled_at_ns, annotator,
            ) = row
            if status != "labeled":
                skipped[f"status_{status}"] += 1
                continue
            if split == "holdout" and not include_holdout:
                skipped["holdout_excluded"] += 1
                continue
            if human_action is None or not 0 <= int(human_action) < len(ACTION_NAMES):
                skipped["invalid_human_action"] += 1
                continue
            state = _from_blob(state_blob, {})
            history = _from_blob(history_blob, [])
            shadow = _from_blob(shadow_blob, {})
            if not isinstance(state, Mapping) or not isinstance(history, list):
                skipped["corrupt_state_or_history"] += 1
                continue
            episode_id = f"dagger:{hashlib.sha256(str(queue_id).encode('utf-8')).hexdigest()[:20]}"
            timestamp_ns = int(round(_number(timestamp_ms) * 1e6))
            previous = _action(current_action)
            if previous is None:
                previous = _previous_action(shadow)
            frame_input = {
                "source": "dagger_manual_corrective",
                "queue_id": str(queue_id),
                "human_corrective_action": int(human_action),
                "human_corrective_action_name": ACTION_NAMES[int(human_action)],
                "hold_duration_ms": hold_duration_ms,
                "annotator": annotator,
                "label_timestamp_ns": labeled_at_ns,
                "raw_available": False,
            }
            controller = {
                "source": "shadow_human_bot_state",
                "queue_id": str(queue_id),
                "shadow_record": shadow,
                "learned_proposal": learned_proposal,
                "model_confidence": confidence,
                "priority_score": priority,
                "selection_reasons": json.loads(reasons_json or "[]"),
                "feature_ood": json.loads(ood_json or "{}"),
                "temporal_history_samples": len(history),
            }
            derived = {
                "source": "dagger_manual_corrective",
                "source_log": source_log,
                "source_line": source_line,
                "source_episode": source_episode,
                "source_tick": source_tick,
                "build": _build_key(shadow),
                "wave": _wave(shadow),
                "human_corrective_action": int(human_action),
                "is_genuine_transition": bool(int(human_action) != int(previous)),
                "temporal_history": history,
            }
            destination.execute(
                "INSERT INTO episodes(episode_id,session_id,started_ns,ended_ns,outcome,start_phase,end_phase,first_frame_id,last_frame_id,first_timestamp_ns,last_timestamp_ns,first_bridge_timestamp_ms,last_bridge_timestamp_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (episode_id, "dagger-corrective", timestamp_ns, timestamp_ns, "dagger_corrective_labeled", "combat", "combat", None, None, timestamp_ns, timestamp_ns, timestamp_ns / 1e6, timestamp_ns / 1e6),
            )
            destination.execute(
                "INSERT INTO frames(episode_id,frame_number,timestamp_ns,wall_time_ns,bridge_timestamp_ms,tick,phase,wave,action,previous_action,action_segment_id,state_blob,input_blob,controller_blob,derived_blob,feature_blob,outcome_blob,reward_blob) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    episode_id, 0, timestamp_ns, int(labeled_at_ns or time.time_ns()), _number(timestamp_ms),
                    int(source_tick), "combat", _wave(shadow), int(human_action), int(previous),
                    f"{episode_id}:segment", state_blob, _json_bytes(frame_input), _json_bytes(controller),
                    _json_bytes(derived), feature_blob, None, None,
                ),
            )
            frame_id = int(destination.execute("SELECT last_insert_rowid()").fetchone()[0])
            destination.execute(
                "UPDATE episodes SET first_frame_id=?,last_frame_id=? WHERE episode_id=?",
                (frame_id, frame_id, episode_id),
            )
            destination.execute(
                "INSERT INTO action_segments(segment_id,episode_id,action,started_ns,ended_ns,duration_ms) VALUES(?,?,?,?,?,?)",
                (f"{episode_id}:segment", episode_id, int(human_action), timestamp_ns, timestamp_ns, hold_duration_ms),
            )
            destination.execute(
                "INSERT INTO labels(frame_id,label,value,annotator,created_ns) VALUES(?,?,?,?,?)",
                (frame_id, "dagger_corrective", json.dumps({"queue_id": queue_id, "split": split}), annotator, labeled_at_ns or time.time_ns()),
            )
            inserted += 1
        destination.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            ("dagger_corrective_merge", json.dumps({
                "correction_database": str(corrections), "inserted_manual_labels": inserted,
                "excluded": dict(skipped), "holdout_included": bool(include_holdout),
                "synthetic_labels": 0, "production_control_changed": False,
            }, separators=(",", ":"))),
        )
        destination.commit()
    finally:
        source.close()
        destination.close()
    return {
        "base": base_summary,
        "output": str(output),
        "inserted_manual_labels": inserted,
        "excluded": dict(skipped),
        "holdout_included": bool(include_holdout),
        "synthetic_labels": 0,
    }


def _model_input_from_item(item: Mapping[str, Any]) -> np.ndarray | None:
    """Rebuild an event input from captured temporal history for evaluation."""

    try:
        from brotato_ai.policy.features import HumanPolicyFeatureBuilder
        from v3.combat_policy import SemanticCombatVectorizer

        state = _mapping(item.get("state"))
        history = item.get("temporal_history")
        if not state or not isinstance(history, list):
            return None
        previous = _action(item.get("current_action"))
        if previous is None:
            previous = _action(_mapping(item.get("safety")).get("actual_applied_action")) or 0
        builder = HumanPolicyFeatureBuilder(vectorizer=SemanticCombatVectorizer())
        observed = False
        for entry in history:
            if isinstance(entry, Mapping) and isinstance(entry.get("state"), Mapping):
                sample = entry["state"]
                timestamp = _optional_number(entry.get("timestamp_ms"))
                held = _action(entry.get("previous_action"))
            elif isinstance(entry, Mapping):
                sample = entry
                timestamp = _optional_number(entry.get("published_at_ms"))
                held = None
            else:
                continue
            builder.observe(sample, held if held is not None else previous, timestamp_ms=timestamp)
            observed = True
        current_timestamp = _optional_number(item.get("source_timestamp_ms"))
        builder.observe(state, previous, timestamp_ms=current_timestamp)
        return builder.build_input(previous) if observed or len(builder) else None
    except Exception:
        return None


def _ece(values: list[tuple[float, bool]]) -> tuple[float | None, list[dict[str, Any]]]:
    bins = [(0.0, 0.35), (0.35, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.000001)]
    total = len(values)
    result = []
    error = 0.0
    for low, high in bins:
        chosen = [(confidence, correct) for confidence, correct in values if low <= confidence < high]
        accuracy = sum(correct for _confidence, correct in chosen) / len(chosen) if chosen else None
        confidence = sum(value for value, _correct in chosen) / len(chosen) if chosen else None
        if chosen and accuracy is not None and confidence is not None:
            error += len(chosen) / max(1, total) * abs(confidence - accuracy)
        result.append({"lower_inclusive": low, "upper_exclusive": high, "samples": len(chosen), "mean_confidence": confidence, "accuracy": accuracy})
    return (error if values else None), result


def evaluate_corrective(database: Path, checkpoint: Path, *, human_dataset: Path | None = None) -> dict[str, Any]:
    """Evaluate a checkpoint on labeled corrective holdout rows."""

    import torch
    from brotato_ai.policy.human_action import EventHumanModel, load_event_checkpoint
    from brotato_ai.control import UnifiedHazardScorer

    connection = sqlite3.connect(str(database))
    try:
        rows = connection.execute(
            """SELECT queue_id,split,source_log,source_line,source_episode,source_tick,
                      source_timestamp_ms,current_action,learned_proposal,model_confidence,
                      state_blob,temporal_history_blob,shadow_blob,feature_blob,status,human_action,
                      hold_duration_ms FROM queue_items WHERE status='labeled' AND split='holdout'
               ORDER BY source_log,source_episode,source_timestamp_ms,queue_id"""
        ).fetchall()
    finally:
        connection.close()
    payload = load_event_checkpoint(checkpoint)
    mean = np.asarray(payload["normalization_mean"], dtype=np.float32)
    std = np.where(np.asarray(payload["normalization_std"], dtype=np.float32) < 1e-5, 1.0, np.asarray(payload["normalization_std"], dtype=np.float32))
    model = EventHumanModel(int(mean.size) + len(ACTION_NAMES), len(ACTION_NAMES))
    model.load_state_dict(dict(payload["model_state"]))
    model.eval()
    threshold = _number(payload.get("change_threshold"), 0.5)
    shield = UnifiedHazardScorer(enabled=True)
    action_values: list[tuple[float, bool]] = []
    change_values: list[tuple[float, bool]] = []
    risk_deltas: list[float] = []
    model_higher = 0
    model_override = 0
    human_higher = 0
    details: list[dict[str, Any]] = []
    bot_vectors: list[np.ndarray] = []
    for row in rows:
        queue_id, split, source_log, source_line, source_episode, source_tick, timestamp_ms, current_action, old_proposal, old_confidence, state_blob, history_blob, shadow_blob, feature_blob, status, human_action, hold_duration_ms = row
        item = {
            "queue_id": queue_id,
            "state": _from_blob(state_blob, {}),
            "temporal_history": _from_blob(history_blob, []),
            "current_action": current_action,
            "source_timestamp_ms": timestamp_ms,
            "safety": _mapping(_from_blob(shadow_blob, {})).get("safety", {}),
        }
        model_input = _model_input_from_item(item)
        if model_input is None or model_input.shape[0] != mean.size + len(ACTION_NAMES):
            details.append({"queue_id": queue_id, "error": "could not reconstruct model input"})
            continue
        normalized = np.concatenate(((model_input[: mean.size] - mean) / std, model_input[mean.size:])).astype(np.float32)
        with torch.no_grad():
            change_logit, action_logit, _duration = model(torch.tensor(normalized[None, :], dtype=torch.float32))
            change_probability = float(torch.sigmoid(change_logit[0]).item())
            probabilities = torch.softmax(action_logit[0], dim=-1).cpu().numpy().astype(np.float64)
        held = _action(current_action) or 0
        scores = probabilities.copy()
        scores[held] = -np.inf
        predicted = int(np.argmax(scores))
        target = _action(human_action)
        if target is None:
            continue
        action_values.append((float(probabilities[predicted]), predicted == target))
        change_values.append((change_probability, bool(target != held)))
        shadow = _from_blob(shadow_blob, {})
        state = _mapping(_from_blob(state_blob, {}))
        vector = _feature_vector(state, held)
        if vector is not None:
            bot_vectors.append(vector)
        handcrafted = _action(_mapping(shadow).get("handcrafted_action"))
        try:
            risks = shield.all_risks(state)
            model_risk = float(risks[predicted].total)
            baseline_risk = float(risks[handcrafted].total) if handcrafted is not None else None
            human_risk = float(risks[target].total)
            if baseline_risk is not None:
                risk_deltas.append(model_risk - baseline_risk)
                model_higher += int(model_risk > baseline_risk + 1e-6)
                human_higher += int(human_risk > baseline_risk + 1e-6)
            model_override += int(bool(shield.choose(risks, predicted, previous_action=held).overridden))
        except Exception:
            model_risk = baseline_risk = human_risk = None
        details.append({
            "queue_id": queue_id, "source_episode": source_episode, "source_tick": source_tick,
            "predicted_action": predicted, "predicted_action_name": ACTION_NAMES[predicted],
            "human_action": target, "human_action_name": ACTION_NAMES[target],
            "held_action": held, "confidence": float(probabilities[predicted]),
            "correct": bool(predicted == target), "change_probability": change_probability,
            "target_is_change": bool(target != held), "model_risk": model_risk,
            "baseline_risk": baseline_risk, "human_label_risk": human_risk,
        })
    ece, bins = _ece(action_values)
    change_ece, change_bins = _ece(change_values)
    stats = human_feature_stats(human_dataset) if human_dataset else {"available": False, "reason": "not provided"}
    report = {
        "schema_version": CORRECTION_SCHEMA_VERSION,
        "database": str(database),
        "checkpoint": str(checkpoint),
        "split": "labeled corrective holdout only; holdout rows are not expected in the retraining dataset",
        "samples": len(details),
        "action_accuracy": (sum(item.get("correct", False) for item in details) / len(details)) if details else None,
        "action_confidence_ece": ece,
        "action_confidence_bins": bins,
        "change_f1": _change_f1(change_values, threshold),
        "change_confidence_ece": change_ece,
        "change_confidence_bins": change_bins,
        "higher_risk_proposal_rate": model_higher / len(risk_deltas) if risk_deltas else None,
        "counterfactual_safety_override_rate": model_override / len(details) if details else None,
        "mean_model_minus_baseline_risk": float(np.mean(risk_deltas)) if risk_deltas else None,
        "human_label_higher_risk_rate": human_higher / len(risk_deltas) if risk_deltas else None,
        "feature_distribution_shift": feature_distribution_shift(stats, bot_vectors),
        "representative_examples": details[:100],
        "autoregressive_accuracy": None,
        "autoregressive_note": "Corrective holdout rows are independent bot-state interventions; autoregression across them would invent temporal continuity and is therefore not reported here.",
    }
    return report


def _change_f1(values: list[tuple[float, bool]], threshold: float) -> dict[str, Any]:
    actual = [truth for _probability, truth in values]
    predicted = [_probability >= threshold for _probability, _truth in values]
    tp = sum(a and p for a, p in zip(actual, predicted))
    fp = sum((not a) and p for a, p in zip(actual, predicted))
    fn = sum(a and (not p) for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"samples": len(values), "threshold": threshold, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "precision": precision, "recall": recall}


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline DAgger corrective queue for the human policy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select", help="select full-state bot-induced records for manual labeling")
    select.add_argument("--shadow-log", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--human-dataset", type=Path)
    select.add_argument("--state-log", type=Path, help="full-state sidecar emitted by shadow_eval.py --capture-full-state")
    select.add_argument("--budget", type=int, default=200)
    select.add_argument("--min-gap-ms", type=float, default=DEFAULT_MIN_GAP_MS)
    select.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    select.add_argument("--report", type=Path)

    label = subparsers.add_parser("label", help="apply manual labels interactively or from a human-authored JSONL")
    label.add_argument("--queue", type=Path, required=True)
    label.add_argument("--database", type=Path, required=True)
    label.add_argument("--labels-jsonl", type=Path)
    label.add_argument("--annotator", default="human")
    label.add_argument("--init-only", action="store_true", help="create/resume the SQLite queue without asking for labels")

    review = subparsers.add_parser("review", help="write an offline visual tactical-map reviewer")
    review.add_argument("--queue", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)

    merge = subparsers.add_parser("merge", help="merge labeled train corrections into a rich training SQLite")
    merge.add_argument("--base-dataset", type=Path, required=True)
    merge.add_argument("--corrections", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--include-holdout", action="store_true", help="for diagnostics only; do not use for retraining")

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint on labeled corrective holdout rows")
    evaluate.add_argument("--corrections", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--human-dataset", type=Path)
    evaluate.add_argument("--report", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="validate dereferenced full-state queue integrity")
    validate.add_argument("--queue", type=Path, required=True)
    validate.add_argument("--report", type=Path)

    args = parser.parse_args()
    if args.command == "select":
        report = select_queue(
            args.shadow_log, args.output, human_dataset=args.human_dataset, budget=args.budget,
            min_gap_ms=args.min_gap_ms, holdout_fraction=args.holdout_fraction, report_path=args.report,
            state_log=args.state_log,
        )
    elif args.command == "label":
        report = label_queue(args.queue, args.database, labels_path=args.labels_jsonl, annotator=args.annotator, init_only=args.init_only)
    elif args.command == "review":
        from v3.dagger_review import render_review_html

        report = render_review_html(args.queue, args.output)
    elif args.command == "merge":
        report = merge_corrective_dataset(args.base_dataset, args.corrections, args.output, include_holdout=args.include_holdout)
    elif args.command == "validate":
        report = validate_queue(args.queue)
        report_path = args.report or args.queue.with_suffix(".validation.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        report = evaluate_corrective(args.corrections, args.checkpoint, human_dataset=args.human_dataset)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
