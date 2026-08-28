"""Leakage-safe diagnostics for the SQLite human behavior-cloning baseline.

This is an offline evaluation tool.  It does not modify the production
controller or checkpoint.  The current checkpoint was trained by
``v3.train_human_demo_bc``; this evaluator reproduces that script's seeded
episode split and reports persistence-aware transition metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from brotato_ai.data.human_demo import _from_blob


ACTION_COUNT = 9
ACTION_NAMES = (
    "IDLE", "UP", "DOWN", "LEFT", "RIGHT",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
)
ACTION_VECTORS = np.asarray(
    (
        (0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-1.0, 0.0),
        (1.0, 0.0), (-math.sqrt(0.5), -math.sqrt(0.5)),
        (math.sqrt(0.5), -math.sqrt(0.5)),
        (-math.sqrt(0.5), math.sqrt(0.5)),
        (math.sqrt(0.5), math.sqrt(0.5)),
    ),
    dtype=np.float64,
)
PREVIOUS_ACTION_SLICE = slice(16, 25)


@dataclass
class Row:
    frame_id: int
    episode_id: str
    frame_number: int
    timestamp_ns: int
    phase: str
    wave: int | None
    action: int
    previous_action: int
    features: np.ndarray
    state: dict[str, Any]
    controller: dict[str, Any]
    derived: dict[str, Any]
    outcomes: dict[str, Any]


def _items(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def load_rows(path: Path) -> list[Row]:
    connection = sqlite3.connect(str(path))
    raw_rows = connection.execute(
        """
        SELECT frame_id,episode_id,frame_number,timestamp_ns,phase,wave,action,
               previous_action,feature_blob,state_blob,controller_blob,
               derived_blob,outcome_blob
        FROM frames
        WHERE feature_blob IS NOT NULL
        ORDER BY episode_id,frame_number,frame_id
        """
    ).fetchall()
    connection.close()
    rows: list[Row] = []
    for raw in raw_rows:
        features = _from_blob(raw[8], [])
        if not isinstance(features, list) or len(features) == 0:
            continue
        action = int(raw[6])
        previous_action = int(raw[7])
        if not 0 <= action < ACTION_COUNT or not 0 <= previous_action < ACTION_COUNT:
            continue
        rows.append(Row(
            frame_id=int(raw[0]), episode_id=str(raw[1]), frame_number=int(raw[2]),
            timestamp_ns=int(raw[3]), phase=str(raw[4]),
            wave=None if raw[5] is None else int(raw[5]), action=action,
            previous_action=previous_action,
            features=np.asarray(features, dtype=np.float32),
            state=_from_blob(raw[9], {}), controller=_from_blob(raw[10], {}),
            derived=_from_blob(raw[11], {}), outcomes=_from_blob(raw[12], {}),
        ))
    return rows


def grouped_split(rows: list[Row], seed: int) -> tuple[set[str], set[str], list[str]]:
    episodes = sorted({row.episode_id for row in rows})
    shuffled = list(episodes)
    np.random.default_rng(seed).shuffle(shuffled)
    split = max(1, int(len(shuffled) * 0.8)) if len(shuffled) > 1 else 1
    train = set(shuffled[:split])
    valid = set(shuffled[split:])
    if not valid:
        valid = set(train)
    return train, valid, shuffled


def load_model(path: Path, width: int):
    import torch
    from torch import nn

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"unsupported BC checkpoint: {path}")
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise RuntimeError("expected the v3.train_human_demo_bc checkpoint format")
    model = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, ACTION_COUNT))
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def logits_for(model, features: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return model(torch.tensor(features, dtype=torch.float32)).cpu().numpy()


def train_diagnostic_model(features: np.ndarray, labels: np.ndarray, seed: int, epochs: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(features.shape[1], 128), nn.ReLU(), nn.Linear(128, ACTION_COUNT))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    x_train = torch.tensor(features, dtype=torch.float32)
    y_train = torch.tensor(labels, dtype=torch.long)
    model.train()
    for _ in range(max(1, epochs)):
        order = torch.randperm(len(x_train))
        for start in range(0, len(order), 256):
            batch = order[start:start + 256]
            loss = nn.functional.cross_entropy(model(x_train[batch]), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def metric_summary(true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    matrix = np.zeros((ACTION_COUNT, ACTION_COUNT), dtype=np.int64)
    for actual, guess in zip(true, pred):
        matrix[int(actual), int(guess)] += 1
    support = matrix.sum(axis=1)
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    per_action: dict[str, Any] = {}
    for action in range(ACTION_COUNT):
        tp = float(matrix[action, action])
        predicted = float(matrix[:, action].sum())
        actual = float(support[action])
        p = tp / predicted if predicted else 0.0
        r = tp / actual if actual else 0.0
        score = 2 * p * r / (p + r) if p + r else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(score)
        per_action[ACTION_NAMES[action]] = {
            "support": int(actual), "precision": p, "recall": r, "f1": score,
        }
    present = support > 0
    return {
        "frames": int(len(true)),
        "accuracy": float(np.mean(true == pred)) if len(true) else 0.0,
        "balanced_accuracy": float(np.mean(np.asarray(recall)[present])) if present.any() else 0.0,
        "macro_f1": float(np.mean(f1)),
        "per_action": per_action,
        "confusion_matrix": matrix.tolist(),
        "labels": list(ACTION_NAMES),
        "present_class_count": int(present.sum()),
    }


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def binary_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum(true & pred))
    fp = int(np.sum(~true & pred))
    fn = int(np.sum(true & ~pred))
    tn = int(np.sum(~true & ~pred))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "positive": int(true.sum()), "negative": int((~true).sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def ranking_metrics(true: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    """Small dependency-free ROC-AUC and average-precision implementation."""

    true = np.asarray(true, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(true.sum())
    negatives = int((~true).sum())
    if not positives or not negatives:
        return {"roc_auc": None, "average_precision": None}
    order = np.argsort(-scores, kind="mergesort")
    labels = true[order].astype(np.int64)
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tpr = np.concatenate(([0.0], tp / positives, [1.0]))
    fpr = np.concatenate(([0.0], fp / negatives, [1.0]))
    roc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / positives
    previous_recall = np.concatenate(([0.0], recall[:-1]))
    ap = float(np.sum((recall - previous_recall) * precision))
    return {"roc_auc": roc, "average_precision": ap}


def action_change_metrics(rows: list[Row], pred: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    previous = np.asarray([row.previous_action for row in rows], dtype=np.int64)
    actual = np.asarray([row.action for row in rows], dtype=np.int64)
    changed = actual != previous
    predicted_changed = pred != previous
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    scores = 1.0 - probabilities[np.arange(len(rows)), previous]
    return {
        "change_detection": {
            "teacher_forced": binary_metrics(changed, predicted_changed),
            **ranking_metrics(changed, scores),
            "definition": "actual action differs from stored previous_action; predicted change means argmax differs from previous_action",
        },
        "new_action_selection": {
            "transition_events": int(changed.sum()),
            "accuracy_at_real_change": safe_div(float(np.sum(pred[changed] == actual[changed])), float(changed.sum())),
            "bc_changed_on_real_change": safe_div(float(np.sum(predicted_changed[changed])), float(changed.sum())),
            "correct_new_action_given_bc_changed": safe_div(
                float(np.sum((pred[changed] == actual[changed]) & predicted_changed[changed])),
                float(np.sum(predicted_changed[changed])),
            ),
        },
    }


def transition_metrics(rows: list[Row], pred: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    by_episode: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_episode[row.episode_id].append(index)
    position_by_index = {
        index: position
        for episode_indices in by_episode.values()
        for position, index in enumerate(episode_indices)
    }
    monotonic_times = {
        episode: np.maximum.accumulate(np.asarray(
            [rows[index].timestamp_ns for index in episode_indices], dtype=np.int64
        ))
        for episode, episode_indices in by_episode.items()
    }
    transition_indices = [
        index for index, row in enumerate(rows) if row.action != row.previous_action
    ]
    transitions = np.asarray(transition_indices, dtype=np.int64)
    actual = np.asarray([row.action for row in rows], dtype=np.int64)
    top3 = np.argsort(-logits, axis=1)[:, :3]
    result: dict[str, Any] = {
        "transition_events": int(len(transitions)),
        "strict_accuracy": safe_div(float(np.sum(pred[transitions] == actual[transitions])), float(len(transitions))),
        "top3_accuracy": safe_div(float(np.sum(np.any(top3[transitions] == actual[transitions, None], axis=1))), float(len(transitions))),
        "windows": {},
    }
    for radius in (1, 2, 3):
        hits = 0
        for index in transition_indices:
            target = actual[index]
            episode_indices = by_episode[rows[index].episode_id]
            position = position_by_index[index]
            nearby = episode_indices[max(0, position - radius):position + radius + 1]
            hits += int(np.any(pred[nearby] == target))
        result["windows"][f"plus_minus_{radius}_frames"] = {
            "events": len(transitions), "accuracy": safe_div(hits, len(transitions)),
        }
    hits_100 = 0
    for index in transition_indices:
        target = actual[index]
        timestamp = rows[index].timestamp_ns
        episode = rows[index].episode_id
        episode_indices = by_episode[episode]
        times = monotonic_times[episode]
        left = int(np.searchsorted(times, timestamp - 100_000_000, side="left"))
        right = int(np.searchsorted(times, timestamp + 100_000_000 + 1, side="left"))
        nearby = episode_indices[left:right]
        hits_100 += int(np.any(pred[nearby] == target))
    result["windows"]["plus_minus_100ms"] = {
        "events": len(transitions), "accuracy": safe_div(hits_100, len(transitions)),
    }
    true_vectors = ACTION_VECTORS[actual[transitions]] if len(transitions) else np.empty((0, 2))
    pred_vectors = ACTION_VECTORS[pred[transitions]] if len(transitions) else np.empty((0, 2))
    movement = (np.linalg.norm(true_vectors, axis=1) > 0) & (np.linalg.norm(pred_vectors, axis=1) > 0)
    angles: list[float] = []
    for left, right in zip(true_vectors[movement], pred_vectors[movement]):
        cosine = np.clip(float(np.dot(left, right)), -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cosine))))
    result["directional_error_degrees"] = {
        "movement_transition_events": int(movement.sum()),
        "mean": float(np.mean(angles)) if angles else None,
        "median": float(np.median(angles)) if angles else None,
        "p90": float(np.percentile(angles, 90)) if angles else None,
        "predicted_idle_rate_at_movement_transition": safe_div(
            float(np.sum(pred_vectors[np.linalg.norm(true_vectors, axis=1) > 0][:, 0] == 0.0)),
            float(np.sum(np.linalg.norm(true_vectors, axis=1) > 0)),
        ) if len(transitions) else 0.0,
    }
    return result


def sequence_intervals(rows: list[Row]) -> dict[str, float]:
    values: dict[str, float] = {}
    for episode in sorted({row.episode_id for row in rows}):
        times = [row.timestamp_ns for row in rows if row.episode_id == episode]
        deltas = [b - a for a, b in zip(times, times[1:]) if 0 < b - a < 2_000_000_000]
        values[episode] = float(np.median(deltas) / 1e6) if deltas else 0.0
    return values


def runs(rows: list[Row], actions: np.ndarray, intervals_ms: Mapping[str, float]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    start = 0
    while start < len(rows):
        end = start + 1
        while end < len(rows) and rows[end].episode_id == rows[start].episode_id and actions[end] == actions[start]:
            end += 1
        cadence = intervals_ms.get(rows[start].episode_id, 0.0)
        duration = max(0.0, (end - start) * cadence)
        result.append({
            "episode_id": rows[start].episode_id, "action": int(actions[start]),
            "action_name": ACTION_NAMES[int(actions[start])], "frames": end - start,
            "duration_ms": duration,
        })
        start = end
    return result


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)), "mean": float(np.mean(array)),
        "median": float(np.median(array)), "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def hold_summary(rows: list[Row], actions: np.ndarray, intervals_ms: Mapping[str, float]) -> dict[str, Any]:
    segments = runs(rows, actions, intervals_ms)
    by_action: dict[str, Any] = {}
    for action in range(ACTION_COUNT):
        by_action[ACTION_NAMES[action]] = distribution(
            segment["duration_ms"] for segment in segments if segment["action"] == action
        )
    changes = 0
    reversals = 0
    total_ms = sum(segment["duration_ms"] for segment in segments)
    for left, right in zip(segments, segments[1:]):
        if left["episode_id"] != right["episode_id"]:
            continue
        changes += 1
        old = ACTION_VECTORS[left["action"]]
        new = ACTION_VECTORS[right["action"]]
        reversals += int(np.dot(old, new) <= -0.5)
    return {
        "overall": distribution(segment["duration_ms"] for segment in segments),
        "by_action": by_action,
        "action_changes": changes,
        "action_changes_per_second": safe_div(changes, total_ms / 1000.0),
        "rapid_reversals": reversals,
        "segments": len(segments),
    }


def clear_previous_action(features: np.ndarray, action: int) -> np.ndarray:
    output = np.asarray(features, dtype=np.float32).copy()
    output[PREVIOUS_ACTION_SLICE] = 0.0
    if 0 <= int(action) < ACTION_COUNT:
        output[16 + int(action)] = 1.0
    return output


def mask_previous_action(features: np.ndarray) -> np.ndarray:
    output = np.asarray(features, dtype=np.float32).copy()
    output[PREVIOUS_ACTION_SLICE] = 0.0
    return output


def context_key(row: Row) -> dict[str, str]:
    derived = row.derived
    state = row.state
    hp = _mapping(state.get("player")).get("health")
    max_hp = max(1.0, _number(_mapping(state.get("player")).get("max_health"), 1.0))
    hp_fraction = _number(hp) / max_hp
    enemy_count = int(_number(derived.get("enemy_count"), len(_items(state.get("enemies")))))
    projectile_count = int(_number(derived.get("projectile_count"), len(_items(state.get("projectiles")))))
    nearest = _number(derived.get("nearest_enemy_distance"), float("inf"))
    build = _mapping(state.get("build"))
    weapons = _items(_mapping(state.get("combat")).get("weapons"))
    weapon_tokens = [str(item.get("id") or item.get("type") or item.get("name") or "") for item in weapons]
    build_token = json.dumps(build, sort_keys=True, separators=(",", ":")) if build else ",".join(sorted(weapon_tokens))
    if not build_token:
        build_token = "unknown"
    return {
        "build": build_token[:240],
        "enemy_density": "0" if enemy_count == 0 else "1-4" if enemy_count < 5 else "5-9" if enemy_count < 10 else "10+",
        "projectile_density": "0" if projectile_count == 0 else "1-4" if projectile_count < 5 else "5-9" if projectile_count < 10 else "10+",
        "nearest_enemy_distance": "<100" if nearest < 100 else "100-170" if nearest < 170 else "170-300" if nearest < 300 else ">=300" if math.isfinite(nearest) else "none",
        "hazard": "actionable" if bool(derived.get("hazard_actionable")) else "not_actionable",
        "ranged_spacing": "inside" if bool(derived.get("inside_desired_ranged_spacing")) else "outside",
        "health": "low" if hp_fraction < 0.5 else "high",
    }


def context_metrics(rows: list[Row], pred: np.ndarray, persistence: np.ndarray) -> dict[str, Any]:
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        for dimension, value in context_key(row).items():
            groups[dimension][value].append(index)
    output: dict[str, Any] = {}
    for dimension, buckets in groups.items():
        output[dimension] = {}
        for bucket, indices in sorted(buckets.items()):
            selected = np.asarray(indices, dtype=np.int64)
            output[dimension][bucket] = {
                "frames": len(indices),
                "bc_accuracy": float(np.mean(pred[selected] == np.asarray([rows[i].action for i in indices]))),
                "persistence_accuracy": float(np.mean(persistence[selected] == np.asarray([rows[i].action for i in indices]))),
            }
    return output


def safety_disagreement(rows: list[Row], pred: np.ndarray, limit: int = 4) -> dict[str, Any]:
    counts = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        human = row.action
        bc = int(pred[index])
        safety_value = row.controller.get("safest_action")
        safety = None if safety_value is None else int(safety_value)
        if safety is None:
            category = "safety_unavailable"
        elif bc == human and safety != human:
            category = "bc_equals_human_safety_differs"
        elif safety == human and bc != human:
            category = "safety_equals_human_bc_differs"
        elif bc == safety and bc != human:
            category = "bc_equals_safety_not_human"
        elif len({bc, safety, human}) == 3:
            category = "all_three_differ"
        else:
            category = "all_three_agree"
        counts[category] += 1
        if len(examples[category]) < limit:
            outcomes_250 = _mapping(row.outcomes.get("250"))
            outcomes_1000 = _mapping(row.outcomes.get("1000"))
            risks = _mapping(row.controller.get("candidate_risks"))
            examples[category].append({
                "frame_id": row.frame_id, "episode_id": row.episode_id,
                "frame_number": row.frame_number, "wave": row.wave,
                "human_action": human, "human_action_name": ACTION_NAMES[human],
                "bc_action": bc, "bc_action_name": ACTION_NAMES[bc],
                "safety_action": safety,
                "hazard_actionable": bool(row.derived.get("hazard_actionable")),
                "nearest_projectile_tti_ms": row.derived.get("nearest_projectile_tti_ms"),
                "nearest_enemy_distance": row.derived.get("nearest_enemy_distance"),
                "projectile_count": row.derived.get("projectile_count"),
                "observed_health_loss_250ms": outcomes_250.get("health_loss"),
                "observed_health_loss_1000ms": outcomes_1000.get("health_loss"),
                "observed_death_1000ms": outcomes_1000.get("dead"),
                "action_risks": risks,
            })
    total = max(1, len(rows))
    return {
        "counts": dict(counts),
        "rates": {key: value / total for key, value in counts.items()},
        "examples": dict(examples),
        "outcome_note": "outcomes are observed after the human action; they are not counterfactual outcomes for BC or safety actions",
    }


def autoregressive(model, rows: list[Row]) -> tuple[np.ndarray, np.ndarray]:
    logits: list[np.ndarray] = []
    predictions: list[int] = []
    previous_by_episode: dict[str, int] = {}
    for row in rows:
        previous = previous_by_episode.get(row.episode_id, row.previous_action)
        features = clear_previous_action(row.features, previous)
        current_logits = logits_for(model, features[None, :])[0]
        prediction = int(np.argmax(current_logits))
        logits.append(current_logits)
        predictions.append(prediction)
        previous_by_episode[row.episode_id] = prediction
    return np.asarray(predictions, dtype=np.int64), np.asarray(logits, dtype=np.float32)


def autoregressive_summary(rows: list[Row], pred: np.ndarray, teacher_pred: np.ndarray,
                           intervals_ms: Mapping[str, float]) -> dict[str, Any]:
    first_divergence: dict[str, Any] = {}
    prefixes: dict[str, dict[str, float]] = {}
    for episode in sorted({row.episode_id for row in rows}):
        indices = [i for i, row in enumerate(rows) if row.episode_id == episode]
        divergence = [position for position, index in enumerate(indices) if pred[index] != teacher_pred[index]]
        first = divergence[0] if divergence else None
        first_divergence[episode] = {
            "frames": len(indices),
            "first_divergence_frame_offset": first,
            "first_divergence_ms": None if first is None else first * intervals_ms.get(episode, 0.0),
            "divergence_rate": safe_div(len(divergence), len(indices)),
        }
        prefixes[episode] = {
            f"first_{int(fraction * 100)}pct_agreement": float(np.mean(
                pred[indices[:max(1, int(len(indices) * fraction))]] ==
                teacher_pred[indices[:max(1, int(len(indices) * fraction))]]
            )) for fraction in (0.1, 0.25, 0.5, 0.75, 1.0)
        }
    return {"per_episode": first_divergence, "prefix_teacher_agreement": prefixes}


def split_metrics(rows: list[Row], pred: np.ndarray, logits: np.ndarray,
                  majority_action: int, *, label: str) -> dict[str, Any]:
    true = np.asarray([row.action for row in rows], dtype=np.int64)
    persistence = np.asarray([row.previous_action for row in rows], dtype=np.int64)
    majority = np.full(len(rows), majority_action, dtype=np.int64)
    nontransition = true == persistence
    transition = ~nontransition
    table: dict[str, Any] = {}
    for name, guess in (("majority", majority), ("previous_action", persistence), ("bc", pred)):
        table[name] = {
            "all_frames": safe_div(float(np.sum(guess == true)), float(len(true))),
            "non_transition_frames": safe_div(float(np.sum(guess[nontransition] == true[nontransition])), float(nontransition.sum())),
            "transition_frames": safe_div(float(np.sum(guess[transition] == true[transition])), float(transition.sum())),
        }
    result = {
        "label": label, "frames": len(rows),
        "action_distribution": {
            ACTION_NAMES[action]: int(np.sum(true == action)) for action in range(ACTION_COUNT)
        },
        "majority_action": majority_action,
        "majority_action_name": ACTION_NAMES[majority_action],
        "baselines_and_bc": table,
        "bc_classification": metric_summary(true, pred),
        "previous_action_classification": metric_summary(true, persistence),
        "majority_classification": metric_summary(true, majority),
        "transition_metrics": transition_metrics(rows, pred, logits),
        "change_metrics": action_change_metrics(rows, pred, logits),
        "transition_count": int(transition.sum()),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a human BC checkpoint against persistence-aware baselines")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ablation-epochs", type=int, default=12)
    args = parser.parse_args()

    rows = load_rows(args.dataset)
    if not rows:
        raise SystemExit("dataset has no valid feature rows")
    width = len(rows[0].features)
    if any(len(row.features) != width for row in rows):
        raise SystemExit("feature width is inconsistent")
    train_ids, valid_ids, shuffled_ids = grouped_split(rows, args.seed)
    train_rows = [row for row in rows if row.episode_id in train_ids]
    valid_rows = [row for row in rows if row.episode_id in valid_ids]
    model, checkpoint = load_model(args.checkpoint, width)

    valid_features = np.asarray([row.features for row in valid_rows], dtype=np.float32)
    valid_true = np.asarray([row.action for row in valid_rows], dtype=np.int64)
    valid_logits = logits_for(model, valid_features)
    valid_pred = np.argmax(valid_logits, axis=1).astype(np.int64)
    train_true = np.asarray([row.action for row in train_rows], dtype=np.int64)
    majority_action = int(Counter(train_true.tolist()).most_common(1)[0][0])
    intervals = sequence_intervals(rows)

    combat_rows = [row for row in valid_rows if row.phase == "combat"]
    combat_indices = np.asarray([index for index, row in enumerate(valid_rows) if row.phase == "combat"], dtype=np.int64)
    combat_metrics = split_metrics(
        combat_rows, valid_pred[combat_indices], valid_logits[combat_indices], majority_action, label="held_out_combat"
    ) if len(combat_rows) else {}

    masked_train = np.asarray([mask_previous_action(row.features) for row in train_rows], dtype=np.float32)
    masked_valid = np.asarray([mask_previous_action(row.features) for row in valid_rows], dtype=np.float32)
    no_previous_model = train_diagnostic_model(
        masked_train, train_true, args.seed, args.ablation_epochs
    )
    no_previous_logits = logits_for(no_previous_model, masked_valid)
    no_previous_pred = np.argmax(no_previous_logits, axis=1).astype(np.int64)

    heldout = split_metrics(valid_rows, valid_pred, valid_logits, majority_action, label="held_out_episode")
    heldout["bc_without_previous_action_retrained"] = metric_summary(valid_true, no_previous_pred)
    heldout["bc_without_previous_action_zeroed_at_inference"] = metric_summary(
        valid_true, np.argmax(logits_for(model, masked_valid), axis=1).astype(np.int64)
    )
    heldout["hold_durations"] = {
        "human": hold_summary(valid_rows, valid_true, intervals),
        "bc_teacher_forced": hold_summary(valid_rows, valid_pred, intervals),
        "previous_action_baseline": hold_summary(valid_rows, valid_true, intervals),
        "bc_without_previous_action": hold_summary(valid_rows, no_previous_pred, intervals),
    }
    heldout["context_metrics"] = context_metrics(valid_rows, valid_pred, np.asarray([row.previous_action for row in valid_rows]))
    heldout["safety_disagreement"] = safety_disagreement(valid_rows, valid_pred)

    ar_pred, ar_logits = autoregressive(model, valid_rows)
    heldout["autoregressive"] = {
        "classification": metric_summary(valid_true, ar_pred),
        "teacher_forced_comparison": {
            "teacher_forced_accuracy": float(np.mean(valid_pred == valid_true)),
            "autoregressive_accuracy": float(np.mean(ar_pred == valid_true)),
            "teacher_forced_vs_autoregressive_agreement": float(np.mean(valid_pred == ar_pred)),
        },
        "transition_metrics": transition_metrics(valid_rows, ar_pred, ar_logits),
        "hold_durations": hold_summary(valid_rows, ar_pred, intervals),
        "action_changes_per_second": hold_summary(valid_rows, ar_pred, intervals)["action_changes_per_second"],
        **autoregressive_summary(valid_rows, ar_pred, valid_pred, intervals),
    }

    per_episode: dict[str, Any] = {}
    for episode in shuffled_ids:
        episode_rows = [row for row in rows if row.episode_id == episode]
        indices = np.asarray([index for index, row in enumerate(valid_rows) if row.episode_id == episode], dtype=np.int64)
        if len(indices):
            episode_pred = valid_pred[indices]
            episode_logits = valid_logits[indices]
            role = "held_out_validation"
        else:
            episode_features = np.asarray([row.features for row in episode_rows], dtype=np.float32)
            episode_logits = logits_for(model, episode_features)
            episode_pred = np.argmax(episode_logits, axis=1).astype(np.int64)
            role = "training_episode_diagnostic_only"
        episode_true = np.asarray([row.action for row in episode_rows], dtype=np.int64)
        per_episode[episode] = {
            "role": role,
            "frames": len(episode_rows),
            "combat_frames": sum(row.phase == "combat" for row in episode_rows),
            "metrics": metric_summary(episode_true, episode_pred),
        }

    leave_one_out: list[dict[str, Any]] = []
    for fold_index, held_out_episode in enumerate(shuffled_ids):
        fold_train = [row for row in rows if row.episode_id != held_out_episode]
        fold_test = [row for row in rows if row.episode_id == held_out_episode]
        fold_x = np.asarray([row.features for row in fold_train], dtype=np.float32)
        fold_y = np.asarray([row.action for row in fold_train], dtype=np.int64)
        fold_model = train_diagnostic_model(fold_x, fold_y, args.seed + fold_index + 1, args.ablation_epochs)
        fold_test_x = np.asarray([row.features for row in fold_test], dtype=np.float32)
        fold_logits = logits_for(fold_model, fold_test_x)
        fold_pred = np.argmax(fold_logits, axis=1).astype(np.int64)
        fold_majority = int(Counter(fold_y.tolist()).most_common(1)[0][0])
        fold_result = split_metrics(
            fold_test, fold_pred, fold_logits, fold_majority,
            label="leave_one_episode_out",
        )
        leave_one_out.append({
            "held_out_episode_id": held_out_episode,
            "training_episode_ids": sorted({row.episode_id for row in fold_train}),
            "training_frames": len(fold_train),
            "test_frames": len(fold_test),
            "test_combat_frames": sum(row.phase == "combat" for row in fold_test),
            "metrics": fold_result,
        })

    split_documentation = {
        "method": "reproduces v3.train_human_demo_bc.py: sorted episode IDs, NumPy default_rng(seed=7), first floor(80%) train and remainder validation",
        "seed": args.seed,
        "episode_order_after_shuffle": shuffled_ids,
        "training_episode_ids": sorted(train_ids),
        "validation_episode_ids": sorted(valid_ids),
        "training_frames": len(train_rows),
        "validation_frames": len(valid_rows),
        "training_combat_frames": sum(row.phase == "combat" for row in train_rows),
        "validation_combat_frames": sum(row.phase == "combat" for row in valid_rows),
        "feature_width": width,
        "normalization": "none; feature blobs are consumed directly, matching the training script",
        "primary_validation_warning": "There are only three episodes, so the primary held-out result contains one complete episode; per-episode diagnostics and a future leave-one-episode-out retraining study are recommended before claiming broad generalization.",
    }
    report = {
        "schema": 1,
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": {
            key: value for key, value in checkpoint.items()
            if key not in {"model", "state_dict"} and isinstance(value, (str, int, float, bool, list, dict, type(None)))
        },
        "input": {
            "feature_rows": len(rows),
            "combat_frames": sum(row.phase == "combat" for row in rows),
            "episodes": len(shuffled_ids),
            "action_names": list(ACTION_NAMES),
            "previous_action_feature_indices": [16, 17, 18, 19, 20, 21, 22, 23, 24],
        },
        "split": split_documentation,
        "held_out": heldout,
        "held_out_combat": combat_metrics,
        "per_episode": per_episode,
        "leave_one_episode_out": leave_one_out,
        "conclusion": {
            "interpretation": "The persistence baseline, transition metrics, and autoregressive drift determine whether frame accuracy reflects decision learning or temporal carryover.",
            "production_controller_changed": False,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.report),
        "feature_rows": len(rows),
        "held_out_frames": len(valid_rows),
        "held_out_combat_frames": len(combat_rows),
        "bc_accuracy": heldout["bc_classification"]["accuracy"],
        "persistence_accuracy": heldout["previous_action_classification"]["accuracy"],
        "transition_count": heldout["transition_count"],
        "transition_bc_accuracy": heldout["transition_metrics"]["strict_accuracy"],
        "ar_accuracy": heldout["autoregressive"]["classification"]["accuracy"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
