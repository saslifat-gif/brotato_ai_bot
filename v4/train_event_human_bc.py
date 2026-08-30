"""Event-based human imitation experiment for Brotato movement.

This module is deliberately offline-only.  It trains a diagnostic model to
decide whether to keep the current movement action, choose a new action when
needed, and estimate the remaining hold time.  It never changes the live
controller or existing framewise BC checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from brotato_ai.data.human_demo import DATASET_SCHEMA_VERSION, _from_blob
from brotato_ai.policy.human_action import (
    EVENT_MAX_HOLD_MS,
    EventHumanModel,
    save_event_checkpoint,
)


ACTION_COUNT = 9
ACTION_NAMES = (
    "IDLE", "UP", "DOWN", "LEFT", "RIGHT",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
)
PREVIOUS_ACTION_SLICE = slice(16, 25)
HISTORY_OFFSETS_MS = (0.0, 200.0, 400.0)
MAX_HOLD_MS = 2_000.0
HARD_NEGATIVE_RISK_THRESHOLD = 0.65
HARD_NEGATIVE_FRACTION = 0.75
HARD_NEGATIVE_SAFETY_WEIGHT = 0.20


@dataclass
class Frame:
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
    derived: dict[str, Any]
    controller: dict[str, Any]


@dataclass
class EventExample:
    frame_index: int
    input: np.ndarray
    change: int
    next_action: int
    remaining_hold_ms: float
    hard_negative_score: float
    unsafe_action_mask: np.ndarray


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def load_frames(path: Path) -> list[Frame]:
    connection = sqlite3.connect(str(path))
    records = connection.execute(
        """
        SELECT frame_id,episode_id,frame_number,timestamp_ns,phase,wave,action,
               previous_action,feature_blob,state_blob,derived_blob,controller_blob
        FROM frames
        WHERE feature_blob IS NOT NULL
        ORDER BY episode_id,frame_number,frame_id
        """
    ).fetchall()
    connection.close()
    frames: list[Frame] = []
    for record in records:
        features = _from_blob(record[8], [])
        action, previous = int(record[6]), int(record[7])
        if not isinstance(features, list) or not features or not (0 <= action < ACTION_COUNT and 0 <= previous < ACTION_COUNT):
            continue
        frames.append(Frame(
            frame_id=int(record[0]), episode_id=str(record[1]), frame_number=int(record[2]),
            timestamp_ns=int(record[3]), phase=str(record[4]),
            wave=None if record[5] is None else int(record[5]), action=action,
            previous_action=previous, features=np.asarray(features, dtype=np.float32),
            state=_from_blob(record[9], {}), derived=_from_blob(record[10], {}),
            controller=_from_blob(record[11], {}),
        ))
    return frames


def episode_split(frames: list[Frame], seed: int) -> tuple[set[str], set[str], list[str]]:
    episodes = sorted({frame.episode_id for frame in frames})
    shuffled = list(episodes)
    np.random.default_rng(seed).shuffle(shuffled)
    count = max(1, int(len(shuffled) * 0.8)) if len(shuffled) > 1 else 1
    train, test = set(shuffled[:count]), set(shuffled[count:])
    if not test:
        test = set(train)
    return train, test, shuffled


def state_features(features: np.ndarray) -> np.ndarray:
    """Remove the old explicit previous-action one-hot from a state vector."""

    result = np.asarray(features, dtype=np.float32).copy()
    result[PREVIOUS_ACTION_SLICE] = 0.0
    return result


def groups(frames: list[Frame]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for index, frame in enumerate(frames):
        result[frame.episode_id].append(index)
    return result


def monotonic_times(indices: list[int], frames: list[Frame]) -> np.ndarray:
    raw = np.asarray([frames[index].timestamp_ns for index in indices], dtype=np.int64)
    return np.maximum.accumulate(raw)


def next_transition_positions(indices: list[int], frames: list[Frame]) -> dict[int, int | None]:
    """Map each sequence position to the next action change at/after it."""

    output: dict[int, int | None] = {}
    next_change: int | None = None
    for position in range(len(indices) - 1, -1, -1):
        frame = frames[indices[position]]
        if frame.action != frame.previous_action:
            next_change = position
        output[position] = next_change
    return output


def unsafe_action_mask(frame: Frame) -> np.ndarray:
    """Return actions marked hard-unsafe by the recorded candidate-risk probe.

    This is a training signal only.  It is deliberately derived from the
    recorded safety diagnostics and never changes the production arbiter.
    The held action is removed by ``build_examples`` when the mask is used as
    a negative example, so a difficult state does not teach the model that all
    movement is forbidden.
    """

    mask = np.zeros(ACTION_COUNT, dtype=bool)
    candidates = _mapping(frame.controller.get("candidate_risks"))
    for raw_action, risk in candidates.items():
        try:
            action = int(raw_action)
        except (TypeError, ValueError):
            continue
        if 0 <= action < ACTION_COUNT and _number(_mapping(risk).get("total"), 0.0) >= HARD_NEGATIVE_RISK_THRESHOLD:
            mask[action] = True
    return mask


def build_examples(frames: list[Frame]) -> list[EventExample]:
    by_episode = groups(frames)
    state_width = len(frames[0].features)
    result: list[EventExample | None] = [None] * len(frames)
    for episode, indices in by_episode.items():
        times = monotonic_times(indices, frames)
        future_change = next_transition_positions(indices, frames)
        for position, global_index in enumerate(indices):
            frame = frames[global_index]
            snapshots: list[np.ndarray] = []
            current_time = int(times[position])
            for offset_ms in HISTORY_OFFSETS_MS:
                target = current_time - int(offset_ms * 1e6)
                history_position = int(np.searchsorted(times, target, side="right") - 1)
                history_position = min(position, max(0, history_position))
                snapshots.append(state_features(frames[indices[history_position]].features))
            current = snapshots[0]
            # Current state plus two state trends.  The action one-hot is added
            # separately so it cannot dominate the raw state normalization.
            input_state = np.concatenate((current, current - snapshots[1], current - snapshots[2]))
            action_one_hot = np.zeros(ACTION_COUNT, dtype=np.float32)
            action_one_hot[frame.previous_action] = 1.0
            model_input = np.concatenate((input_state, action_one_hot)).astype(np.float32)
            next_position = future_change[position]
            if next_position is None:
                remaining = MAX_HOLD_MS
            else:
                remaining = min(MAX_HOLD_MS, max(0.0, (times[next_position] - current_time) / 1e6))
            change = int(frame.action != frame.previous_action)
            unsafe = unsafe_action_mask(frame)
            hard_score = 0.0
            if not change:
                if next_position is not None and 0.0 < remaining <= 500.0:
                    hard_score += 3.0
                if bool(frame.derived.get("hazard_actionable")):
                    hard_score += 1.5
                if _number(frame.derived.get("projectile_count")) > 0:
                    hard_score += 1.0
                if _number(frame.derived.get("nearest_enemy_distance"), float("inf")) < 170.0:
                    hard_score += 1.0
                if frame.phase == "combat":
                    hard_score += 0.5
                unsafe_alternatives = unsafe.copy()
                if 0 <= frame.previous_action < ACTION_COUNT:
                    unsafe_alternatives[frame.previous_action] = False
                unsafe = unsafe_alternatives
                if bool(unsafe.any()):
                    hard_score += 2.0 + 0.5 * int(unsafe.sum())
                if bool(frame.controller.get("no_safe_action")):
                    hard_score += 2.0
            result[global_index] = EventExample(
                frame_index=global_index, input=model_input, change=change,
                next_action=frame.action, remaining_hold_ms=remaining,
                hard_negative_score=hard_score, unsafe_action_mask=unsafe,
            )
    return [example for example in result if example is not None]


def select_training_examples(examples: list[EventExample], train_indices: set[int], seed: int,
                             negative_ratio: int,
                             hard_negative_fraction: float = HARD_NEGATIVE_FRACTION) -> list[EventExample]:
    positives = [example for example in examples if example.frame_index in train_indices and example.change]
    negatives = [example for example in examples if example.frame_index in train_indices and not example.change]
    if not positives:
        raise RuntimeError("training split contains no action-transition events")
    target = min(len(negatives), len(positives) * max(1, negative_ratio))
    rng = np.random.default_rng(seed)
    hard_indices = [index for index, example in enumerate(negatives) if example.hard_negative_score > 0.0]
    hard_count = min(len(hard_indices), int(round(target * max(0.0, min(1.0, hard_negative_fraction)))))
    hard_indices.sort(key=lambda index: negatives[index].hard_negative_score, reverse=True)
    selected_indices = hard_indices[:hard_count]
    remaining_count = target - len(selected_indices)
    remaining_indices = [index for index in range(len(negatives)) if index not in set(selected_indices)]
    if remaining_count > 0 and remaining_indices:
        weights = np.asarray(
            [1.0 + negatives[index].hard_negative_score for index in remaining_indices],
            dtype=np.float64,
        )
        weights /= weights.sum()
        sampled = rng.choice(len(remaining_indices), size=min(remaining_count, len(remaining_indices)), replace=False, p=weights)
        selected_indices.extend(remaining_indices[int(index)] for index in sampled)
    values = positives + [negatives[int(index)] for index in selected_indices]
    rng.shuffle(values)
    return values


def normalize_inputs(train: list[EventExample], all_examples: list[EventExample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_width = len(train[0].input) - ACTION_COUNT
    train_values = np.asarray([item.input[:state_width] for item in train], dtype=np.float32)
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < 1e-5, 1.0, std)
    normalized = []
    for item in all_examples:
        state = (item.input[:state_width] - mean) / std
        normalized.append(np.concatenate((state, item.input[state_width:])).astype(np.float32))
    return np.asarray(normalized, dtype=np.float32), mean, std


def train_model(examples: list[EventExample], normalized: np.ndarray, selected_indices: list[int],
                seed: int, epochs: int, batch_size: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    # EventHumanModel (brotato_ai.policy.human_action) is the shared
    # architecture used by both offline training and the runtime adapter.
    model = EventHumanModel(int(normalized.shape[1]), ACTION_COUNT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    index_array = np.asarray(selected_indices, dtype=np.int64)
    x = torch.tensor(normalized[index_array], dtype=torch.float32)
    y_change = torch.tensor([examples[index].change for index in index_array], dtype=torch.float32)
    y_action = torch.tensor([examples[index].next_action for index in index_array], dtype=torch.long)
    y_duration = torch.tensor(
        [math.log1p(min(MAX_HOLD_MS, examples[index].remaining_hold_ms)) for index in index_array],
        dtype=torch.float32,
    )
    unsafe_mask = torch.tensor(
        np.asarray([examples[index].unsafe_action_mask for index in index_array], dtype=np.bool_),
        dtype=torch.bool,
    )
    generator = torch.Generator().manual_seed(seed)
    for _ in range(max(1, epochs)):
        order = torch.randperm(len(x), generator=generator)
        model.train()
        for start in range(0, len(order), max(16, batch_size)):
            batch = order[start:start + max(16, batch_size)]
            change_logits, action_logits, duration = model(x[batch])
            change_loss = nn.functional.binary_cross_entropy_with_logits(change_logits, y_change[batch])
            positive = y_change[batch] > 0.5
            if bool(positive.any()):
                action_loss = nn.functional.cross_entropy(action_logits[positive], y_action[batch][positive])
            else:
                action_loss = torch.zeros((), dtype=torch.float32)
            hard_negative = (~positive) & unsafe_mask[batch].any(dim=1)
            if bool(hard_negative.any()):
                probabilities = torch.softmax(action_logits[hard_negative], dim=-1)
                unsafe_mass = (probabilities * unsafe_mask[batch][hard_negative].float()).sum(dim=1)
                safety_loss = -torch.log(torch.clamp(1.0 - unsafe_mass, min=1e-5)).mean()
            else:
                safety_loss = torch.zeros((), dtype=torch.float32)
            duration_loss = nn.functional.smooth_l1_loss(duration, y_duration[batch])
            loss = change_loss + action_loss + 0.15 * duration_loss + HARD_NEGATIVE_SAFETY_WEIGHT * safety_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict(model, normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    with torch.no_grad():
        change_logits, action_logits, duration = model(torch.tensor(normalized, dtype=torch.float32))
        change_probability = torch.sigmoid(change_logits).cpu().numpy()
        actions = action_logits.cpu().numpy()
        # The duration head is only a diagnostic auxiliary target.  Clamp in
        # log space before exponentiation so an unstable cross-validation fold
        # cannot overflow the report path.
        hold_ms = np.expm1(np.clip(duration.cpu().numpy(), -10.0, math.log1p(MAX_HOLD_MS)))
    return change_probability.astype(np.float64), actions.astype(np.float64), np.clip(hold_ms, 0.0, MAX_HOLD_MS)


def binary_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual, predicted = actual.astype(bool), predicted.astype(bool)
    tp = int(np.sum(actual & predicted)); fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted)); tn = int(np.sum(~actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "positive": int(actual.sum()), "negative": int((~actual).sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def choose_threshold(actual: np.ndarray, probability: np.ndarray) -> float:
    # This calibration uses only the training episodes.  It prevents using the
    # very different event class frequency of the held-out episode.
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        f1 = float(binary_metrics(actual, probability >= threshold)["f1"])
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), f1
    return best_threshold


def selected_action(action_logits: np.ndarray, current_action: np.ndarray) -> np.ndarray:
    scores = np.asarray(action_logits, dtype=np.float64).copy()
    scores[np.arange(len(scores)), current_action] = -np.inf
    return np.argmax(scores, axis=1).astype(np.int64)


def timing_metrics(frames: list[Frame], indices: list[int], actual_change: np.ndarray,
                   predicted_change: np.ndarray) -> dict[str, Any]:
    by_episode: dict[str, list[int]] = defaultdict(list)
    for local, global_index in enumerate(indices):
        by_episode[frames[global_index].episode_id].append(local)
    offsets: list[float] = []
    matched = 0
    total = int(actual_change.sum())
    for locals_ in by_episode.values():
        truth = [position for position in locals_ if actual_change[position]]
        predicted = [position for position in locals_ if predicted_change[position]]
        used: set[int] = set()
        for event in truth:
            candidates = [position for position in predicted if position not in used]
            if not candidates:
                continue
            closest = min(candidates, key=lambda position: abs(frames[indices[position]].timestamp_ns - frames[indices[event]].timestamp_ns))
            delta = (frames[indices[closest]].timestamp_ns - frames[indices[event]].timestamp_ns) / 1e6
            if abs(delta) <= 400.0:
                used.add(closest); offsets.append(delta); matched += 1
    absolute = [abs(value) for value in offsets]
    return {
        "true_events": total, "matched_within_400ms": matched,
        "match_rate_within_400ms": matched / total if total else 0.0,
        "mean_signed_error_ms": float(np.mean(offsets)) if offsets else None,
        "mean_absolute_error_ms": float(np.mean(absolute)) if absolute else None,
        "median_absolute_error_ms": float(np.median(absolute)) if absolute else None,
        "p90_absolute_error_ms": float(np.percentile(absolute, 90)) if absolute else None,
        "within_1_frame_equivalent": sum(value <= 50.0 for value in absolute) / total if total else 0.0,
        "within_100ms": sum(value <= 100.0 for value in absolute) / total if total else 0.0,
    }


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)), "p90": float(np.percentile(array, 90))}


def context(frame: Frame) -> dict[str, str]:
    state, derived = frame.state, frame.derived
    player = _mapping(state.get("player"))
    hp = _number(player.get("health")) / max(1.0, _number(player.get("max_health"), 1.0))
    enemy_count = int(_number(derived.get("enemy_count"), len(_items(state.get("enemies")))))
    projectile_count = int(_number(derived.get("projectile_count"), len(_items(state.get("projectiles")))))
    distance = _number(derived.get("nearest_enemy_distance"), float("inf"))
    wave = frame.wave if frame.wave is not None else int(_number(_mapping(state.get("wave")).get("number"), 0))
    weapons = _items(_mapping(state.get("combat")).get("weapons"))
    weapon_key = ",".join(sorted(str(item.get("id") or item.get("type") or "") for item in weapons)) or "unknown"
    return {
        "build": weapon_key,
        "wave": str(max(0, int(wave))),
        "hazard": "actionable" if bool(derived.get("hazard_actionable")) else "not_actionable",
        "health": "low" if hp < 0.5 else "high",
        "enemy_density": "0" if enemy_count == 0 else "1-4" if enemy_count < 5 else "5-9" if enemy_count < 10 else "10+",
        "projectile_density": "0" if projectile_count == 0 else "1-4" if projectile_count < 5 else "5-9" if projectile_count < 10 else "10+",
        "nearest_enemy_distance": "<100" if distance < 100 else "100-170" if distance < 170 else "170-300" if distance < 300 else ">=300" if math.isfinite(distance) else "none",
    }


def context_report(frames: list[Frame], indices: list[int], actual: np.ndarray,
                   predicted: np.ndarray, new_action: np.ndarray, true_action: np.ndarray) -> dict[str, Any]:
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for local, global_index in enumerate(indices):
        if frames[global_index].phase != "combat":
            continue
        for key, value in context(frames[global_index]).items():
            buckets[key][value].append(local)
    output: dict[str, Any] = {}
    for key, values in buckets.items():
        output[key] = {}
        for value, positions in sorted(values.items()):
            positions_array = np.asarray(positions, dtype=np.int64)
            transitions = actual[positions_array]
            output[key][value] = {
                "frames": len(positions), "transition_events": int(transitions.sum()),
                "change_f1": binary_metrics(transitions, predicted[positions_array])["f1"],
                "new_action_accuracy_on_transition": float(np.mean(
                    new_action[positions_array][transitions] == true_action[positions_array][transitions]
                )) if transitions.any() else None,
            }
    return output


def autoregressive(frames: list[Frame], examples: list[EventExample], normalized: np.ndarray,
                   model, indices: list[int], threshold: float, mean: np.ndarray, std: np.ndarray) -> dict[str, Any]:
    state_width = len(mean)
    by_episode: dict[str, list[int]] = defaultdict(list)
    for local, global_index in enumerate(indices):
        by_episode[frames[global_index].episode_id].append(local)
    predicted_actions = np.zeros(len(indices), dtype=np.int64)
    predicted_changes = np.zeros(len(indices), dtype=bool)
    for locals_ in by_episode.values():
        current_action = frames[indices[locals_[0]]].previous_action
        for local in locals_:
            raw = examples[indices[local]].input
            value = np.concatenate(((raw[:state_width] - mean) / std, np.eye(ACTION_COUNT, dtype=np.float32)[current_action]))
            probability, logits, _hold = predict(model, value[None, :])
            change = bool(probability[0] >= threshold)
            if change:
                current_action = int(selected_action(logits, np.asarray([current_action]))[0])
            predicted_actions[local] = current_action
            predicted_changes[local] = change
    actual_actions = np.asarray([frames[index].action for index in indices], dtype=np.int64)
    actual_changes = np.asarray([examples[index].change for index in indices], dtype=bool)
    runs = 1 + int(np.sum(predicted_actions[1:] != predicted_actions[:-1])) if len(predicted_actions) else 0
    duration_ms = max(1.0, (frames[indices[-1]].timestamp_ns - frames[indices[0]].timestamp_ns) / 1e6) if indices else 1.0
    return {
        "action_accuracy": float(np.mean(predicted_actions == actual_actions)) if len(indices) else 0.0,
        "change_metrics": binary_metrics(actual_changes, predicted_changes),
        "transition_timing": timing_metrics(frames, indices, actual_changes, predicted_changes),
        "predicted_segments": runs,
        "action_changes_per_second": max(0, runs - 1) / (duration_ms / 1000.0),
    }


def evaluate_fold(frames: list[Frame], examples: list[EventExample], train_ids: set[str], test_ids: set[str],
                  seed: int, epochs: int, negative_ratio: int,
                  artifacts: dict[str, Any] | None = None) -> dict[str, Any]:
    train_indices = {index for index, frame in enumerate(frames) if frame.episode_id in train_ids}
    test_indices = [index for index, frame in enumerate(frames) if frame.episode_id in test_ids]
    selected = select_training_examples(examples, train_indices, seed, negative_ratio)
    selected_positions = [example.frame_index for example in selected]
    normalized, mean, std = normalize_inputs(selected, examples)
    model = train_model(examples, normalized, selected_positions, seed, epochs, 256)
    probability, logits, hold = predict(model, normalized)
    train_all = sorted(train_indices)
    train_actual = np.asarray([examples[index].change for index in train_all], dtype=bool)
    threshold = choose_threshold(train_actual, probability[np.asarray(train_all)])
    if artifacts is not None:
        artifacts.update({"model": model, "mean": mean, "std": std, "threshold": threshold})
    test_array = np.asarray(test_indices, dtype=np.int64)
    test_actual = np.asarray([examples[index].change for index in test_indices], dtype=bool)
    test_predicted = probability[test_array] >= threshold
    test_current = np.asarray([frames[index].previous_action for index in test_indices], dtype=np.int64)
    test_action = np.asarray([frames[index].action for index in test_indices], dtype=np.int64)
    test_new_action = selected_action(logits[test_array], test_current)
    positive = test_actual
    duration_true = np.asarray([examples[index].remaining_hold_ms for index in test_indices], dtype=np.float64)
    duration_error = hold[test_array] - duration_true
    return {
        "training_episode_ids": sorted(train_ids), "test_episode_ids": sorted(test_ids),
        "training_frames": len(train_indices), "test_frames": len(test_indices),
        "training_event_examples": len(selected),
        "training_positive_events": sum(item.change for item in selected),
        "training_hard_negative_holds": sum(not item.change for item in selected),
        "training_hard_negative_states": sum(
            int((not item.change) and (item.hard_negative_score >= 2.0 or bool(item.unsafe_action_mask.any())))
            for item in selected
        ),
        "training_unsafe_alternative_labels": sum(
            int((not item.change) and bool(item.unsafe_action_mask.any())) for item in selected
        ),
        "change_threshold_selected_on_training_only": threshold,
        "teacher_forced": {
            "change_metrics": binary_metrics(test_actual, test_predicted),
            "next_action_accuracy_on_true_change": float(np.mean(test_new_action[positive] == test_action[positive])) if positive.any() else None,
            "next_action_accuracy_when_change_detected": float(np.mean(
                test_new_action[positive & test_predicted] == test_action[positive & test_predicted]
            )) if np.any(positive & test_predicted) else None,
            "transition_timing": timing_metrics(frames, test_indices, test_actual, test_predicted),
            "hold_duration_ms": {
                "true": distribution(duration_true.tolist()),
                "predicted": distribution(hold[test_array].tolist()),
                "mean_absolute_error": float(np.mean(np.abs(duration_error))),
                "median_absolute_error": float(np.median(np.abs(duration_error))),
            },
            "context": context_report(frames, test_indices, test_actual, test_predicted, test_new_action, test_action),
        },
        "autoregressive": autoregressive(frames, examples, normalized, model, test_indices, threshold, mean, std),
        "baselines": {
            "previous_action_persistence": {
                "change_metrics": binary_metrics(test_actual, np.zeros(len(test_actual), dtype=bool)),
                "next_action_accuracy_on_true_change": 0.0,
            },
        },
    }


def prior_framewise_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    held = _mapping(report.get("held_out"))
    return {
        "source": str(path),
        "frame_accuracy": _mapping(held.get("bc_classification")).get("accuracy"),
        "previous_action_accuracy": _mapping(held.get("previous_action_classification")).get("accuracy"),
        "transition_accuracy": _mapping(_mapping(held.get("transition_metrics"))).get("strict_accuracy"),
        "autoregressive_accuracy": _mapping(_mapping(held.get("autoregressive")).get("classification")).get("accuracy"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate event-based human imitation offline")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--framewise-report", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--leave-one-episode-out", action="store_true")
    parser.add_argument(
        "--checkpoint", type=Path,
        help="save the primary-fold model with its full inference contract "
             "(still an offline diagnostic; production modes load it explicitly)",
    )
    args = parser.parse_args()
    frames = load_frames(args.dataset)
    if not frames:
        raise SystemExit("dataset has no valid feature rows")
    if len({len(frame.features) for frame in frames}) != 1:
        raise SystemExit("inconsistent feature width")
    examples = build_examples(frames)
    train_ids, test_ids, shuffled_ids = episode_split(frames, args.seed)
    primary_artifacts: dict[str, Any] = {}
    primary = evaluate_fold(
        frames, examples, train_ids, test_ids, args.seed, args.epochs, args.negative_ratio,
        artifacts=primary_artifacts,
    )
    checkpoint_path = None
    if args.checkpoint is not None:
        checkpoint_path = save_event_checkpoint(
            args.checkpoint,
            model=primary_artifacts["model"],
            mean=primary_artifacts["mean"],
            std=primary_artifacts["std"],
            change_threshold=primary_artifacts["threshold"],
            metrics={"teacher_forced": primary["teacher_forced"], "autoregressive": primary["autoregressive"]},
            dataset=str(args.dataset),
            seed=args.seed,
            action_names=ACTION_NAMES,
            history_offsets_ms=HISTORY_OFFSETS_MS,
            previous_action_slice=(PREVIOUS_ACTION_SLICE.start, PREVIOUS_ACTION_SLICE.stop),
            max_hold_ms=MAX_HOLD_MS,
            feature_schema_version=DATASET_SCHEMA_VERSION,
        )
    loo = []
    if args.leave_one_episode_out:
        for fold, held_out in enumerate(shuffled_ids):
            fold_train = set(shuffled_ids) - {held_out}
            loo.append(evaluate_fold(
                frames, examples, fold_train, {held_out}, args.seed,
                args.epochs, args.negative_ratio,
            ))
    report = {
        "schema": 1,
        "purpose": "offline event-based human imitation; production controller unchanged",
        "dataset": str(args.dataset),
        "input": {
            "feature_rows": len(frames), "combat_frames": sum(frame.phase == "combat" for frame in frames),
            "episodes": len(shuffled_ids), "feature_width": len(frames[0].features),
            "history_offsets_ms": list(HISTORY_OFFSETS_MS),
            "previous_action_feature_removed_from_state": list(range(16, 25)),
            "current_action_is_explicit_input": True,
            "normalization": "mean/std fitted on sampled training examples only; action one-hot is not normalized",
            "training_target": "all transition events plus weighted hard-negative hold states (near an upcoming transition, hazards, projectile pressure, close enemies)",
            "hard_negative_risk_threshold": HARD_NEGATIVE_RISK_THRESHOLD,
            "hard_negative_fraction": HARD_NEGATIVE_FRACTION,
            "hard_negative_safety_loss_weight": HARD_NEGATIVE_SAFETY_WEIGHT,
        },
        "split": {
            "seed": args.seed, "episode_order_after_shuffle": shuffled_ids,
            "training_episode_ids": sorted(train_ids), "test_episode_ids": sorted(test_ids),
            "method": "complete episodes only; no temporally adjacent frames cross the split",
        },
        "primary": primary,
        "leave_one_episode_out": loo,
        "old_framewise_bc": prior_framewise_summary(args.framewise_report),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = primary["teacher_forced"]
    print(json.dumps({
        "report": str(args.report),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "held_out_change_f1": summary["change_metrics"]["f1"],
        "held_out_next_action_accuracy": summary["next_action_accuracy_on_true_change"],
        "held_out_transition_timing_mae_ms": summary["transition_timing"]["mean_absolute_error_ms"],
        "autoregressive_action_accuracy": primary["autoregressive"]["action_accuracy"],
        "autoregressive_change_f1": primary["autoregressive"]["change_metrics"]["f1"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
