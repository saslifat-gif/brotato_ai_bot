"""Behavior-clone the compact rich combat base from safe structured decisions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from v3.combat_policy import CombatPolicyBase, RICH_OBSERVATION_SIZE


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            features = record.get("features", [])
            action = int(record.get("action", -1))
            input_age = int(record.get("human_input_age_ms", -1))
            if record.get("source") == "human_wasd" and input_age > 250:
                continue
            if len(features) == RICH_OBSERVATION_SIZE and 0 <= action < 9:
                records.append(record)
    return records


def split_records_by_episode(
    records: list[dict], *, seed: int, validation_fraction: float = 0.1
) -> tuple[list[dict], list[dict]]:
    """Keep complete demonstrations on one side of the validation split."""

    episode_keys = {
        (str(record.get("session", "")), int(record.get("episode", -1)))
        for record in records
        if record.get("session") and record.get("episode") is not None
    }
    use_wave_segments = 0 < len(episode_keys) < 5
    groups: dict[tuple, list[dict]] = {}
    for index, record in enumerate(records):
        session = str(record.get("session", ""))
        episode = record.get("episode")
        if session and episode is not None:
            key = (session, int(episode))
            if use_wave_segments:
                key += (int(record.get("wave", -1)),)
        else:
            key = ("row", index)
        groups.setdefault(key, []).append(record)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    validation_groups = max(1, round(len(keys) * float(validation_fraction)))
    validation_groups = min(validation_groups, max(1, len(keys) - 1))
    validation_keys = set(keys[:validation_groups])
    train = [record for key in keys if key not in validation_keys for record in groups[key]]
    validation = [record for key in keys if key in validation_keys for record in groups[key]]
    if not train:
        train, validation = validation[:-1], validation[-1:]
    return train, validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the compact Brotato combat BC base")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-records", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    records = load_records(args.dataset)
    if len(records) < max(100, int(args.min_records)):
        raise RuntimeError(
            f"only {len(records)} valid combat decisions in {args.dataset}; "
            f"collect at least {max(100, int(args.min_records))} before training"
        )
    train, validation = split_records_by_episode(records, seed=args.seed)
    x_train = torch.tensor(np.asarray([r["features"] for r in train]), dtype=torch.float32)
    y_train = torch.tensor([int(r["action"]) for r in train], dtype=torch.long)
    x_valid = torch.tensor(np.asarray([r["features"] for r in validation]), dtype=torch.float32)
    y_valid = torch.tensor([int(r["action"]) for r in validation], dtype=torch.long)
    torch.manual_seed(args.seed)
    model = CombatPolicyBase()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    batch_size = max(16, int(args.batch_size))
    best_accuracy = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(max(1, int(args.epochs))):
        model.train()
        permutation = torch.randperm(len(train), generator=generator)
        total_loss = 0.0
        for start in range(0, len(train), batch_size):
            indices = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x_train[indices]), y_train[indices])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
        model.eval()
        with torch.no_grad():
            accuracy = float((model(x_valid).argmax(dim=1) == y_valid).float().mean().item())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"[combat-bc] epoch={epoch + 1} loss={total_loss / len(train):.5f} "
            f"validation_accuracy={accuracy:.3f}"
        )
    assert best_state is not None
    model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "brotato_combat_base_v1",
        "state_dict": model.state_dict(),
        "parameters": model.parameter_count,
        "training_records": len(train),
        "validation_records": len(validation),
        "training_episodes": len({(r.get("session"), r.get("episode")) for r in train}),
        "validation_episodes": len({(r.get("session"), r.get("episode")) for r in validation}),
        "training_segments": len({(r.get("session"), r.get("episode"), r.get("wave")) for r in train}),
        "validation_segments": len({(r.get("session"), r.get("episode"), r.get("wave")) for r in validation}),
        "validation_accuracy": best_accuracy,
        "best_epoch": best_epoch,
    }, args.output)
    print(f"[combat-bc] saved={args.output} parameters={model.parameter_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
