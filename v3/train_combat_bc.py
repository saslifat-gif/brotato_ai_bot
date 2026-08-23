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
            if len(features) == RICH_OBSERVATION_SIZE and 0 <= action < 9:
                records.append(record)
    return records


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
    random.Random(args.seed).shuffle(records)
    split = max(1, int(len(records) * 0.9))
    train, validation = records[:split], records[split:] or records[-1:]
    x_train = torch.tensor(np.asarray([r["features"] for r in train]), dtype=torch.float32)
    y_train = torch.tensor([int(r["action"]) for r in train], dtype=torch.long)
    x_valid = torch.tensor(np.asarray([r["features"] for r in validation]), dtype=torch.float32)
    y_valid = torch.tensor([int(r["action"]) for r in validation], dtype=torch.long)
    torch.manual_seed(args.seed)
    model = CombatPolicyBase()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    batch_size = max(16, int(args.batch_size))
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
        print(
            f"[combat-bc] epoch={epoch + 1} loss={total_loss / len(train):.5f} "
            f"validation_accuracy={accuracy:.3f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "brotato_combat_base_v1",
        "state_dict": model.state_dict(),
        "parameters": model.parameter_count,
        "training_records": len(train),
        "validation_accuracy": accuracy,
    }, args.output)
    print(f"[combat-bc] saved={args.output} parameters={model.parameter_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
