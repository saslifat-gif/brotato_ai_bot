"""Train the API-semantic combat actor while preserving the human base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from v4.combat_base import (
    SEMANTIC_OBSERVATION_SIZE,
    SemanticCombatPolicyBase,
    load_combat_base,
)
from v4.train_combat_bc import split_records_by_episode


def load_semantic_records(path: Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            features = record.get("features", [])
            action = int(record.get("action", -1))
            input_age = int(record.get("human_input_age_ms", -1))
            if input_age > 250:
                continue
            if (
                record.get("dataset") == "human_semantic_combat_v2"
                and len(features) == SEMANTIC_OBSERVATION_SIZE
                and 0 <= action < 9
            ):
                records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the semantic Brotato combat base")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-records", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    records = load_semantic_records(args.dataset)
    if len(records) < max(100, int(args.min_records)):
        raise RuntimeError(
            f"only {len(records)} semantic decisions in {args.dataset}; "
            f"collect at least {max(100, int(args.min_records))}"
        )
    train, validation = split_records_by_episode(records, seed=args.seed)
    x_train = torch.tensor(np.asarray([r["features"] for r in train]), dtype=torch.float32)
    y_train = torch.tensor([int(r["action"]) for r in train], dtype=torch.long)
    x_valid = torch.tensor(np.asarray([r["features"] for r in validation]), dtype=torch.float32)
    y_valid = torch.tensor([int(r["action"]) for r in validation], dtype=torch.long)
    old_base, old_metadata = load_combat_base(args.base_model)
    torch.manual_seed(args.seed)
    model = SemanticCombatPolicyBase(old_base)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    batch_size = max(16, int(args.batch_size))
    best_accuracy = -1.0
    best_epoch = 0
    best_state = None
    log_dir = args.tensorboard_dir or args.output.parent / "logs" / "SemanticCombatBC"
    writer = SummaryWriter(str(log_dir))
    try:
        for epoch in range(max(1, int(args.epochs))):
            model.train()
            permutation = torch.randperm(len(train), generator=generator)
            total_loss = 0.0
            for start in range(0, len(train), batch_size):
                indices = permutation[start:start + batch_size]
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x_train[indices]), y_train[indices])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item()) * len(indices)
            model.eval()
            with torch.no_grad():
                valid_logits = model(x_valid)
                valid_loss = float(F.cross_entropy(valid_logits, y_valid).item())
                accuracy = float((valid_logits.argmax(dim=1) == y_valid).float().mean().item())
            train_loss = total_loss / len(train)
            writer.add_scalar("semantic_bc/train_loss", train_loss, epoch + 1)
            writer.add_scalar("semantic_bc/validation_loss", valid_loss, epoch + 1)
            writer.add_scalar("semantic_bc/validation_accuracy", accuracy, epoch + 1)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            print(
                f"[semantic-bc] epoch={epoch + 1} loss={train_loss:.5f} "
                f"validation_loss={valid_loss:.5f} validation_accuracy={accuracy:.3f}"
            )
    finally:
        writer.close()
    assert best_state is not None
    model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "brotato_semantic_combat_base_v2",
        "state_dict": model.state_dict(),
        "parameters": model.parameter_count,
        "observation_size": SEMANTIC_OBSERVATION_SIZE,
        "training_records": len(train),
        "validation_records": len(validation),
        "validation_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "source_base": str(args.base_model.resolve()),
        "source_base_validation_accuracy": old_metadata.get("validation_accuracy"),
    }, args.output)
    print(
        f"[semantic-bc] saved={args.output} parameters={model.parameter_count} "
        f"best_epoch={best_epoch} validation_accuracy={best_accuracy:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
