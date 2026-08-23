"""Train the small UI Build Base by imitating recorded safe decisions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from v3.ui_build_policy import UiBuildBase, UiChoiceVectorizer


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            selected = int(record.get("selected_index", -1))
            actions = record.get("actions", [])
            if 0 <= selected < len(actions) and actions[selected].get("choice"):
                records.append(record)
    return records


def record_loss(model, vectorizer, record):
    state = {
        "phase": record.get("phase"),
        "wave": record.get("wave", {}),
        "counters": record.get("counters", {}),
        "build": record.get("build", {}),
    }
    actions = [action for action in record["actions"] if action.get("choice")]
    selected_id = str(record.get("selected_id", ""))
    target = next(
        index for index, action in enumerate(actions) if str(action.get("id", "")) == selected_id
    )
    rows = [vectorizer.build(state, action) for action in actions]
    scores = model(
        torch.from_numpy(np.stack([row.context for row in rows])),
        torch.from_numpy(np.stack([row.choice for row in rows])),
        torch.tensor([row.item_bucket for row in rows], dtype=torch.long),
        torch.tensor([row.base_bucket for row in rows], dtype=torch.long),
    )
    return F.cross_entropy(scores.unsqueeze(0), torch.tensor([target])), int(scores.argmax()) == target


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the compact Brotato UI Build Base")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    records = load_records(args.dataset)
    if not records:
        raise RuntimeError(f"no trainable structured UI decisions in {args.dataset}")
    random.Random(args.seed).shuffle(records)
    random.seed(args.seed)
    split = max(1, int(len(records) * 0.9))
    train_records = records[:split]
    validation_records = records[split:] or records[-1:]
    torch.manual_seed(args.seed)
    model = UiBuildBase()
    vectorizer = UiChoiceVectorizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for epoch in range(max(1, args.epochs)):
        model.train()
        random.shuffle(train_records)
        total_loss = 0.0
        for record in train_records:
            optimizer.zero_grad()
            loss, _correct = record_loss(model, vectorizer, record)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        model.eval()
        with torch.no_grad():
            correct = sum(record_loss(model, vectorizer, record)[1] for record in validation_records)
        accuracy = correct / len(validation_records)
        print(
            f"[ui-build] epoch={epoch + 1} loss={total_loss / len(train_records):.5f} "
            f"validation_accuracy={accuracy:.3f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "brotato_ui_build_base_v1",
            "state_dict": model.state_dict(),
            "parameters": model.parameter_count,
            "training_records": len(train_records),
            "validation_accuracy": accuracy,
        },
        args.output,
    )
    print(f"[ui-build] saved={args.output} parameters={model.parameter_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
