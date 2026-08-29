"""Small grouped-holdout behavior-cloning baseline for human demonstrations.

LEGACY / EXPERIMENT (framewise BC).  The framewise baseline is retained for
reproducibility of the 93.1% teacher-forced result in
docs/human_bc_validation_results.md; that metric hid persistence leakage and
this model must not be imported by or influence the production runtime.  The
active learned-policy research path is the event-based model
(``v3_event_human_bc.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from brotato_ai.data.human_demo import load_training_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a discrete BC baseline from human demonstrations")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required for the BC baseline") from exc
    rows = load_training_rows(args.dataset)
    if not rows:
        raise SystemExit("dataset has no frames with semantic feature vectors")
    episodes = sorted({str(row["episode_id"]) for row in rows})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(episodes)
    split = max(1, int(len(episodes) * 0.8)) if len(episodes) > 1 else 1
    train_ids, valid_ids = set(episodes[:split]), set(episodes[split:])
    if not valid_ids:
        valid_ids = train_ids
    train = [row for row in rows if row["episode_id"] in train_ids]
    valid = [row for row in rows if row["episode_id"] in valid_ids]
    width = len(train[0]["features"])
    x_train = torch.tensor([row["features"] for row in train], dtype=torch.float32)
    y_train = torch.tensor([row["action"] for row in train], dtype=torch.long)
    x_valid = torch.tensor([row["features"] for row in valid], dtype=torch.float32)
    y_valid = torch.tensor([row["action"] for row in valid], dtype=torch.long)
    torch.manual_seed(args.seed)
    model = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, 9))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(max(1, args.epochs)):
        order = torch.randperm(len(x_train))
        model.train()
        losses = []
        for start in range(0, len(order), max(1, args.batch_size)):
            batch = order[start : start + max(1, args.batch_size)]
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            predictions = model(x_valid).argmax(dim=1)
            accuracy = float((predictions == y_valid).float().mean())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "valid_accuracy": accuracy})
        print(f"epoch={epoch + 1} train_loss={history[-1]['train_loss']:.4f} valid_accuracy={accuracy:.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "input_width": width, "schema": 1, "history": history}, args.output)
    report = {"schema": 1, "dataset": str(args.dataset), "train_frames": len(train), "valid_frames": len(valid), "episodes": len(episodes), "history": history}
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
