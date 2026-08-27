"""Fixed-input temporal ablations for the v4 actor."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def ablation_observations(
    observations: np.ndarray,
    *,
    history_start: int,
    history_size: int,
    seed: int = 17,
) -> dict[str, np.ndarray]:
    """Create deterministic normal, zero-history, and shuffled-history inputs."""
    base = np.asarray(observations, dtype=np.float32).copy()
    if base.ndim != 2:
        raise ValueError("observations must be a two-dimensional array")
    start = int(history_start)
    stop = start + int(history_size)
    if start < 0 or stop > base.shape[1]:
        raise ValueError("history slice is outside the observation")
    zeroed = base.copy()
    zeroed[:, start:stop] = 0.0
    shuffled = base.copy()
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(base))
    shuffled[:, start:stop] = base[order, start:stop]
    return {"base": base, "history_zeroed": zeroed, "history_shuffled": shuffled}


def _kl_from_base(base_logits, candidate_logits):
    import torch

    base_probability = torch.softmax(base_logits, dim=-1).clamp_min(1e-8)
    return torch.sum(
        base_probability
        * (torch.log(base_probability) - torch.log_softmax(candidate_logits, dim=-1)),
        dim=-1,
    )


def evaluate_temporal_ablation(
    actor,
    observations: np.ndarray,
    *,
    history_start: int,
    history_size: int,
    seed: int = 17,
) -> dict[str, float | int]:
    """Compare policy outputs on identical inputs with only history changed."""
    import torch

    batches = ablation_observations(
        observations,
        history_start=history_start,
        history_size=history_size,
        seed=seed,
    )
    policy = getattr(actor, "policy", actor)
    device = next(policy.parameters()).device
    def policy_logits(tensor):
        features = policy.extract_features(tensor)
        if isinstance(features, tuple):
            features = features[0]
        latent = policy.mlp_extractor.forward_actor(features)
        return policy.action_net(latent)

    with torch.no_grad():
        tensors = {
            name: torch.as_tensor(value, dtype=torch.float32, device=device)
            for name, value in batches.items()
        }
        logits = {name: policy_logits(tensor) for name, tensor in tensors.items()}
        base = logits["base"]
        residual_norm = []
        extractor = getattr(policy, "pi_features_extractor", None)
        if extractor is not None and hasattr(extractor, "actor_components"):
            _, residual = extractor.actor_components(tensors["base"])
            residual_norm = [float(residual.norm(dim=-1).mean().cpu().item())]
        result: dict[str, float | int] = {
            "samples": int(len(observations)),
            "base_residual_logit_norm": residual_norm[0] if residual_norm else 0.0,
        }
        for name in ("history_zeroed", "history_shuffled"):
            result[f"{name}_action_disagreement_rate"] = float(
                (base.argmax(dim=-1) != logits[name].argmax(dim=-1))
                .float()
                .mean()
                .cpu()
                .item()
            )
            result[f"{name}_mean_logit_l1_delta"] = float(
                (base - logits[name]).abs().mean().cpu().item()
            )
            result[f"{name}_mean_kl_from_base"] = float(
                _kl_from_base(base, logits[name]).mean().cpu().item()
            )
        return result
