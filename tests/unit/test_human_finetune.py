import numpy as np
import pytest
import torch

gym = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from v4.combat_base import CombatPolicyBase, RICH_OBSERVATION_SIZE
from v4.train_combat_finetune import (
    CombatLayerNormExtractor,
    HumanAnchoredPPO,
    actor_logits,
    initialize_actor_from_human_base,
)


class DummyRichEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(RICH_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(9)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(RICH_OBSERVATION_SIZE, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(RICH_OBSERVATION_SIZE, dtype=np.float32), 0.0, False, False, {}


def test_human_actor_transfers_exactly_into_ppo():
    torch.manual_seed(11)
    base = CombatPolicyBase()
    model = HumanAnchoredPPO(
        "MlpPolicy",
        DummyRichEnv(),
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={
            "features_extractor_class": CombatLayerNormExtractor,
            "net_arch": {"pi": [128, 64], "vf": [128, 64]},
            "activation_fn": torch.nn.Tanh,
        },
        verbose=0,
    )
    difference = initialize_actor_from_human_base(model, base)
    observations = torch.randn(12, RICH_OBSERVATION_SIZE, device=model.device)
    with torch.no_grad():
        expected = base.to(model.device)(observations)
        actual = actor_logits(model.policy, observations)
    assert difference <= 1e-5
    assert torch.allclose(expected, actual, atol=1e-5)


def test_human_anchor_is_excluded_from_checkpoints():
    model = HumanAnchoredPPO(
        "MlpPolicy",
        DummyRichEnv(),
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={
            "features_extractor_class": CombatLayerNormExtractor,
            "net_arch": {"pi": [128, 64], "vf": [128, 64]},
            "activation_fn": torch.nn.Tanh,
        },
        verbose=0,
    )
    model.set_human_anchor(
        np.zeros((4, RICH_OBSERVATION_SIZE), dtype=np.float32),
        np.asarray([0, 1, 2, 3], dtype=np.int64),
    )
    assert "_bc_features" in model._excluded_save_params()
    assert "_bc_actions" in model._excluded_save_params()
