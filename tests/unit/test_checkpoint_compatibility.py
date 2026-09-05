import importlib
import sys

import pytest

from brotato_ai.training.checkpoints import legacy_checkpoint_modules


def test_legacy_alias_is_scoped_even_after_failure():
    before = {name: sys.modules.get(name) for name in ("v3", "v3.train_bullet_hell_finetune")}
    with pytest.raises(RuntimeError):
        with legacy_checkpoint_modules():
            legacy = importlib.import_module("v3.train_bullet_hell_finetune")
            current = importlib.import_module("v4.train_bullet_hell_finetune")
            assert legacy.BulletHellActorExtractor is current.BulletHellActorExtractor
            raise RuntimeError("failed checkpoint")
    assert {name: sys.modules.get(name) for name in before} == before
