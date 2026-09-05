"""Load trusted local PPO checkpoints saved before the V3 module rename."""
from contextlib import contextmanager
import importlib
import sys
import types


@contextmanager
def legacy_checkpoint_modules():
    # The temporal checkpoint embeds a reference to this former import path.
    # Resolve it to the unchanged extractor without altering saved weights or
    # policy kwargs. Keep the alias scoped to loading, not runtime imports.
    aliases = {
        "v3": types.ModuleType("v3"),
        "v3.train_bullet_hell_finetune": importlib.import_module("v4.train_bullet_hell_finetune"),
    }
    added = []
    try:
        for name, module in aliases.items():
            if name not in sys.modules:
                sys.modules[name] = module
                added.append(name)
        yield
    finally:
        for name in reversed(added):
            del sys.modules[name]


def load_temporal_checkpoint(path, **kwargs):
    from v4.train_temporal_hierarchical import HumanAnchoredPPO
    with legacy_checkpoint_modules():
        return HumanAnchoredPPO.load(path, **kwargs)
