"""Legacy experiment code must stay out of the production import graph.

The framewise BC scripts are historical artifacts (their headline accuracy
was persistence leakage).  They must carry a LEGACY label and must never be
imported by the runtime package or the active v3/v4 entrypoints.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_MODULES = {
    "v3.train_combat_bc",
    "v3.train_human_demo_bc",
}
ACTIVE_MODULES = {
    "v3_event_human_bc",
    "v3_validate_human_bc",
}

RUNTIME_ROOTS = (ROOT / "src" / "brotato_ai",)
RUNTIME_ENTRYPOINTS = (
    ROOT / "v3" / "env" / "brotato_api_env.py",
    ROOT / "v3" / "combat_policy.py",
    ROOT / "v3" / "ui_build_policy.py",
    ROOT / "v4" / "train_temporal_hierarchical.py",
    ROOT / "v4" / "run_frozen.py",
    ROOT / "v4" / "combat_policy.py",
)


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_legacy_scripts_carry_a_visible_label():
    for name in ("train_combat_bc.py", "train_human_demo_bc.py"):
        header = (ROOT / "v3" / name).read_text(encoding="utf-8")[:800]
        assert "LEGACY" in header, f"{name} is missing its LEGACY label"


def test_runtime_package_does_not_import_legacy_experiments():
    for root in RUNTIME_ROOTS:
        for path in root.rglob("*.py"):
            # Skip AppleDouble companion files (._*.py) that scp copies from
            # macOS; they are binary metadata, not source.
            if path.name.startswith("._"):
                continue
            imports = _imports_of(path)
            for legacy in LEGACY_MODULES:
                assert legacy not in imports, f"{path} imports legacy {legacy}"


def test_active_entrypoints_do_not_import_legacy_experiments():
    for path in RUNTIME_ENTRYPOINTS:
        if not path.is_file():
            continue
        imports = _imports_of(path)
        for legacy in LEGACY_MODULES:
            assert legacy not in imports, f"{path} imports legacy {legacy}"


def test_event_model_script_uses_the_shared_runtime_model_class():
    """The offline script and the runtime adapter must share one architecture."""

    imports = _imports_of(ROOT / "v3_event_human_bc.py")
    assert "brotato_ai.policy.human_action" in imports
