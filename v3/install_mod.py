"""Install the local v3 bridge mod into a Brotato game directory."""

import argparse
import json
import shutil
from pathlib import Path


MOD_DIR_NAME = "Lifat-BrotatoRLBridge"


def choose_game_directory() -> Path | None:
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title="Select the folder containing Brotato.exe")
        root.destroy()
        return Path(selected).resolve() if selected else None
    except Exception as exc:
        print(f"[v3-install] folder picker unavailable: {exc}")
        return None


def resolve_game_directory(selected: Path) -> Path:
    path = selected.resolve()
    if path.name.lower() == "mods-unpacked":
        path = path.parent
    if not (path / "Brotato.exe").is_file():
        raise RuntimeError(f"Brotato.exe was not found in {path}")
    return path


def install_mod(game_directory: Path, source: Path | None = None) -> Path:
    game_directory = resolve_game_directory(game_directory)
    source = source or Path(__file__).resolve().parent / "mod" / MOD_DIR_NAME
    manifest = source / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"bridge manifest is missing: {manifest}")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    expected = f"{metadata.get('namespace')}-{metadata.get('name')}"
    if expected != MOD_DIR_NAME:
        raise RuntimeError(f"manifest id mismatch: expected {MOD_DIR_NAME}, found {expected}")
    destination = game_directory / "mods-unpacked" / MOD_DIR_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the Brotato v3 RL bridge mod")
    parser.add_argument("--game-dir", help="folder containing Brotato.exe")
    args = parser.parse_args()
    selected = Path(args.game_dir) if args.game_dir else choose_game_directory()
    if selected is None:
        print("[v3-install] cancelled")
        return 1
    destination = install_mod(selected)
    print(f"[v3-install] installed={destination}")
    print("[v3-install] enable BrotatoRLBridge in Brotato's Mods menu, then restart the game")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
