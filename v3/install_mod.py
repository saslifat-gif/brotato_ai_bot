"""Install the local v3 bridge mod into a Brotato game directory."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path


MOD_DIR_NAME = "Lifat-BrotatoRLBridge"
BROTATO_STEAM_APP_ID = "1942280"


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
    if path.name.lower() in {"mods", "mods-unpacked"}:
        path = path.parent
    if not (path / "Brotato.exe").is_file():
        raise RuntimeError(f"Brotato.exe was not found in {path}")
    return path


def steam_workshop_target(game_directory: Path, package_name: str) -> Path | None:
    steamapps = next(
        (parent for parent in game_directory.parents if parent.name.lower() == "steamapps"),
        None,
    )
    if steamapps is None:
        return None
    return (
        steamapps
        / "workshop"
        / "content"
        / BROTATO_STEAM_APP_ID
        / f"{MOD_DIR_NAME}-local"
        / package_name
    )


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
    version = str(metadata.get("version_number", "")).strip()
    if not version:
        raise RuntimeError("bridge manifest has no version_number")

    unpacked = game_directory / "mods-unpacked" / MOD_DIR_NAME
    unpacked.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, unpacked, dirs_exist_ok=True)

    mods = game_directory / "mods"
    mods.mkdir(parents=True, exist_ok=True)
    package = mods / f"{MOD_DIR_NAME}-{version}.zip"
    archive_root = Path("mods-unpacked") / MOD_DIR_NAME
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, (archive_root / path.relative_to(source)).as_posix())
    workshop_package = steam_workshop_target(game_directory, package.name)
    if workshop_package is not None:
        workshop_package.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package, workshop_package)
    return package


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the Brotato v3 RL bridge mod")
    parser.add_argument("--game-dir", help="folder containing Brotato.exe")
    args = parser.parse_args()
    selected = Path(args.game_dir) if args.game_dir else choose_game_directory()
    if selected is None:
        print("[v3-install] cancelled")
        return 1
    package = install_mod(selected)
    print(f"[v3-install] package={package}")
    print(
        "[v3-install] editable_copy="
        f"{package.parent.parent / 'mods-unpacked' / MOD_DIR_NAME}"
    )
    workshop_package = steam_workshop_target(package.parent.parent, package.name)
    if workshop_package is not None:
        print(f"[v3-install] steam_test_package={workshop_package}")
    else:
        print("[v3-install] non-Steam layout; no Steam workshop test copy created")
    print("[v3-install] enable BrotatoRLBridge in Brotato's Mods menu, then restart the game")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
