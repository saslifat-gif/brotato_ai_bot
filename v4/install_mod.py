"""Install the local v4 bridge mod into a Brotato game directory."""

import argparse
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


MOD_DIR_NAME = "Lifat-BrotatoRLBridge"
BROTATO_STEAM_APP_ID = "1942280"


def remove_stale_packages(directory: Path, keep_name: str) -> list[Path]:
    """Remove older bridge archives that confuse ModLoader's folder scan."""

    removed: list[Path] = []
    if not directory.is_dir():
        return removed
    for candidate in directory.glob(f"{MOD_DIR_NAME}-*.zip"):
        if candidate.name != keep_name:
            try:
                candidate.unlink()
            except PermissionError:
                # Brotato/Steam may keep the currently loaded Workshop ZIP
                # open while the game is running.  The new version has a
                # distinct filename, so leaving the locked archive in place
                # is safe and lets installation proceed without closing the
                # game first.
                print(f"[v4-install] stale package locked; keeping {candidate}")
            else:
                removed.append(candidate)
    return removed


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
        print(f"[v4-install] folder picker unavailable: {exc}")
        return None


def resolve_game_directory(selected: Path) -> Path:
    path = selected.resolve()
    if path.name.lower() in {"mods", "mods-unpacked"}:
        path = path.parent
    if not (path / "Brotato.exe").is_file():
        raise RuntimeError(f"Brotato.exe was not found in {path}")
    return path


def steam_workshop_root(game_directory: Path) -> Path | None:
    steamapps = next(
        (parent for parent in game_directory.parents if parent.name.lower() == "steamapps"),
        None,
    )
    if steamapps is None:
        return None
    return steamapps / "workshop" / "content" / BROTATO_STEAM_APP_ID


def steam_workshop_target(game_directory: Path, package_name: str) -> Path | None:
    """Choose a Workshop folder that this Brotato build will actually scan.

    Older Brotato ModLoader builds only inspect numeric directories belonging to
    subscribed Workshop items. A custom sibling such as ``MyMod-local`` appears
    in the log but is silently skipped. Keep the local bridge beside an existing
    subscribed mod ZIP so it is discoverable without publishing the bridge.
    """

    workshop_root = steam_workshop_root(game_directory)
    if workshop_root is None or not workshop_root.is_dir():
        return None
    candidates = sorted(
        directory
        for directory in workshop_root.iterdir()
        if directory.is_dir()
        and directory.name.isdigit()
        and any(directory.glob("*.zip"))
    )
    if not candidates:
        return None
    return candidates[0] / package_name


def default_profile_path() -> Path | None:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    return Path(appdata) / "Brotato" / "mod_user_profiles.json"


def rotate_stale_godot_log() -> Path | None:
    """Move a prior mod-load error aside so Brotato can retry mod startup.

    Brotato's crash-recovery path can disable all mods when the previous
    ``godot.log`` contains a script error mentioning ``mods-unpacked``. Keep
    the log for diagnosis, but do not let a fixed bridge remain blocked by the
    old error on the next launch.
    """

    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    log_path = Path(appdata) / "Brotato" / "logs" / "godot.log"
    if not log_path.is_file():
        return None
    try:
        contents = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if "mods-unpacked" not in contents or "SCRIPT ERROR" not in contents:
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = log_path.with_name(f"godot.before-mod-retry-{stamp}.log")
    try:
        log_path.replace(backup)
    except OSError as exc:
        print(f"[v4-install] could not rotate stale Godot log: {exc}")
        return None
    return backup


def activate_mod_profile(profile_path: Path, package: Path) -> bool:
    """Enable the bridge in the current ModLoader profile, preserving a backup."""

    if not profile_path.is_file():
        return False
    data = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    current_profile = str(data.get("current_profile") or "default")
    profiles = data.setdefault("profiles", {})
    profile = profiles.setdefault(current_profile, {})
    mod_list = profile.setdefault("mod_list", {})
    mod_list[MOD_DIR_NAME] = {
        "is_active": True,
        "zip_path": package.as_posix(),
    }

    backup = profile_path.with_name(f"{profile_path.name}.before-v4.bak")
    if not backup.exists():
        shutil.copy2(profile_path, backup)
    temporary = profile_path.with_name(f"{profile_path.name}.v4.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(profile_path)
    return True


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
    remove_stale_packages(mods, package.name)
    archive_root = Path("mods-unpacked") / MOD_DIR_NAME
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, (archive_root / path.relative_to(source)).as_posix())
    workshop_package = steam_workshop_target(game_directory, package.name)
    if workshop_package is not None:
        remove_stale_packages(workshop_package.parent, workshop_package.name)
        shutil.copy2(package, workshop_package)
    return package


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the Brotato v4 RL bridge mod")
    parser.add_argument("--game-dir", help="folder containing Brotato.exe")
    args = parser.parse_args()
    selected = Path(args.game_dir) if args.game_dir else choose_game_directory()
    if selected is None:
        print("[v4-install] cancelled")
        return 1
    package = install_mod(selected)
    rotated_log = rotate_stale_godot_log()
    print(f"[v4-install] package={package}")
    if rotated_log is not None:
        print(f"[v4-install] preserved stale crash log={rotated_log}")
    print(
        "[v4-install] editable_copy="
        f"{package.parent.parent / 'mods-unpacked' / MOD_DIR_NAME}"
    )
    workshop_package = steam_workshop_target(package.parent.parent, package.name)
    if workshop_package is not None:
        print(f"[v4-install] discoverable_workshop_package={workshop_package}")
        profile_path = default_profile_path()
        if profile_path is not None and activate_mod_profile(profile_path, workshop_package):
            print(f"[v4-install] activated_profile={profile_path}")
        else:
            print("[v4-install] profile not found; enable BrotatoRLBridge in the Mods menu")
    else:
        print(
            "[v4-install] no subscribed numeric Workshop folder found; "
            "subscribe to any Brotato mod, launch once, and rerun this installer"
        )
    print("[v4-install] launch Brotato with --enable-mods, then restart the game")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
