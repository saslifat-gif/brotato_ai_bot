"""Observable oldest-first cache enforcement with a hard 10 GiB default."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CACHE_LIMIT_BYTES = 10 * 1024**3


@dataclass(frozen=True)
class CacheReport:
    root: Path
    max_bytes: int
    before_bytes: int
    after_bytes: int
    removed: tuple[Path, ...]
    removed_bytes: int
    within_limit: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "max_bytes": self.max_bytes,
            "before_bytes": self.before_bytes,
            "after_bytes": self.after_bytes,
            "removed": [str(path) for path in self.removed],
            "removed_bytes": self.removed_bytes,
            "within_limit": self.within_limit,
        }


class BoundedCache:
    """Enforce a hard byte limit while protecting currently written files."""

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int = DEFAULT_CACHE_LIMIT_BYTES,
        patterns: Iterable[str] = ("*.jsonl", "*.npz"),
    ):
        self.root = Path(root)
        self.max_bytes = max(1, int(max_bytes))
        self.patterns = tuple(str(pattern) for pattern in patterns)

    def _files(self) -> list[tuple[float, Path, int]]:
        seen: set[Path] = set()
        files: list[tuple[float, Path, int]] = []
        if not self.root.exists():
            return files
        for pattern in self.patterns:
            for candidate in self.root.rglob(pattern):
                try:
                    resolved = candidate.resolve()
                    stat = candidate.stat()
                except OSError:
                    continue
                if not candidate.is_file() or resolved in seen:
                    continue
                seen.add(resolved)
                files.append((stat.st_mtime, resolved, stat.st_size))
        return files

    def enforce(self, *, protected: Iterable[Path] = ()) -> CacheReport:
        protected_paths = set()
        for path in protected:
            try:
                protected_paths.add(Path(path).resolve())
            except OSError:
                continue
        files = self._files()
        before = sum(size for _, _, size in files)
        total = before
        removed: list[Path] = []
        removed_bytes = 0
        if total > self.max_bytes:
            for _, candidate, size in sorted(files):
                if candidate in protected_paths:
                    continue
                try:
                    candidate.unlink()
                except OSError:
                    continue
                removed.append(candidate)
                removed_bytes += size
                total -= size
                print(
                    f"[cache] evicted={candidate} bytes={size} "
                    f"remaining_bytes={total} limit_bytes={self.max_bytes}",
                    flush=True,
                )
                if total <= self.max_bytes:
                    break
        report = CacheReport(
            root=self.root.resolve(),
            max_bytes=self.max_bytes,
            before_bytes=before,
            after_bytes=max(0, total),
            removed=tuple(removed),
            removed_bytes=removed_bytes,
            within_limit=total <= self.max_bytes,
        )
        print(
            f"[cache] files={len(files)} bytes={report.after_bytes} "
            f"limit_bytes={report.max_bytes} removed={len(report.removed)} "
            f"within_limit={report.within_limit}",
            flush=True,
        )
        return report

