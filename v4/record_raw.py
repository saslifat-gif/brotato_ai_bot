"""Compatibility launcher for the independent active raw recorder."""

from brotato_ai.data.cache import BoundedCache
from brotato_ai.data.recorder import main


def enforce_library_limit(root, active, max_bytes):
    """Legacy boolean facade over the observable bounded-cache report."""

    return BoundedCache(root, max_bytes=max_bytes).enforce(protected=(active,)).within_limit


if __name__ == "__main__":
    raise SystemExit(main())
