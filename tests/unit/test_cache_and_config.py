import os

import pytest

from brotato_ai.data.cache import BoundedCache
from brotato_ai.training.configs import load_config


def test_cache_evicts_oldest_and_protects_active_file(tmp_path):
    oldest = tmp_path / "old.jsonl"
    newest = tmp_path / "new.jsonl"
    oldest.write_bytes(b"a" * 80)
    newest.write_bytes(b"b" * 80)
    os.utime(oldest, (1, 1))
    os.utime(newest, (2, 2))
    report = BoundedCache(tmp_path, max_bytes=100).enforce(protected=(newest,))
    assert report.within_limit
    assert report.removed == (oldest.resolve(),)
    assert not oldest.exists()
    assert newest.exists()


def test_cache_reports_when_protected_file_alone_exceeds_limit(tmp_path):
    active = tmp_path / "active.jsonl"
    active.write_bytes(b"x" * 101)
    report = BoundedCache(tmp_path, max_bytes=100).enforce(protected=(active,))
    assert not report.within_limit
    assert report.after_bytes == 101
    assert active.exists()


def test_config_is_validated_and_has_stable_startup_summary(tmp_path):
    cfg = load_config(
        {
            "BROTATO_V3_OUTPUT_DIR": str(tmp_path),
            "BROTATO_V4_CONTROL_HZ": "24",
            "BROTATO_V4_RECORDER_HZ": "60",
            "BROTATO_V4_CACHE_MAX_GIB": "10",
        }
    )
    assert cfg.control_hz == 24
    assert "control_hz=24" in cfg.startup_summary()
    with pytest.raises(ValueError, match="control_hz"):
        load_config(
            {
                "BROTATO_V3_OUTPUT_DIR": str(tmp_path),
                "BROTATO_V4_CONTROL_HZ": "30",
            }
        )

