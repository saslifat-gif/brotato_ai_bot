import json
import socket
import time

from brotato_ai.data.recorder import DecisionTraceLogger
from brotato_ai.bridge.client import BridgeClient
from brotato_ai.performance import RuntimeProfiler
from brotato_ai.control import CombatDecisionPipeline, CombatSafetyShield, CrowdRecoveryGuard
from v3.diagnose_source import SourceCollector


def _trace():
    shield = CombatSafetyShield()
    pipeline = CombatDecisionPipeline(
        safety_shield=shield,
        crowd_recovery_guard=CrowdRecoveryGuard(shield=shield),
    )
    state = {
        "phase": "combat",
        "session": "test",
        "tick": 1,
        "published_at_ms": 1000,
        "player": {"position": {"x": 500, "y": 300}, "health": 10, "max_health": 10},
        "arena": {"width": 1000, "height": 600},
        "enemies": [],
        "projectiles": [],
        "attack_indicators": [],
        "projectile_paths": {},
    }
    return pipeline.apply(state, 4, previous_action=4)


def test_disabled_profiler_does_not_collect_samples():
    profiler = RuntimeProfiler.disabled()
    started = profiler.begin("stage")
    profiler.end("stage", started)
    assert profiler.report()["stages"] == {}


def test_profiler_reports_percentiles_and_counters():
    profiler = RuntimeProfiler(enabled=True, sample_limit=100)
    for _ in range(3):
        started = profiler.begin("stage")
        time.sleep(0.0001)
        profiler.end("stage", started)
    profiler.count("states")
    report = profiler.report()
    assert report["stages"]["stage"]["calls"] == 3
    assert report["stages"]["stage"]["p95_ms"] is not None
    assert report["counters"]["states"] == 1


def test_decision_trace_logger_is_bounded_and_flushes(tmp_path):
    path = tmp_path / "decisions.jsonl"
    logger = DecisionTraceLogger(path, queue_size=16)
    for _ in range(4):
        logger.record(_trace())
    logger.close()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert logger.written_count == 4
    assert logger.dropped_count == 0


def test_source_boundary_captures_transport_and_processing_markers():
    profiler = RuntimeProfiler(enabled=True, sample_limit=100)
    message = {
        "type": "state",
        "session": "source-test",
        "tick": 7,
        "published_at_ms": 1234,
        "bridge_eligible_at_ms": 1233,
        "bridge_dispatch_at_ms": 1234,
        "enemies": [{"runtime_id": "e1"}],
        "projectiles": [{"runtime_id": "p1"}, {"runtime_id": "p2"}],
    }
    index = profiler.source_boundary(
        receive_call_start_ns=100,
        recv_start_ns=200,
        response_arrival_ns=500,
        payload_complete_ns=500,
        parse_start_ns=520,
        parse_end_ns=700,
        payload_size_bytes=321,
        message=message,
    )
    profiler.update_source_boundary(index, "processing_start_ns", 800)
    profiler.update_source_boundary(index, "processing_end_ns", 900)
    profiler.update_source_boundary(index, "action_decision_ns", 1000)
    profiler.update_source_boundary(index, "action_sent_ns", 1100)
    sample = profiler.report()["source_samples"][0]
    assert sample["source_timestamp_ms"] == 1234
    assert sample["source_tick"] == 7
    assert sample["payload_size_bytes"] == 321
    assert sample["enemy_count"] == 1
    assert sample["projectile_count"] == 2
    assert sample["bridge_eligible_at_ms"] == 1233
    assert sample["bridge_dispatch_at_ms"] == 1234
    assert sample["processing_start_ns"] == 800
    assert sample["action_decision_ns"] == 1000
    assert sample["action_sent_ns"] == 1100


def test_source_collector_ignores_duplicate_source_states_for_fresh_intervals():
    collector = SourceCollector(sample_limit=100)
    base = {
        "type": "raw_state",
        "session": "source-test",
        "enemies": [],
        "projectiles": [{"runtime_id": "p1"}],
    }
    collector.add({**base, "published_at_ms": 1000}, local_receive_ns=0)
    collector.add({**base, "published_at_ms": 1000}, local_receive_ns=5_000_000)
    collector.add({**base, "published_at_ms": 1020}, local_receive_ns=20_000_000)
    report = collector.report(transport="test")
    assert report["records"]["received"] == 3
    assert report["records"]["fresh"] == 2
    assert report["records"]["duplicates"] == 1
    assert report["fresh_state_intervals_ms"]["mean"] == 20.0
    assert report["local_receive_intervals_ms"]["mean"] == 20.0


def test_bridge_wait_for_state_returns_newest_buffered_eligible_state():
    profiler = RuntimeProfiler(enabled=True, sample_limit=100)
    client = BridgeClient(profiler=profiler)
    receiver, sender = socket.socketpair()
    client._client = receiver
    client._connection_generation = 1
    try:
        rows = []
        for tick in (1, 2):
            rows.append(
                json.dumps(
                    {
                        "protocol": 1,
                        "type": "state",
                        "session": "buffer-test",
                        "tick": tick,
                        "published_at_ms": tick * 17,
                        "sequence": 1,
                        "phase": "combat",
                    }
                )
            )
        sender.sendall(("\n".join(rows) + "\n").encode("utf-8"))
        state = client.wait_for_state(
            timeout_sec=1.0,
            after_tick=0,
            minimum_sequence=1,
        )
        assert state["tick"] == 2
        assert client.profiler.report()["counters"]["bridge_messages_state"] == 2
    finally:
        client.close()
        sender.close()
