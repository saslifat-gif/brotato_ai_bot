import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from v3.bridge_server import BridgeServer
from v3.combat_policy import (
    CombatHeuristicTeacher,
    CombatPolicyBase,
    CombatSafetyShield,
    HumanCombatDecisionLogger,
    SemanticHumanCombatDecisionLogger,
    RICH_OBSERVATION_SIZE,
    RichCombatVectorizer,
    SEMANTIC_OBSERVATION_SIZE,
    SemanticCombatPolicyBase,
    SemanticCombatVectorizer,
    FULL_ARENA_GRID_SIZE,
    FULL_ARENA_OBSERVATION_SIZE,
    FullArenaCombatVectorizer,
)
from v3.install_mod import MOD_DIR_NAME, activate_mod_profile, install_mod
from v3.record_human import require_human_input_capability, should_record
from v3.protocol import (
    BridgeProtocolError,
    action_message,
    configure_message,
    decode_message,
    encode_message,
    ui_action_message,
)
from v3.reward import ApiRewardEngine
from v3.run_frozen import load_combat_bc
from v3.train_ui_build import load_records
from v3.train_combat_bc import split_records_by_episode
from v3.vectorizer import ApiStateVectorizer, OBSERVATION_SIZE
from v3.ui_automation import AutoUiController, available_actions, shop_budget_limits
from v3.ui_build_policy import (
    CHOICE_SIZE,
    CONTEXT_SIZE,
    StickMeleeTeacher,
    UiBuildBase,
    UiChoiceVectorizer,
)


def _state(*, hp=10, max_hp=10, wave=1, materials=0, kills=0):
    return {
        "type": "state",
        "protocol": 1,
        "tick": 1,
        "phase": "combat",
        "arena": {"width": 1000, "height": 600},
        "player": {
            "position": {"x": 500, "y": 300},
            "velocity": {"x": 0, "y": 0},
            "health": hp,
            "max_health": max_hp,
        },
        "wave": {"number": wave, "time_left": 30, "duration": 60},
        "counters": {"materials": materials, "kills": kills},
        "enemies": [],
        "projectiles": [],
        "pickups": [],
        "ui": {"actions": []},
        "dead": False,
        "victory": False,
    }


def test_protocol_round_trip_and_action_validation():
    message = action_message(8, sequence=12)
    assert decode_message(encode_message(message)) == {
        "type": "action",
        "protocol": 1,
        "sequence": 12,
        "action": 8,
    }
    with pytest.raises(BridgeProtocolError):
        action_message(9, sequence=1)
    assert ui_action_message("/root/Shop/NextWave", 13) == {
        "type": "ui_action",
        "sequence": 13,
        "target": "/root/Shop/NextWave",
    }
    with pytest.raises(BridgeProtocolError):
        ui_action_message("relative/path", sequence=1)
    assert configure_message(state_hz=8) == {"type": "configure", "state_hz": 8.0}
    with pytest.raises(BridgeProtocolError):
        configure_message(state_hz=30)


def test_vectorizer_has_fixed_finite_shape_and_nearest_enemy_first():
    state = _state()
    state["enemies"] = [
        {"position": {"x": 900, "y": 300}, "health": 1, "max_health": 1},
        {"position": {"x": 550, "y": 300}, "health": 1, "max_health": 1},
    ]
    observation = ApiStateVectorizer().build(state, previous_action=4)
    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert observation[16] == pytest.approx(0.05)


def test_rich_combat_vectorizer_and_base_are_versioned_and_small():
    state = _state()
    state["combat"] = {
        "weapon_count": 2,
        "melee_count": 2,
        "ranged_count": 0,
        "weapon_range": 180,
        "move_speed": 320,
        "armor": 4,
        "attack_speed": 15,
    }
    state["enemies"] = [{
        "position": {"x": 550, "y": 300},
        "velocity": {"x": -10, "y": 0},
        "health": 8,
        "max_health": 10,
        "radius": 35,
        "is_charging": True,
    }]
    observation = RichCombatVectorizer().build(state, previous_action=4)
    assert observation.shape == (RICH_OBSERVATION_SIZE,)
    assert np.isfinite(observation).all()
    assert observation[16 + 4] == 1.0
    model = CombatPolicyBase()
    assert model.parameter_count < 100_000
    assert model(torch.from_numpy(observation[None, :])).shape == (1, 9)


def test_frozen_runner_loads_combat_bc_checkpoint(tmp_path):
    path = tmp_path / "combat.pt"
    source = CombatPolicyBase()
    torch.save(
        {
            "format": "brotato_combat_base_v1",
            "state_dict": source.state_dict(),
            "validation_accuracy": 0.625,
            "best_epoch": 14,
        },
        path,
    )
    loaded, metadata = load_combat_bc(path)
    assert metadata["validation_accuracy"] == 0.625
    observation = torch.zeros((1, RICH_OBSERVATION_SIZE))
    assert loaded(observation).shape == (1, 9)


def test_human_combat_logger_writes_bc_record(tmp_path):
    path = tmp_path / "human.jsonl"
    state = dict(_state(wave=3), session="demo-session", human_input_age_ms=4)
    HumanCombatDecisionLogger(path).record(
        state, 8, previous_action=4, episode=2
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["dataset"] == "human_combat_v1"
    assert record["source"] == "human_wasd"
    assert record["session"] == "demo-session"
    assert record["episode"] == 2
    assert record["action"] == 8
    assert len(record["features"]) == RICH_OBSERVATION_SIZE


def test_semantic_vector_and_model_preserve_old_actor_with_new_api_fields(tmp_path):
    state = _state()
    state["enemies"] = [{
        "position": {"x": 700, "y": 330},
        "velocity": {"x": -40, "y": 0},
        "health": 30,
        "max_health": 30,
        "radius": 70,
        "id": "enemy_charger",
        "type": "charger",
        "width": 140,
        "height": 90,
        "contact_damage": 12,
        "attack_cooldown_remaining": 0.3,
        "is_attacking": True,
        "is_elite": True,
        "attack_type": "charge",
        "movement_type": "follow_player",
    }]
    state["pickups"] = [{
        "id": "fruit",
        "type": "healing_fruit",
        "category": "healing",
        "position": {"x": 520, "y": 320},
        "healing": 4,
        "material_value": 0,
        "crate_value": 0,
        "width": 30,
        "height": 30,
    }]
    state["combat"] = {"weapons": [{
        "id": "weapon_stick_1",
        "attack_type": "melee",
        "range": 180,
        "cooldown_remaining": 0.2,
        "cooldown_duration": 1.0,
        "reload_remaining": 0,
        "ammo": -1,
        "ammo_capacity": -1,
        "ready": False,
        "is_attacking": True,
        "is_reloading": False,
    }]}
    state["attack_indicators"] = [{
        "id": "boss_warning",
        "type": "aoe_warning",
        "position": {"x": 600, "y": 300},
        "direction": {"x": -1, "y": 0},
        "width": 200,
        "height": 100,
        "time_to_activate": 0.5,
        "damage": 15,
        "active": False,
    }]
    vectorizer = SemanticCombatVectorizer()
    features = vectorizer.build(state, previous_action=4)
    assert features.shape == (SEMANTIC_OBSERVATION_SIZE,)
    assert np.isfinite(features).all()
    assert np.count_nonzero(features[RICH_OBSERVATION_SIZE:]) > 20
    old = CombatPolicyBase()
    semantic = SemanticCombatPolicyBase(old)
    assert semantic.parameter_count < 100_000
    batch = torch.from_numpy(features[None, :])
    with torch.no_grad():
        assert torch.allclose(semantic(batch), old(batch[:, :RICH_OBSERVATION_SIZE]))

    path = tmp_path / "semantic.jsonl"
    SemanticHumanCombatDecisionLogger(path).record(
        state, 4, previous_action=3, episode=1
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["dataset"] == "human_semantic_combat_v2"
    assert record["schema"] == 2
    assert len(record["features"]) == SEMANTIC_OBSERVATION_SIZE
    assert record["counts"] == {
        "enemies": 1,
        "pickups": 1,
        "indicators": 1,
        "weapons": 1,
    }


def test_semantic_ppo_actor_initialization_is_exact():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    from gymnasium import spaces
    from v3.train_combat_finetune import HumanAnchoredPPO
    from v3.train_semantic_finetune import (
        SemanticActorExtractor,
        initialize_actor_from_semantic_base,
    )

    class DummySemanticEnv(gym.Env):
        action_space = spaces.Discrete(9)
        observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(SEMANTIC_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(SEMANTIC_OBSERVATION_SIZE, dtype=np.float32), {}

        def step(self, _action):
            return (
                np.zeros(SEMANTIC_OBSERVATION_SIZE, dtype=np.float32),
                0.0,
                False,
                False,
                {},
            )

    base = SemanticCombatPolicyBase()
    model = HumanAnchoredPPO(
        "MlpPolicy",
        DummySemanticEnv(),
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={
            "features_extractor_class": SemanticActorExtractor,
            "net_arch": {"pi": [], "vf": [16]},
            "activation_fn": torch.nn.Tanh,
            "share_features_extractor": False,
        },
    )
    assert initialize_actor_from_semantic_base(model, base) <= 1e-5


def test_full_arena_vector_preserves_semantic_prefix_and_attack_geometry():
    state = _state()
    state["enemies"] = [{
        "position": {"x": 700, "y": 330},
        "velocity": {"x": -40, "y": 10},
        "radius": 70,
        "contact_damage": 4,
        "is_attacking": True,
        "charge_direction": {"x": -1, "y": 0.25},
        "attack_target": {"x": 520, "y": 360},
    }]
    semantic = SemanticCombatVectorizer().build(state, previous_action=4)
    full = FullArenaCombatVectorizer().build(state, previous_action=4)
    assert full.shape == (FULL_ARENA_OBSERVATION_SIZE,)
    assert np.array_equal(full[:SEMANTIC_OBSERVATION_SIZE], semantic)
    assert np.count_nonzero(
        full[SEMANTIC_OBSERVATION_SIZE:SEMANTIC_OBSERVATION_SIZE + FULL_ARENA_GRID_SIZE]
    ) > 0
    attack = full[SEMANTIC_OBSERVATION_SIZE + FULL_ARENA_GRID_SIZE:]
    assert attack[:4] == pytest.approx([-1.0, 0.25, 0.02, 0.10])


def test_full_arena_vector_prefers_bridge_grid_that_includes_all_enemies():
    state = _state()
    state["enemies"] = []
    exported = np.zeros(10 * 6 * 4, dtype=np.float32)
    exported[-4:] = [0.75, 0.5, -0.25, 0.125]
    state["arena_grid"] = {"enemy": exported.tolist()}
    full = FullArenaCombatVectorizer().build(state)
    grid = full[
        SEMANTIC_OBSERVATION_SIZE:SEMANTIC_OBSERVATION_SIZE + FULL_ARENA_GRID_SIZE
    ].reshape(-1, 10)
    assert grid[-1, :4] == pytest.approx([0.75, 0.5, -0.25, 0.125])


def test_full_arena_ppo_transfer_preserves_trained_semantic_logits():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    from gymnasium import spaces
    from v3.train_combat_finetune import HumanAnchoredPPO, actor_logits
    from v3.train_full_arena_finetune import (
        FullArenaActorExtractor,
        initialize_full_arena_from_semantic_ppo,
    )
    from v3.train_semantic_finetune import SemanticActorExtractor

    class DummyEnv(gym.Env):
        action_space = spaces.Discrete(9)

        def __init__(self, observation_size):
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(observation_size,),
                dtype=np.float32,
            )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, _action):
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}

    source = HumanAnchoredPPO(
        "MlpPolicy",
        DummyEnv(SEMANTIC_OBSERVATION_SIZE),
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={
            "features_extractor_class": SemanticActorExtractor,
            "net_arch": {"pi": [], "vf": [16]},
            "activation_fn": torch.nn.Tanh,
            "share_features_extractor": False,
        },
    )
    with torch.no_grad():
        source.policy.action_net.weight.normal_(0.0, 0.03)
        source.policy.action_net.bias.normal_(0.0, 0.03)
    target = HumanAnchoredPPO(
        "MlpPolicy",
        DummyEnv(FULL_ARENA_OBSERVATION_SIZE),
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={
            "features_extractor_class": FullArenaActorExtractor,
            "net_arch": {"pi": [], "vf": [16]},
            "activation_fn": torch.nn.Tanh,
            "share_features_extractor": False,
        },
    )
    assert initialize_full_arena_from_semantic_ppo(target, source) <= 1e-5
    old_observation = torch.rand((4, SEMANTIC_OBSERVATION_SIZE)) * 2.0 - 1.0
    full_observation = torch.rand((4, FULL_ARENA_OBSERVATION_SIZE)) * 2.0 - 1.0
    full_observation[:, :SEMANTIC_OBSERVATION_SIZE] = old_observation
    with torch.no_grad():
        assert torch.allclose(
            actor_logits(source.policy, old_observation),
            actor_logits(target.policy, full_observation),
            atol=1e-6,
        )


def test_human_recorder_samples_transitions_and_throttles_idle():
    assert should_record(4, 3, 0.01, sample_hz=8, idle_hz=2)
    assert not should_record(4, 4, 0.05, sample_hz=8, idle_hz=2)
    assert should_record(4, 4, 0.13, sample_hz=8, idle_hz=2)
    assert not should_record(0, 0, 0.3, sample_hz=8, idle_hz=2)
    assert should_record(0, 0, 0.51, sample_hz=8, idle_hz=2)


def test_human_recorder_rejects_old_bridge():
    require_human_input_capability({
        "capabilities": ["human_input_observation", "semantic_entities_v2"]
    })
    with pytest.raises(RuntimeError, match="Bridge 0.3.3"):
        require_human_input_capability({"capabilities": ["structured_state"]})


def test_combat_bc_validation_split_keeps_episodes_together():
    records = [
        {"session": "s", "episode": episode, "row": row}
        for episode in range(5)
        for row in range(4)
    ]
    train, validation = split_records_by_episode(records, seed=7)
    train_episodes = {(row["session"], row["episode"]) for row in train}
    validation_episodes = {(row["session"], row["episode"]) for row in validation}
    assert train_episodes
    assert validation_episodes
    assert train_episodes.isdisjoint(validation_episodes)


def test_combat_bc_falls_back_to_whole_wave_split_for_few_runs():
    records = [
        {"session": "s", "episode": 0, "wave": wave, "row": row}
        for wave in range(1, 8)
        for row in range(3)
    ]
    train, validation = split_records_by_episode(records, seed=7)
    train_waves = {row["wave"] for row in train}
    validation_waves = {row["wave"] for row in validation}
    assert train_waves.isdisjoint(validation_waves)


def test_trainers_save_the_best_validation_epoch():
    combat_trainer = (ROOT / "v3" / "train_combat_bc.py").read_text(encoding="utf-8")
    ui_trainer = (ROOT / "v3" / "train_ui_build.py").read_text(encoding="utf-8")
    for source in (combat_trainer, ui_trainer):
        assert "best_accuracy = -1.0" in source
        assert '"best_epoch": best_epoch' in source
        assert "model.load_state_dict(best_state)" in source


def test_safety_shield_sidesteps_an_incoming_projectile():
    state = _state()
    state["projectiles"] = [{
        "position": {"x": 700, "y": 300},
        "velocity": {"x": -600, "y": 0},
        "radius": 15,
    }]
    shield = CombatSafetyShield()
    decision = shield.apply(state, requested_action=4)
    assert decision.overridden
    assert decision.applied_action in {1, 2, 5, 7}
    assert decision.applied_risk < decision.requested_risk


def test_combat_teacher_moves_toward_a_safe_distant_enemy():
    state = _state()
    state["combat"] = {"weapon_range": 120}
    state["enemies"] = [{
        "position": {"x": 850, "y": 300},
        "velocity": {"x": 0, "y": 0},
        "health": 1,
        "max_health": 1,
    }]
    assert CombatHeuristicTeacher().select(state) in {4, 6, 8}


def test_exact_state_reward_penalizes_damage_and_rewards_progress():
    engine = ApiRewardEngine()
    engine.reset(_state(hp=10, materials=0, kills=0))
    reward = engine.step(_state(hp=8, materials=2, kills=1))
    assert reward < 0
    progress = engine.step(_state(hp=8, wave=2, materials=3, kills=2))
    assert progress > 10

    engine.reset(_state())
    wave_end = dict(_state(), phase="wave_end")
    assert engine.step(wave_end) > 9

    late_engine = ApiRewardEngine()
    late_engine.reset(_state(wave=10))
    late_death = late_engine.step(dict(_state(wave=10), dead=True))
    early_engine = ApiRewardEngine()
    early_engine.reset(_state(wave=1))
    early_death = early_engine.step(dict(_state(wave=1), dead=True))
    assert late_death < early_death


def test_mod_manifest_and_extension_are_packaged():
    mod = ROOT / "v3" / "mod" / MOD_DIR_NAME
    manifest = json.loads((mod / "manifest.json").read_text(encoding="utf-8"))
    assert f"{manifest['namespace']}-{manifest['name']}" == MOD_DIR_NAME
    assert manifest["extra"]["godot"]["compatible_mod_loader_version"]
    assert (mod / "mod_main.gd").is_file()
    assert (mod / "bridge.gd").is_file()
    assert not (mod / "extensions" / "main.gd").exists()
    assert (
        mod
        / "extensions"
        / "entities"
        / "units"
        / "movement_behaviors"
        / "player_movement_behavior.gd"
    ).is_file()


def test_bridge_uses_godot3_safe_boolean_type_and_load_guard():
    mod = ROOT / "v3" / "mod" / MOD_DIR_NAME
    bridge = (mod / "bridge.gd").read_text(encoding="utf-8")
    mod_main = (mod / "mod_main.gd").read_text(encoding="utf-8")
    assert "var dead: bool =" in bridge
    assert "var player = TempStats.player" in bridge
    assert "func observe_movement_behavior(behavior, human_movement" in bridge
    assert '"human_action": _latest_human_action' in bridge
    assert '"human_input_observation"' in bridge
    assert "_find_player_descendant(main)" in bridge
    assert 'enemy.connect("died", self, "_on_enemy_died_observed")' in bridge
    assert "func _run_player_data(run_data, player)" in bridge
    assert 'if typeof(object) == TYPE_DICTIONARY:' in bridge
    assert '["gold", "materials"]' in bridge
    assert "func _collect_ui_actions(node, output: Array, phase: String)" in bridge
    assert 'if role != "other":' in bridge
    assert "func _detect_visible_ui_phase(node)" in bridge
    assert "func _ui_action_context(node)" not in bridge
    assert "_last_ui_action_result" not in bridge
    assert 'button_name == "choosebutton"' in bridge
    assert "visible_ui_phase" in bridge
    assert 'node.emit_signal("pressed")' in bridge
    assert 'str(node.name).to_lower() == "gobutton"' in bridge
    assert '"realtime_control"' in bridge
    assert "const EARLY_STATE_INTERVAL_SEC := 1.0 / 24.0" in bridge
    assert "const MID_STATE_INTERVAL_SEC := 1.0 / 16.0" in bridge
    assert "const LATE_STATE_INTERVAL_SEC := 1.0 / 12.0" in bridge
    assert "const MAX_ENEMIES := 64" in bridge
    assert "const MAX_PROJECTILES := 64" in bridge
    assert "const MAX_PICKUPS := 32" in bridge
    assert "func _state_interval_sec() -> float:" in bridge
    assert '"semantic_entities_v2"' in bridge
    assert '"pickup_semantics"' in bridge
    assert '"weapon_readiness"' in bridge
    assert '"attack_indicators"' in bridge
    assert '"full_arena_grid_v1"' in bridge
    assert '"arena_grid"' in bridge
    assert "func _full_arena_enemy_grid(" in bridge
    assert '"configurable_state_rate"' in bridge
    assert "_tick - _last_indicator_scan_tick >= 6" in bridge
    assert "var _property_name_cache := {}" in bridge
    assert '"width": static_data["width"]' in bridge
    assert '"healing": static_data["healing"]' in bridge
    assert "set_pause(true)" not in bridge
    assert 'if lower == "main":' in bridge
    assert "main_extension" not in mod_main
    assert "vanilla death/drop logic preserved" in mod_main
    assert "bridge_script.can_instance()" in mod_main
    assert "Bridge script failed to load" in mod_main
    assert 'call_deferred("_attach_bridge", root, _bridge)' in mod_main
    assert "_bridge.queue_free()" not in mod_main
    movement_extension = (
        mod
        / "extensions"
        / "entities"
        / "units"
        / "movement_behaviors"
        / "player_movement_behavior.gd"
    ).read_text(encoding="utf-8")
    assert "bridge.observe_movement_behavior(self, human_movement)" in movement_extension
    assert "var human_movement = .get_movement()" in movement_extension


def test_ui_automation_buys_then_rerolls_then_starts_wave():
    controller = AutoUiController(max_shop_buys=1, max_shop_rerolls=1)
    state = dict(_state(materials=20), phase="shop")
    state["ui"] = {
        "actions": [
            {"id": "/root/Shop/Buy", "role": "buy", "enabled": True},
            {"id": "/root/Shop/Reroll", "role": "reroll", "enabled": True},
            {"id": "/root/Shop/Go", "role": "next_wave", "enabled": True},
        ]
    }
    buy = controller.choose(state)
    assert buy["role"] == "buy"
    controller.mark_sent(state, buy)
    reroll = controller.choose(state)
    assert reroll["role"] == "reroll"
    controller.mark_sent(state, reroll)
    assert controller.choose(state)["role"] == "next_wave"


def test_late_rich_shop_spends_instead_of_hoarding():
    buys, rerolls, reserve = shop_budget_limits(
        9, 1000, base_buys=4, base_rerolls=1
    )
    assert buys >= 40
    assert rerolls >= 16
    assert reserve <= 100
    assert shop_budget_limits(9, 1000, base_buys=0, base_rerolls=0)[:2] == (0, 0)


def test_rich_shop_accepts_positive_low_score_item():
    state = _state(wave=8, materials=570)
    onion = {
        "id": "/root/Shop/Onion/BuyButton",
        "role": "buy",
        "enabled": True,
        "choice": {
            "id": "item_terrified_onion",
            "category": "item",
            "price": 32,
            "affordable": True,
            "effects": [
                {"key": "stat_speed", "value": 4},
                {"key": "stat_luck", "value": -5},
            ],
        },
    }
    selected = StickMeleeTeacher().select(state, [onion])
    assert selected is not None
    assert selected.action["choice"]["id"] == "item_terrified_onion"


def test_ui_automation_only_uses_enabled_exact_targets():
    state = dict(_state(), phase="upgrade")
    state["ui"] = {
        "actions": [
            {"id": "relative", "role": "upgrade_choice", "enabled": True},
            {"id": "/root/Disabled", "role": "upgrade_choice", "enabled": False},
            {"id": "/root/Choice", "role": "upgrade_choice", "enabled": True},
        ]
    }
    assert available_actions(state, "upgrade_choice") == [
        {"id": "/root/Choice", "role": "upgrade_choice", "enabled": True}
    ]
    assert AutoUiController().choose(state)["id"] == "/root/Choice"


def test_ui_automation_takes_found_item_before_recycle():
    state = dict(_state(wave=7), phase="item_found")
    state["ui"] = {
        "actions": [
            {
                "id": "/root/Main/UI/ItemFound/RecycleButton",
                "role": "recycle_item",
                "enabled": True,
            },
            {
                "id": "/root/Main/UI/ItemFound/TakeButton",
                "role": "take_item",
                "enabled": True,
            },
        ]
    }
    controller = AutoUiController()
    action = controller.choose(state)
    assert action["role"] == "take_item"
    controller.mark_sent(state, action)


def test_bridge_detects_found_item_take_and_recycle_buttons():
    bridge = (
        ROOT / "v3" / "mod" / MOD_DIR_NAME / "bridge.gd"
    ).read_text(encoding="utf-8")
    assert 'return "item_found"' in bridge
    assert 'return "take_item"' in bridge
    assert 'return "recycle_item"' in bridge
    assert 'button_text.find("拿取")' in bridge
    assert 'button_text.find("回收")' in bridge


def test_bridge_advertises_language_independent_build_choices():
    bridge = (
        ROOT / "v3" / "mod" / MOD_DIR_NAME / "bridge.gd"
    ).read_text(encoding="utf-8")
    assert '"choice"' in bridge
    assert '"build": build_state' in bridge
    assert '"base_id": weapon_id if not weapon_id.empty() else upgrade_id' in bridge
    assert '"affordable"' in bridge
    assert '"stat_melee_damage"' in bridge
    assert 'property_name = "upgrade_data"' in bridge
    assert '"combat": combat_state' in bridge
    assert '"combat_build_summary"' in bridge
    assert '"threat_geometry"' in bridge
    assert '"is_charging"' in bridge
    assert 'func _collision_shape_data(object) -> Dictionary:' in bridge


def test_stick_melee_teacher_prioritizes_stick_and_melee_upgrade():
    teacher = StickMeleeTeacher()
    shop = dict(_state(wave=4, materials=100), phase="shop")
    stick = {
        "id": "/root/Shop/Stick/BuyButton",
        "role": "buy",
        "enabled": True,
        "choice": {
            "id": "weapon_stick_1",
            "base_id": "weapon_stick",
            "category": "weapon",
            "weapon_type": 0,
            "price": 30,
            "affordable": True,
            "tier": 0,
            "effects": [],
        },
    }
    ranged = {
        "id": "/root/Shop/Gun/BuyButton",
        "role": "buy",
        "enabled": True,
        "choice": {
            "id": "weapon_pistol_1",
            "base_id": "weapon_pistol",
            "category": "weapon",
            "weapon_type": 1,
            "price": 10,
            "affordable": True,
            "tier": 0,
            "effects": [],
        },
    }
    assert teacher.select(shop, [ranged, stick]).action["id"] == stick["id"]
    spear = dict(
        ranged,
        id="/root/Shop/Spear/BuyButton",
        choice={
            "id": "weapon_spear_1",
            "base_id": "weapon_spear",
            "category": "weapon",
            "weapon_type": 0,
            "price": 10,
            "affordable": True,
            "tier": 0,
            "effects": [],
        },
    )
    assert teacher.select(shop, [spear]) is None

    upgrade = dict(_state(wave=4), phase="upgrade")
    melee = dict(
        stick,
        id="/root/Upgrade/Melee",
        role="upgrade_choice",
        choice={
            "id": "upgrade_melee_damage_2",
            "base_id": "upgrade_melee_damage",
            "category": "upgrade",
            "tier": 1,
            "effects": [{"key": "stat_melee_damage", "value": 4}],
        },
    )
    elemental = dict(
        melee,
        id="/root/Upgrade/Elemental",
        choice={
            "id": "upgrade_elemental_damage_2",
            "base_id": "upgrade_elemental_damage",
            "category": "upgrade",
            "tier": 1,
            "effects": [{"key": "stat_elemental_damage", "value": 4}],
        },
    )
    assert teacher.select(upgrade, [elemental, melee]).action["id"] == melee["id"]


def test_ui_build_base_is_small_and_has_stable_features():
    state = dict(_state(wave=7, materials=120), phase="shop")
    state["build"] = {
        "character_id": "character_well_rounded",
        "weapons": [{"id": "weapon_stick_1", "base_id": "weapon_stick"}],
        "items": [],
        "stats": {"stat_melee_damage": 8, "stat_attack_speed": 12},
    }
    action = {
        "id": "/root/Shop/Stick/BuyButton",
        "role": "buy",
        "choice": {
            "id": "weapon_stick_2",
            "base_id": "weapon_stick",
            "category": "weapon",
            "weapon_type": 0,
            "tier": 1,
            "price": 55,
            "affordable": True,
            "effects": [{"key": "stat_melee_damage", "value": 2}],
        },
    }
    features = UiChoiceVectorizer().build(state, action)
    assert features.context.shape == (CONTEXT_SIZE,)
    assert features.choice.shape == (CHOICE_SIZE,)
    model = UiBuildBase()
    assert model.parameter_count < 200_000
    score = model(
        torch.from_numpy(features.context[None, :]),
        torch.from_numpy(features.choice[None, :]),
        torch.tensor([features.item_bucket]),
        torch.tensor([features.base_bucket]),
    )
    assert score.shape == (1,)


def test_ui_build_training_filters_stale_and_unstructured_decisions(tmp_path):
    dataset = tmp_path / "ui_decisions.jsonl"
    valid_action = {
        "id": "/root/Shop/Stick/BuyButton",
        "role": "buy",
        "choice": {"id": "weapon_stick_1", "category": "weapon"},
    }
    records = [
        {
            "policy_source": "stick_melee_teacher",
            "selected_index": 0,
            "actions": [valid_action],
        },
        {
            "policy_source": "stick_melee_teacher_v2",
            "selected_index": 0,
            "actions": [{"id": "/root/Shop/Reroll", "role": "reroll"}],
        },
        {
            "policy_source": "stick_melee_teacher_v2",
            "selected_index": 0,
            "actions": [valid_action],
        },
    ]
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    loaded = load_records(dataset, "stick_melee_teacher_v2")

    assert len(loaded) == 1
    assert loaded[0]["actions"][0]["choice"]["id"] == "weapon_stick_1"


def test_stick_melee_teacher_recycles_conflicting_found_item():
    state = dict(_state(wave=7), phase="item_found")
    choice = {
        "id": "item_book",
        "base_id": "",
        "category": "item",
        "tier": 0,
        "effects": [
            {"key": "stat_engineering", "value": 2},
            {"key": "stat_elemental_damage", "value": 1},
            {"key": "stat_luck", "value": -1},
        ],
    }
    take = {
        "id": "/root/Main/UI/ItemBoxUI/TakeButton",
        "role": "take_item",
        "enabled": True,
        "choice": choice,
    }
    recycle = {
        "id": "/root/Main/UI/ItemBoxUI/DiscardButton",
        "role": "recycle_item",
        "enabled": True,
        "choice": choice,
    }
    selected = StickMeleeTeacher().select(state, [take, recycle])
    assert selected.action["role"] == "recycle_item"


def test_ui_automation_handles_multiple_upgrades_in_one_wave():
    choice_path = "/root/Main/UI/UpgradesUI/UpgradeUI/ChooseButton"
    go_path = "/root/Shop/GoButton"
    upgrade = dict(_state(wave=3), phase="upgrade", tick=10)
    upgrade["ui"] = {
        "actions": [
            {
                "id": choice_path,
                "name": "ChooseButton",
                "role": "upgrade_choice",
                "enabled": True,
            }
        ],
        "last_result": {},
    }
    second_upgrade_states = []
    for tick in range(11, 16):
        second_upgrade = dict(upgrade, tick=tick)
        second_upgrade["ui"] = dict(upgrade["ui"])
        second_upgrade_states.append(second_upgrade)
    shop = dict(_state(wave=3), phase="shop", tick=16)
    shop["ui"] = {
        "actions": [
            {
                "id": go_path,
                "name": "GoButton",
                "role": "next_wave",
                "enabled": True,
            }
        ],
        "last_result": {"sequence": 7, "ok": True, "changed": True},
    }
    combat = dict(_state(wave=4), tick=17)

    class FakeServer:
        def __init__(self):
            self.states = iter([*second_upgrade_states, shop, combat])
            self.sent = []

        def send(self, message, timeout_sec):
            self.sent.append(message)

        def wait_for_state(self, **_kwargs):
            return next(self.states)

    server = FakeServer()
    result = AutoUiController(max_shop_buys=0, max_shop_rerolls=0).advance(
        server,
        upgrade,
        sequence=5,
        timeout_sec=5,
    )
    assert [message["target"] for message in server.sent] == [
        choice_path,
        choice_path,
        go_path,
    ]
    assert result.confirmed_roles == [
        "upgrade_choice",
        "upgrade_choice",
        "next_wave",
    ]


def test_ui_automation_waits_for_slow_retry_scene_change():
    retry_ok_path = "/root/Main/UI/RetryWave/Menu/OkButton"
    restart_path = "/root/EndRun/RestartButton"
    game_over = dict(_state(wave=3), phase="game_over", tick=10)
    game_over["ui"] = {
        "actions": [
            {
                "id": retry_ok_path,
                "name": "OkButton",
                "role": "restart",
                "enabled": True,
            }
        ],
        "last_result": {},
    }
    end_run = dict(game_over, tick=11)
    end_run["ui"] = {
        "actions": [
            {
                "id": restart_path,
                "name": "RestartButton",
                "role": "restart",
                "enabled": True,
            }
        ],
        "last_result": {},
    }
    loading_states = []
    for tick in range(12, 62):
        loading = dict(game_over, tick=tick)
        loading["ui"] = {
            "actions": [],
            "last_result": {},
        }
        loading_states.append(loading)
    combat = dict(_state(wave=3), tick=62)

    class FakeServer:
        def __init__(self):
            self.states = iter([end_run, *loading_states, combat])
            self.sent = []

        def send(self, message, timeout_sec):
            self.sent.append(message)

        def wait_for_state(self, **_kwargs):
            return next(self.states)

    server = FakeServer()
    result = AutoUiController().advance(
        server,
        game_over,
        sequence=5,
        timeout_sec=30,
        allow_restart=True,
    )
    assert [message["target"] for message in server.sent] == [
        retry_ok_path,
        restart_path,
    ]
    assert result.state["phase"] == "combat"
    assert result.confirmed_roles == ["restart", "restart"]


def test_ui_automation_advances_wave_end_shop_and_next_wave():
    wave_end = dict(_state(wave=3), phase="wave_end", tick=10)
    shop = dict(_state(wave=3, materials=20), phase="shop", tick=11)
    shop["ui"] = {
        "actions": [
            {"id": "/root/Shop/Buy", "role": "buy", "enabled": True},
            {"id": "/root/Shop/Go", "role": "next_wave", "enabled": True},
        ]
    }
    after_buy = dict(shop, tick=12, ui={"actions": shop["ui"]["actions"][1:]})
    shop_closing = dict(after_buy, tick=13)
    combat = dict(_state(wave=4), tick=14)

    class FakeServer:
        def __init__(self):
            self.states = iter([shop, after_buy, shop_closing, combat])
            self.sent = []

        def send(self, message, timeout_sec):
            self.sent.append(message)

        def wait_for_state(self, **_kwargs):
            return next(self.states)

    server = FakeServer()
    result = AutoUiController(max_shop_buys=1, max_shop_rerolls=0).advance(
        server,
        wave_end,
        sequence=5,
        timeout_sec=5,
    )
    assert result.state["phase"] == "combat"
    assert [message["target"] for message in server.sent] == [
        "/root/Shop/Buy",
        "/root/Shop/Go",
    ]
    assert result.sequence == 7
    assert result.sent_roles == ["buy", "next_wave"]
    assert result.confirmed_roles == ["next_wave"]


def test_ui_automation_waits_for_reroll_refresh_before_deciding_again():
    reroll_path = "/root/Shop/Reroll"
    go_path = "/root/Shop/Go"

    def shop_state(tick, *, sequence=-1, changed=False, reroll_enabled=True):
        state = dict(_state(wave=1, materials=50), phase="shop", tick=tick)
        state["ui"] = {
            "actions": [
                {"id": reroll_path, "role": "reroll", "enabled": reroll_enabled},
                {"id": go_path, "role": "next_wave", "enabled": True},
            ],
            "last_result": {
                "sequence": sequence,
                "ok": sequence >= 0,
                "changed": changed,
            },
        }
        return state

    states = iter([
        shop_state(11, sequence=6, reroll_enabled=False),
        shop_state(12, sequence=6, changed=True),
        shop_state(13, sequence=7, reroll_enabled=False),
        shop_state(14, sequence=7, changed=True),
        dict(_state(wave=2), tick=15),
    ])

    class FakeServer:
        def __init__(self):
            self.sent = []

        def send(self, message, timeout_sec):
            self.sent.append(message)

        def wait_for_state(self, **_kwargs):
            return next(states)

    server = FakeServer()
    result = AutoUiController(max_shop_buys=0, max_shop_rerolls=2).advance(
        server,
        shop_state(10),
        sequence=5,
        timeout_sec=5,
    )
    assert [message["target"] for message in server.sent] == [
        reroll_path,
        reroll_path,
        go_path,
    ]
    assert result.confirmed_roles == ["reroll", "reroll", "next_wave"]


def test_ui_automation_tolerates_slow_late_wave_shop_refresh():
    reroll_path = "/root/Shop/Reroll"
    go_path = "/root/Shop/Go"
    initial = dict(_state(wave=13, materials=379), phase="shop", tick=10)
    initial["ui"] = {"actions": [
        {"id": reroll_path, "role": "reroll", "enabled": True},
        {"id": go_path, "role": "next_wave", "enabled": True},
    ]}
    refresh = dict(initial, tick=11)
    refresh["ui"] = {
        "actions": [],
        "last_result": {"sequence": 6, "ok": True, "changed": True},
    }
    blank_states = []
    for tick in range(12, 52):
        blank = dict(initial, tick=tick)
        blank["ui"] = {"actions": [], "last_result": {}}
        blank_states.append(blank)
    ready = dict(initial, tick=52)
    ready["ui"] = {
        "actions": [{"id": go_path, "role": "next_wave", "enabled": True}]
    }
    combat = dict(_state(wave=14), tick=53)

    class FakeServer:
        def __init__(self):
            self.states = iter([refresh, *blank_states, ready, combat])
            self.sent = []

        def send(self, message, timeout_sec):
            self.sent.append(message)

        def wait_for_state(self, **_kwargs):
            return next(self.states)

    server = FakeServer()
    result = AutoUiController(max_shop_buys=0, max_shop_rerolls=1).advance(
        server,
        initial,
        sequence=5,
        timeout_sec=30,
    )
    assert [message["target"] for message in server.sent] == [reroll_path, go_path]
    assert result.state["phase"] == "combat"
    assert result.confirmed_roles == ["reroll", "next_wave"]


def test_wait_for_state_accepts_low_tick_after_reconnect(monkeypatch):
    server = BridgeServer()
    server._connection_generation = 1

    def receive(_timeout):
        server._connection_generation = 2
        return _state()

    monkeypatch.setattr(server, "receive", receive)
    assert server.wait_for_state(after_tick=500)["tick"] == 1


def test_wait_for_state_skips_state_before_action_sequence(monkeypatch):
    server = BridgeServer()
    messages = iter([dict(_state(), sequence=4), dict(_state(), tick=2, sequence=5)])
    monkeypatch.setattr(server, "receive", lambda _timeout: next(messages))
    assert server.wait_for_state(minimum_sequence=5)["tick"] == 2


def test_installer_builds_runtime_zip_and_editable_copy(tmp_path):
    game = tmp_path / "Steam" / "steamapps" / "common" / "Brotato"
    game.mkdir(parents=True)
    (game / "Brotato.exe").touch()
    workshop_host = (
        tmp_path
        / "Steam"
        / "steamapps"
        / "workshop"
        / "content"
        / "1942280"
        / "2931388196"
    )
    workshop_host.mkdir(parents=True)
    (workshop_host / "Subscribed-Mod.zip").touch()
    stale_local = game / "mods" / f"{MOD_DIR_NAME}-0.1.1.zip"
    stale_local.parent.mkdir(parents=True)
    stale_local.touch()
    stale_workshop = workshop_host / f"{MOD_DIR_NAME}-0.1.1.zip"
    stale_workshop.touch()
    package = install_mod(game)
    assert package == game / "mods" / f"{MOD_DIR_NAME}-0.3.6.zip"
    assert (game / "mods-unpacked" / MOD_DIR_NAME / "manifest.json").is_file()
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
    assert f"mods-unpacked/{MOD_DIR_NAME}/manifest.json" in names
    assert f"mods-unpacked/{MOD_DIR_NAME}/mod_main.gd" in names
    workshop_package = (
        tmp_path
        / "Steam"
        / "steamapps"
        / "workshop"
        / "content"
        / "1942280"
        / "2931388196"
        / package.name
    )
    assert workshop_package.read_bytes() == package.read_bytes()
    assert not stale_local.exists()
    assert not stale_workshop.exists()


def test_installer_activates_bridge_in_current_profile(tmp_path):
    profile_path = tmp_path / "Brotato" / "mod_user_profiles.json"
    profile_path.parent.mkdir()
    profile_path.write_text(
        json.dumps(
            {
                "current_profile": "training",
                "profiles": {"training": {"mod_list": {}}},
            }
        ),
        encoding="utf-8",
    )
    package = tmp_path / "1942280" / "2931388196" / "bridge.zip"
    assert activate_mod_profile(profile_path, package)

    saved = json.loads(profile_path.read_text(encoding="utf-8"))
    entry = saved["profiles"]["training"]["mod_list"][MOD_DIR_NAME]
    assert entry == {"is_active": True, "zip_path": package.as_posix()}
    assert profile_path.with_name("mod_user_profiles.json.before-v3.bak").is_file()
