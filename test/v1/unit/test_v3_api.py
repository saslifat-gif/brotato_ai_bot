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
from v3.install_mod import MOD_DIR_NAME, activate_mod_profile, install_mod
from v3.protocol import (
    BridgeProtocolError,
    action_message,
    decode_message,
    encode_message,
    ui_action_message,
)
from v3.reward import ApiRewardEngine
from v3.vectorizer import ApiStateVectorizer, OBSERVATION_SIZE
from v3.ui_automation import AutoUiController, available_actions
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
    assert "func observe_movement_behavior(behavior)" in bridge
    assert "_find_player_descendant(main)" in bridge
    assert 'enemy.connect("died", self, "_on_enemy_died_observed")' in bridge
    assert "func _run_player_data(run_data, player)" in bridge
    assert 'if typeof(object) == TYPE_DICTIONARY:' in bridge
    assert '["gold", "materials"]' in bridge
    assert "func _collect_ui_actions(node, output: Array, phase: String)" in bridge
    assert "func _detect_visible_ui_phase(node)" in bridge
    assert "func _ui_action_context(node)" not in bridge
    assert "_last_ui_action_result" not in bridge
    assert 'button_name == "choosebutton"' in bridge
    assert "visible_ui_phase" in bridge
    assert 'node.emit_signal("pressed")' in bridge
    assert 'str(node.name).to_lower() == "gobutton"' in bridge
    assert '"realtime_control"' in bridge
    assert "const STATE_INTERVAL_SEC := 1.0 / 24.0" in bridge
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
    assert "bridge.observe_movement_behavior(self)" in movement_extension


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
    package = install_mod(game)
    assert package == game / "mods" / f"{MOD_DIR_NAME}-0.1.1.zip"
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
