extends Node

const PROTOCOL_VERSION := 1
const MOD_VERSION := "0.1.1"
const HOST := "127.0.0.1"
const PORT := 4242
const RECONNECT_MS := 1000
const ACTION_STALE_MS := 1500
const STATE_INTERVAL_SEC := 1.0 / 24.0
const MAX_ENEMIES := 128
const MAX_PROJECTILES := 128
const MAX_PICKUPS := 64
const MAX_UI_ACTIONS := 64

var _stream: StreamPeerTCP = StreamPeerTCP.new()
var _receive_buffer := ""
var _last_status: int = 0
var _next_connect_ms := 0
var _connected := false
var _state_elapsed := 0.0
var _tick := 0
var _latest_action := 0
var _last_action_ms := 0
var _last_sequence := -1
var _session_id := "%d-%d" % [OS.get_unix_time(), OS.get_ticks_msec()]
var _kills_this_wave := 0
var _last_wave_number := -1
var _reset_kills_on_combat := false
var _observed_player = null
var _logged_player_probe := false


func _ready() -> void:
	set_pause_mode(PAUSE_MODE_PROCESS)
	_next_connect_ms = 0
	print("[BrotatoRLBridge] ready; waiting for trainer at %s:%d" % [HOST, PORT])


func _process(delta: float) -> void:
	_poll_connection()
	if not _connected:
		return
	_read_messages()
	_state_elapsed += delta
	if _state_elapsed >= STATE_INTERVAL_SEC:
		_state_elapsed = 0.0
		_publish_state()


func _poll_connection() -> void:
	var status := _stream.get_status()
	if status != _last_status:
		_last_status = status
		if status == _stream.STATUS_CONNECTED:
			_connected = true
			_receive_buffer = ""
			_send_hello()
			print("[BrotatoRLBridge] trainer connected")
		elif status == _stream.STATUS_ERROR or status == _stream.STATUS_NONE:
			if _connected:
				print("[BrotatoRLBridge] trainer disconnected; human control restored")
			_connected = false
			_resume_game()

	if _connected or status == _stream.STATUS_CONNECTING:
		return
	var now := OS.get_ticks_msec()
	if now < _next_connect_ms:
		return
	_stream = StreamPeerTCP.new()
	_last_status = _stream.STATUS_NONE
	_next_connect_ms = now + RECONNECT_MS
	var error := _stream.connect_to_host(HOST, PORT)
	if error != OK:
		_connected = false


func _read_messages() -> void:
	var available := _stream.get_available_bytes()
	if available <= 0:
		return
	var result := _stream.get_partial_data(available)
	if result[0] != OK:
		_connected = false
		_resume_game()
		return
	_receive_buffer += result[1].get_string_from_utf8()
	while true:
		var newline := _receive_buffer.find("\n")
		if newline < 0:
			break
		var line := _receive_buffer.substr(0, newline).strip_edges()
		_receive_buffer = _receive_buffer.substr(newline + 1)
		if not line.empty():
			_handle_message(line)


func _handle_message(line: String) -> void:
	var parsed := JSON.parse(line)
	if parsed.error != OK or typeof(parsed.result) != TYPE_DICTIONARY:
		_send_error("invalid_json")
		return
	var message: Dictionary = parsed.result
	if int(message.get("protocol", -1)) != PROTOCOL_VERSION:
		_send_error("protocol_mismatch")
		return
	var message_type := str(message.get("type", ""))
	if message_type == "action":
		var action := int(message.get("action", 0))
		if action < 0 or action > 8:
			_send_error("invalid_action")
			return
		_latest_action = action
		_last_sequence = int(message.get("sequence", -1))
		_last_action_ms = OS.get_ticks_msec()
		_state_elapsed = 0.0
		_resume_game()
	elif message_type == "reset":
		_latest_action = 0
		_last_action_ms = 0
		_resume_game()
		# Scene-specific automatic reset is intentionally deferred until the
		# installed game version's menu hooks have been verified. The Python
		# environment waits safely while the user starts the next wave/run.
		_send({
			"type": "event",
			"event": "manual_reset_required",
			"sequence": int(message.get("sequence", -1))
		})
	elif message_type == "ui_action":
		_last_sequence = int(message.get("sequence", -1))
		_activate_ui_action(str(message.get("target", "")), _last_sequence)
	else:
		_send_error("unknown_message_type")


func should_control() -> bool:
	return _connected and OS.get_ticks_msec() - _last_action_ms <= ACTION_STALE_MS


func get_movement() -> Vector2:
	match _latest_action:
		1:
			return Vector2(0, -1)
		2:
			return Vector2(0, 1)
		3:
			return Vector2(-1, 0)
		4:
			return Vector2(1, 0)
		5:
			return Vector2(-1, -1).normalized()
		6:
			return Vector2(1, -1).normalized()
		7:
			return Vector2(-1, 1).normalized()
		8:
			return Vector2(1, 1).normalized()
		_:
			return Vector2.ZERO


func observe_movement_behavior(behavior) -> void:
	# This hook runs from the movement behavior owned by the live player. It is
	# more reliable than version-specific Main/TempStats fields, especially in
	# newer co-op-capable builds.
	if behavior == null:
		return
	var candidate = _property(behavior, "player", null)
	if candidate == null:
		candidate = behavior.get_parent()
	while candidate != null:
		if _property(candidate, "current_stats", null) != null:
			_observed_player = candidate
			return
		candidate = candidate.get_parent()


func record_enemy_death() -> void:
	_kills_this_wave += 1


func _observe_enemy_death(enemy) -> void:
	if enemy == null or not enemy.has_signal("died"):
		return
	if not enemy.is_connected("died", self, "_on_enemy_died_observed"):
		enemy.connect("died", self, "_on_enemy_died_observed")


func _on_enemy_died_observed(_enemy, _death_data) -> void:
	record_enemy_death()


func record_player_death() -> void:
	_reset_kills_on_combat = true


func _activate_ui_action(target: String, sequence: int) -> void:
	var root := get_tree().get_root()
	var scene = get_tree().current_scene
	var node = root.get_node_or_null(NodePath(target))
	if node == null or not (node is BaseButton):
		_send_ui_result(sequence, target, false, "target_not_button")
		return
	if scene == null or (node != scene and not scene.is_a_parent_of(node)):
		_send_ui_result(sequence, target, false, "target_outside_scene")
		return
	if bool(_property(node, "disabled", false)) or not node.is_visible_in_tree():
		_send_ui_result(sequence, target, false, "target_unavailable")
		return
	node.emit_signal("pressed")
	_send_ui_result(sequence, target, true, "")
	_state_elapsed = STATE_INTERVAL_SEC


func _send_ui_result(sequence: int, target: String, ok: bool, error: String) -> void:
	_send({
		"type": "event",
		"event": "ui_action_result",
		"sequence": sequence,
		"target": target,
		"ok": ok,
		"error": error
	})


func _publish_state() -> void:
	_tick += 1
	var state := _build_state()
	_send(state)


func _resume_game() -> void:
	get_tree().set_pause(false)


func _exit_tree() -> void:
	_resume_game()


func _build_state() -> Dictionary:
	var root := get_tree().get_root()
	var main = root.get_node_or_null("Main")
	var scene_name := ""
	if get_tree().current_scene != null:
		scene_name = str(get_tree().current_scene.name)
	var player = _find_player(root, main)
	if player != null and not is_instance_valid(player):
		player = null
	if player == null and scene_name.to_lower() == "main" and not _logged_player_probe:
		_logged_player_probe = true
		_log_player_probe(root, main)
	var player_state := _player_state(player)
	var enemies := []
	var projectiles := []
	var pickups := []
	var ui_actions := []

	if main != null:
		var spawner = main.get_node_or_null("EntitySpawner")
		var spawned_enemies = _property(spawner, "enemies", [])
		if typeof(spawned_enemies) == TYPE_ARRAY:
			for enemy in spawned_enemies:
				if enemies.size() >= MAX_ENEMIES:
					break
				if is_instance_valid(enemy):
					_observe_enemy_death(enemy)
					enemies.append(_entity_state(enemy))
		_append_children(main.get_node_or_null("Projectiles"), projectiles, "projectile", MAX_PROJECTILES)
		_append_children(main.get_node_or_null("Items"), pickups, "item", MAX_PICKUPS)
		_append_children(main.get_node_or_null("Consumables"), pickups, "consumable", MAX_PICKUPS)

	var timer = _property(main, "_wave_timer", null)
	var run_data = root.get_node_or_null("RunData")
	var run_player_data = _run_player_data(run_data, player)
	var zone_service = root.get_node_or_null("ZoneService")
	var arena_size = _property(zone_service, "current_zone_max_position", Vector2(1920, 1080))
	if typeof(arena_size) != TYPE_VECTOR2:
		arena_size = Vector2(1920, 1080)
	var wave_number := int(_first_property(run_data, ["current_wave", "wave"], 0))
	var health := float(player_state.get("health", 0.0))
	var run_lost := bool(_property(main, "_is_run_lost", false))
	var run_won := bool(_property(main, "_is_run_won", false)) or bool(
		_property(run_data, "run_won", false)
	)
	var dead: bool = run_lost or (player != null and (
		bool(_property(player, "dead", false)) or health <= 0.0
	))
	# Upgrade and retry overlays live inside Main in current Brotato builds.  A
	# cached player reference can therefore remain valid while one of those
	# overlays is waiting for input.  Detect an actionable visible overlay
	# before using the player reference to call the state combat.
	var visible_ui_phase := _detect_visible_ui_phase(get_tree().current_scene)
	var phase := _phase_for_scene(
		scene_name,
		player,
		main,
		dead,
		run_won,
		visible_ui_phase
	)
	if phase != "combat" and phase != "wave_end":
		_collect_ui_actions(get_tree().current_scene, ui_actions, phase)
	if wave_number != _last_wave_number:
		_last_wave_number = wave_number
		_kills_this_wave = 0
	if _reset_kills_on_combat and phase == "combat":
		_kills_this_wave = 0
		_reset_kills_on_combat = false

	return {
		"type": "state",
		"session": _session_id,
		"tick": _tick,
		"sequence": _last_sequence,
		"phase": phase,
		"scene": scene_name,
		"player": player_state,
		"arena": {"width": arena_size.x, "height": arena_size.y},
		"wave": {
			"number": wave_number,
			"time_left": float(_property(timer, "time_left", 0.0)),
			"duration": float(_property(timer, "wait_time", 60.0))
		},
		"counters": {
			"materials": int(_first_property(
				run_player_data,
				["gold", "materials"],
				_first_property(run_data, ["gold", "materials"], 0)
			)),
			"kills": _kills_this_wave
		},
		"enemies": enemies,
		"projectiles": projectiles,
		"pickups": pickups,
		"ui": {"actions": ui_actions},
		"dead": dead,
		"victory": run_won
	}


func _collect_ui_actions(node, output: Array, phase: String) -> void:
	if node == null or output.size() >= MAX_UI_ACTIONS:
		return
	if node is BaseButton and node.is_visible_in_tree():
		var path := str(node.get_path())
		var text := str(_property(node, "text", "")).strip_edges()
		output.append({
			"id": path,
			"name": str(node.name),
			"text": text,
			"role": _ui_role(node, phase, text),
			"enabled": not bool(_property(node, "disabled", false)),
			"pressed": bool(_property(node, "pressed", false))
		})
	for child in node.get_children():
		if output.size() >= MAX_UI_ACTIONS:
			break
		_collect_ui_actions(child, output, phase)


func _ui_role(node, phase: String, text: String) -> String:
	var token := text.to_lower()
	var cursor = node
	var depth := 0
	while cursor != null and depth < 6:
		token += " " + str(cursor.name).to_lower()
		var script = cursor.get_script()
		if script != null:
			token += " " + str(script.resource_path).to_lower()
		cursor = cursor.get_parent()
		depth += 1
	if phase == "item_found":
		if token.find("recycle") >= 0 or token.find("回收") >= 0:
			return "recycle_item"
		if token.find("take") >= 0 or token.find("keep") >= 0 or token.find("拿取") >= 0:
			return "take_item"
	if token.find("reroll") >= 0 or token.find("刷新") >= 0:
		return "reroll"
	if token.find("next_wave") >= 0 or token.find("next wave") >= 0 or token.find("下一波") >= 0:
		return "next_wave"
	if phase == "shop" and str(node.name).to_lower() == "gobutton":
		return "next_wave"
	if token.find("restart") >= 0 or token.find("retry") >= 0 or token.find("重新开始") >= 0:
		return "restart"
	if token.find("lock") >= 0 or token.find("锁") >= 0:
		return "lock"
	if phase == "upgrade" and str(node.name).to_lower() == "choosebutton" and (
		token.find("upgradeui") >= 0 or token.find("upgrade_ui") >= 0
	):
		return "upgrade_choice"
	if phase == "shop" and token.find("shop_item") >= 0 and token.find("lock") < 0:
		return "buy"
	if token.find("start") >= 0 or token.find("开始") >= 0:
		return "start"
	return "other"


func _detect_visible_ui_phase(node) -> String:
	if node == null:
		return ""
	if node is CanvasItem and not node.is_visible_in_tree():
		return ""
	var token := str(node.name).to_lower() + " " + str(node.get_path()).to_lower()
	var script = node.get_script()
	if script != null:
		token += " " + str(script.resource_path).to_lower()
	if token.find("retrywave") >= 0 or token.find("end_run") >= 0:
		return "game_over"
	if node is BaseButton:
		var button_name := str(node.name).to_lower()
		var button_text := str(_property(node, "text", "")).strip_edges().to_lower()
		if (
			button_name.find("recycle") >= 0
			or button_text.find("recycle") >= 0
			or button_text.find("回收") >= 0
			or button_name.find("take") >= 0
			or button_text == "take"
			or button_text.find("拿取") >= 0
		):
			return "item_found"
		if button_name == "choosebutton" and (
			token.find("upgradeui") >= 0 or token.find("upgrade_ui") >= 0
		):
			return "upgrade"
		if button_name == "gobutton":
			return "shop"
	for child in node.get_children():
		var child_phase := _detect_visible_ui_phase(child)
		if not child_phase.empty():
			return child_phase
	return ""


func _phase_for_scene(
	scene_name: String,
	player,
	main,
	dead: bool,
	victory: bool,
	visible_ui_phase: String
) -> String:
	if victory:
		return "victory"
	if dead:
		return "game_over"
	if not visible_ui_phase.empty():
		return visible_ui_phase
	var lower := scene_name.to_lower()
	if player != null:
		if bool(_property(main, "_cleaning_up", false)):
			return "wave_end"
		return "combat"
	if lower == "main":
		# Main also remains active while a wave is being cleaned up. Without a
		# live TempStats.player, do not report a trainable combat observation.
		return "wave_end"
	if lower.find("shop") >= 0:
		return "shop"
	if lower.find("upgrade") >= 0:
		return "upgrade"
	if lower.find("end_run") >= 0 or lower.find("game_over") >= 0:
		return "game_over"
	return "menu"


func _find_player(root, main):
	if _observed_player != null and is_instance_valid(_observed_player):
		return _observed_player
	# Brotato 1.1.x exposes the live player through the TempStats singleton.
	# TempStats is an AutoLoad singleton, not a child named "TempStats" under
	# the current scene root. Access it directly, as Brotato and Brotils do.
	var player = TempStats.player
	if player == null:
		var temp_players = _first_property(TempStats, ["players", "player_nodes"], [])
		if typeof(temp_players) == TYPE_ARRAY and not temp_players.empty():
			player = temp_players[0]
	# Keep the Main fallbacks for older game builds.
	if player == null:
		player = _first_property(main, ["_player", "player"], null)
	if player == null:
		var main_players = _first_property(main, ["_players", "players"], [])
		if typeof(main_players) == TYPE_ARRAY and not main_players.empty():
			player = main_players[0]
	if player == null and main != null:
		player = main.get_node_or_null("Player")
	if player == null:
		for group_name in ["player", "players"]:
			var grouped = get_tree().get_nodes_in_group(group_name)
			if not grouped.empty():
				player = grouped[0]
				break
	if player == null:
		player = _find_player_descendant(main)
	return player


func _find_player_descendant(node):
	if node == null:
		return null
	for child in node.get_children():
		var script_path := ""
		var script = child.get_script()
		if script != null:
			script_path = str(script.resource_path).to_lower()
		var looks_like_player := str(child.name).to_lower().find("player") >= 0
		looks_like_player = looks_like_player or script_path.find("/player/") >= 0
		looks_like_player = looks_like_player or script_path.ends_with("/player.gd")
		if looks_like_player and _property(child, "current_stats", null) != null:
			return child
		var nested = _find_player_descendant(child)
		if nested != null:
			return nested
	return null


func _log_player_probe(root, main) -> void:
	var root_names := []
	for child in root.get_children():
		root_names.append(str(child.name))
	print("[BrotatoRLBridge] player lookup pending; root_children=%s" % [root_names])
	print("[BrotatoRLBridge] TempStats player fields=%s" % [
		_matching_property_names(TempStats, "player")
	])
	print("[BrotatoRLBridge] Main player fields=%s" % [
		_matching_property_names(main, "player")
	])


func _matching_property_names(object, needle: String) -> Array:
	var matches := []
	if object == null:
		return matches
	for descriptor in object.get_property_list():
		var property_name := str(descriptor.get("name", ""))
		if property_name.to_lower().find(needle) >= 0:
			matches.append(property_name)
	return matches


func _player_state(player) -> Dictionary:
	if player == null:
		return {
			"position": _vector_json(Vector2.ZERO),
			"velocity": _vector_json(Vector2.ZERO),
			"health": 0.0,
			"max_health": 1.0
		}
	var current_stats = _property(player, "current_stats", null)
	var max_stats = _property(player, "max_stats", null)
	var health = _first_property(current_stats, ["health", "current_health", "hp"], null)
	if health == null:
		health = _first_property(player, ["health", "current_health", "hp"], 0.0)
	var max_health = _first_property(max_stats, ["health", "max_health", "hp"], null)
	if max_health == null:
		max_health = _first_property(current_stats, ["max_health", "health_max"], null)
	if max_health == null:
		max_health = _first_property(player, ["max_health", "health_max", "max_hp"], 1.0)
	return {
		"position": _vector_json(_property(player, "position", Vector2.ZERO)),
		"velocity": _vector_json(_property(player, "linear_velocity", Vector2.ZERO)),
		"health": float(health),
		"max_health": max(1.0, float(max_health))
	}


func _run_player_data(run_data, player):
	# Brotato 1.1.15 stores currency and other run values per player. Preserve
	# the old global fallback in _build_state for earlier single-player builds.
	var players_data = _property(run_data, "players_data", [])
	if typeof(players_data) != TYPE_ARRAY or players_data.empty():
		return null
	var player_index := int(_first_property(
		player,
		["player_index", "player_id", "index"],
		0
	))
	player_index = int(clamp(player_index, 0, players_data.size() - 1))
	return players_data[player_index]


func _entity_state(entity) -> Dictionary:
	var current_stats = _property(entity, "current_stats", null)
	var max_stats = _property(entity, "max_stats", null)
	return {
		"position": _vector_json(_property(entity, "position", Vector2.ZERO)),
		"velocity": _vector_json(_property(entity, "linear_velocity", Vector2.ZERO)),
		"health": float(_property(current_stats, "health", 1.0)),
		"max_health": max(1.0, float(_property(max_stats, "health", 1.0)))
	}


func _append_children(container, output: Array, kind: String, maximum: int) -> void:
	if container == null:
		return
	for child in container.get_children():
		if output.size() >= maximum:
			break
		output.append({
			"position": _vector_json(_property(child, "position", Vector2.ZERO)),
			"velocity": _vector_json(_property(child, "linear_velocity", Vector2.ZERO)),
			"rotation": float(_property(child, "rotation", 0.0)),
			"kind": kind
		})


func _property(object, property_name: String, fallback):
	if object == null:
		return fallback
	if typeof(object) == TYPE_DICTIONARY:
		var dictionary: Dictionary = object
		var dictionary_value = dictionary.get(property_name, fallback)
		return fallback if dictionary_value == null else dictionary_value
	for descriptor in object.get_property_list():
		if descriptor.has("name") and str(descriptor["name"]) == property_name:
			var value = object.get(property_name)
			return fallback if value == null else value
	return fallback


func _first_property(object, names: Array, fallback):
	for property_name in names:
		var value = _property(object, str(property_name), null)
		if value != null:
			return value
	return fallback


func _vector_json(value) -> Dictionary:
	if typeof(value) != TYPE_VECTOR2:
		value = Vector2.ZERO
	return {"x": float(value.x), "y": float(value.y)}


func _send_hello() -> void:
	_send({
		"type": "hello",
		"session": _session_id,
		"mod_version": MOD_VERSION,
		"game_version": str(ProjectSettings.get_setting("application/config/version")),
		"capabilities": ["structured_state", "movement", "realtime_control", "ui_actions"]
	})


func _send_error(code: String) -> void:
	_send({"type": "error", "code": code})


func _send(payload: Dictionary) -> void:
	if not _connected:
		return
	payload["protocol"] = PROTOCOL_VERSION
	var data := (to_json(payload) + "\n").to_utf8()
	if _stream.put_data(data) != OK:
		_connected = false
		_resume_game()
