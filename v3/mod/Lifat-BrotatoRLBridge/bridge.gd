extends Node

const PROTOCOL_VERSION := 1
const MOD_VERSION := "0.1.1"
const HOST := "127.0.0.1"
const PORT := 4242
const RECONNECT_MS := 1000
const ACTION_STALE_MS := 1500
const STATE_INTERVAL_SEC := 1.0 / 15.0
const MAX_ENEMIES := 128
const MAX_PROJECTILES := 128
const MAX_PICKUPS := 64

var _stream: StreamPeerTCP = StreamPeerTCP.new()
var _receive_buffer := ""
var _last_status: int = 0
var _next_connect_ms := 0
var _connected := false
var _state_elapsed := 0.0
var _step_paused := false
var _tick := 0
var _latest_action := 0
var _last_action_ms := 0
var _last_sequence := -1
var _session_id := "%d-%d" % [OS.get_unix_time(), OS.get_ticks_msec()]
var _kills_this_wave := 0
var _last_wave_number := -1
var _reset_kills_on_combat := false


func _ready() -> void:
	set_pause_mode(PAUSE_MODE_PROCESS)
	_next_connect_ms = 0
	print("[BrotatoRLBridge] ready; waiting for trainer at %s:%d" % [HOST, PORT])


func _process(delta: float) -> void:
	_poll_connection()
	if not _connected:
		return
	_read_messages()
	if _step_paused:
		return
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
		_step_paused = false
		get_tree().set_pause(false)
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


func record_enemy_death() -> void:
	_kills_this_wave += 1


func record_player_death() -> void:
	_reset_kills_on_combat = true


func _publish_state() -> void:
	_tick += 1
	var state := _build_state()
	_send(state)
	if _connected and _last_sequence >= 0 and state.get("phase") == "combat":
		_step_paused = true
		get_tree().set_pause(true)


func _resume_game() -> void:
	_step_paused = false
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
	var player_state := _player_state(player)
	var enemies := []
	var projectiles := []
	var pickups := []

	if main != null:
		var spawner = main.get_node_or_null("EntitySpawner")
		var spawned_enemies = _property(spawner, "enemies", [])
		if typeof(spawned_enemies) == TYPE_ARRAY:
			for enemy in spawned_enemies:
				if enemies.size() >= MAX_ENEMIES:
					break
				if is_instance_valid(enemy):
					enemies.append(_entity_state(enemy))
		_append_children(main.get_node_or_null("Projectiles"), projectiles, "projectile", MAX_PROJECTILES)
		_append_children(main.get_node_or_null("Items"), pickups, "item", MAX_PICKUPS)
		_append_children(main.get_node_or_null("Consumables"), pickups, "consumable", MAX_PICKUPS)

	var timer = _property(main, "_wave_timer", null)
	var run_data = root.get_node_or_null("RunData")
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
	var phase := _phase_for_scene(scene_name, player, main, dead, run_won)
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
			"materials": int(_first_property(run_data, ["gold", "materials"], 0)),
			"kills": _kills_this_wave
		},
		"enemies": enemies,
		"projectiles": projectiles,
		"pickups": pickups,
		"dead": dead,
		"victory": run_won
	}


func _phase_for_scene(
	scene_name: String,
	player,
	main,
	dead: bool,
	victory: bool
) -> String:
	if victory:
		return "victory"
	if dead:
		return "game_over"
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
	# Brotato 1.1.x exposes the live player through the TempStats singleton.
	# TempStats is an AutoLoad singleton, not a child named "TempStats" under
	# the current scene root. Access it directly, as Brotato and Brotils do.
	var player = TempStats.player
	# Keep the Main fallbacks for older game builds.
	if player == null:
		player = _property(main, "_player", null)
	if player == null:
		player = _property(main, "player", null)
	if player == null and main != null:
		player = main.get_node_or_null("Player")
	return player


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
	return {
		"position": _vector_json(_property(player, "position", Vector2.ZERO)),
		"velocity": _vector_json(_property(player, "linear_velocity", Vector2.ZERO)),
		"health": float(_property(current_stats, "health", 0.0)),
		"max_health": max(1.0, float(_property(max_stats, "health", 1.0)))
	}


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
		"capabilities": ["structured_state", "movement", "step_pause", "manual_reset"]
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
