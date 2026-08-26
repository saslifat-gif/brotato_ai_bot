extends Node

const PROTOCOL_VERSION := 1
const MOD_VERSION := "0.3.17"
const HOST := "127.0.0.1"
const PORT := 4242
const RAW_RECORD_PORT := 4243
const RAW_RECORD_HZ := 60.0
const RECONNECT_MS := 1000
const ACTION_STALE_MS := 1500
const DEFAULT_STATE_HZ := 24.0
# The rich policy consumes at most 20 enemies, 20 projectiles and 8 pickups.
# Keep generous headroom without reflecting and serializing hundreds of nodes
# on Godot's main thread every frame in dense late waves.
const MAX_ENEMIES := 64
const MAX_PROJECTILES := 64
const MAX_PICKUPS := 32
const MAX_ATTACK_INDICATORS := 32
const MAX_COMBAT_WEAPONS := 6
const MAX_INDICATOR_SCAN_NODES := 500
const MAX_UI_ACTIONS := 64
const MAX_BUILD_ITEMS := 128
const FULL_ARENA_GRID_COLUMNS := 10
const FULL_ARENA_GRID_ROWS := 6
const BULLET_GRID_COLUMNS := 20
const BULLET_GRID_ROWS := 12
const BULLET_GRID_CHANNELS := 10
const BULLET_GRID_WIDTH := 1600.0
const BULLET_GRID_HEIGHT := 900.0
const BULLET_PATH_HORIZONS := [0.0, 0.25, 0.5, 0.75, 1.0]
# Path risk is deliberately refreshed less often than the control state.  The
# full state still publishes every requested tick, while this bounded forecast
# is allowed to be up to ~167 ms old at 24 Hz.
const PROJECTILE_PATH_REFRESH_TICKS := 4
const ARENA_GRID_REFRESH_TICKS := 2
const BULLET_ACTION_VECTORS := [
	Vector2.ZERO,
	Vector2(0, -1),
	Vector2(0, 1),
	Vector2(-1, 0),
	Vector2(1, 0),
	Vector2(-0.70710678, -0.70710678),
	Vector2(0.70710678, -0.70710678),
	Vector2(-0.70710678, 0.70710678),
	Vector2(0.70710678, 0.70710678)
]
const BUILD_STAT_KEYS := [
	"stat_max_hp",
	"stat_armor",
	"stat_crit_chance",
	"stat_luck",
	"stat_attack_speed",
	"stat_elemental_damage",
	"stat_hp_regeneration",
	"stat_lifesteal",
	"stat_melee_damage",
	"stat_percent_damage",
	"stat_dodge",
	"stat_engineering",
	"stat_range",
	"stat_ranged_damage",
	"stat_speed",
	"stat_harvesting"
]

var _stream: StreamPeerTCP = StreamPeerTCP.new()
var _raw_server: TCP_Server = TCP_Server.new()
var _raw_stream: StreamPeerTCP = null
var _raw_state_elapsed := 0.0
var _raw_record_connected := false
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
var _entity_static_cache := {}
var _projectile_static_cache := {}
var _pickup_static_cache := {}
var _latest_human_action := 0
var _last_human_input_ms := 0
var _last_attack_indicators := []
var _indicator_scan_nodes := 0
var _indicator_scan_seen := {}
var _projectile_scan_nodes := 0
var _logged_semantic_probes := {}
var _last_indicator_scan_tick := -999999
var _last_projectile_path_tick := -999999
var _last_projectile_paths := {}
var _last_arena_grid_tick := -999999
var _last_arena_enemy_grid := []
var _last_state_profile_tick := -999999
var _requested_state_hz := 24.0
var _property_name_cache := {}
var _training_paused := false
var _wave_restart_state = null
var _wave_restart_number := -1

const BRIDGE_RESTART_WAVE_ACTION := "bridge://restart_wave"


func _ready() -> void:
	set_pause_mode(PAUSE_MODE_PROCESS)
	_next_connect_ms = 0
	var raw_error := _raw_server.listen(RAW_RECORD_PORT, HOST)
	if raw_error == OK:
		print("[BrotatoRLBridge] raw recorder listening on %s:%d" % [HOST, RAW_RECORD_PORT])
	else:
		print("[BrotatoRLBridge] raw recorder disabled; listen error=%d" % raw_error)
	print("[BrotatoRLBridge] ready; waiting for trainer at %s:%d" % [HOST, PORT])


func _process(delta: float) -> void:
	_poll_raw_recorder()
	_raw_state_elapsed += delta
	if _raw_state_elapsed >= 1.0 / RAW_RECORD_HZ:
		_raw_state_elapsed = fmod(_raw_state_elapsed, 1.0 / RAW_RECORD_HZ)
		_publish_raw_state()
	_poll_connection()
	if not _connected:
		return
	_read_messages()
	if _training_paused:
		return
	_state_elapsed += delta
	var interval := _state_interval_sec()
	if _state_elapsed >= interval:
		_state_elapsed = max(0.0, _state_elapsed - interval)
		_publish_state()

func _state_interval_sec() -> float:
	# Respect the trainer's requested control rate at every wave. The previous
	# adaptive cap silently reduced late-wave control to 12 Hz exactly when boss
	# projectiles require the fastest reaction loop.
	return 1.0 / max(4.0, _requested_state_hz)


func _poll_connection() -> void:
	var status := _stream.get_status()
	if status != _last_status:
		_last_status = status
		if status == _stream.STATUS_CONNECTED:
			_connected = true
			_requested_state_hz = 24.0
			_receive_buffer = ""
			_send_hello()
			print("[BrotatoRLBridge] trainer connected")
		elif status == _stream.STATUS_ERROR or status == _stream.STATUS_NONE:
			if _connected:
				print("[BrotatoRLBridge] trainer disconnected; human control restored")
			_connected = false
			_training_paused = false
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


func _poll_raw_recorder() -> void:
	if _raw_server.is_connection_available():
		if _raw_record_connected and _raw_stream != null:
			_raw_stream.disconnect_from_host()
		_raw_stream = _raw_server.take_connection()
		_raw_record_connected = _raw_stream != null
		if _raw_record_connected:
			print("[BrotatoRLBridge] raw recorder connected")
	if not _raw_record_connected or _raw_stream == null:
		return
	if _raw_stream.get_status() != _raw_stream.STATUS_CONNECTED:
		_raw_record_connected = false
		_raw_stream = null
		print("[BrotatoRLBridge] raw recorder disconnected")


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
		_training_paused = false
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
	elif message_type == "configure":
		_requested_state_hz = clamp(float(message.get("state_hz", 24.0)), 4.0, 24.0)
		_send({
			"type": "event",
			"event": "configured",
			"state_hz": _requested_state_hz
		})
	elif message_type == "training_pause":
		_training_paused = bool(message.get("paused", false))
		if _training_paused:
			get_tree().set_pause(true)
		else:
			# Give the next inference step a full stale-action window. An action
			# message also clears this pause as a fail-safe.
			_last_action_ms = OS.get_ticks_msec()
			_resume_game()
		_send({
			"type": "event",
			"event": "training_pause",
			"paused": _training_paused
		})
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


func observe_movement_behavior(behavior, human_movement = Vector2.ZERO) -> void:
	# This hook runs from the movement behavior owned by the live player. It is
	# more reliable than version-specific Main/TempStats fields, especially in
	# newer co-op-capable builds.
	if behavior == null:
		return
	_latest_human_action = _movement_to_action(human_movement)
	_last_human_input_ms = OS.get_ticks_msec()
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
	if target == BRIDGE_RESTART_WAVE_ACTION:
		_restart_saved_wave(sequence, target)
		return
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
	_state_elapsed = _state_interval_sec()


func _restart_saved_wave(sequence: int, target: String) -> void:
	if _wave_restart_state == null or _wave_restart_number <= 1:
		_send_ui_result(sequence, target, false, "restart_state_unavailable")
		return
	var restored_state = _wave_restart_state.duplicate(true)
	ProgressData.current_run_state = restored_state
	RunData.resume_from_state(restored_state)
	_resume_game()
	var error = get_tree().change_scene(MenuData.shop_scene)
	if error != OK:
		_send_ui_result(sequence, target, false, "change_scene_%d" % error)
		return
	_latest_action = 0
	_last_action_ms = 0
	_reset_kills_on_combat = true
	_send_ui_result(sequence, target, true, "")
	_state_elapsed = _state_interval_sec()
	print("[BrotatoRLBridge] restored shop before wave %d" % _wave_restart_number)


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
	var started_ms := OS.get_ticks_msec()
	_tick += 1
	var state := _build_state()
	_send(state)
	var elapsed_ms := OS.get_ticks_msec() - started_ms
	if elapsed_ms >= 30 and _tick - _last_state_profile_tick >= 24:
		_last_state_profile_tick = _tick
		print(
			"[BrotatoRLBridge] slow_state_ms=%d enemies=%d projectiles=%d indicators=%d" % [
				elapsed_ms,
				state.get("enemies", []).size(),
				state.get("projectiles", []).size(),
				state.get("attack_indicators", []).size()
			]
		)


func _publish_raw_state() -> void:
	if not _raw_record_connected or _raw_stream == null:
		return
	var payload := _build_raw_state()
	payload["protocol"] = PROTOCOL_VERSION
	var data := (to_json(payload) + "\n").to_utf8()
	if _raw_stream.put_data(data) != OK:
		_raw_record_connected = false
		_raw_stream = null


func _build_raw_state() -> Dictionary:
	var root := get_tree().get_root()
	var main = root.get_node_or_null("Main")
	var player = _find_player(root, main)
	var enemies := []
	var spawner = main.get_node_or_null("EntitySpawner") if main != null else null
	var spawned = _property(spawner, "enemies", [])
	if typeof(spawned) == TYPE_ARRAY:
		for enemy in spawned:
			if enemies.size() >= MAX_ENEMIES or not is_instance_valid(enemy):
				break
			enemies.append({
				"runtime_id": str(enemy.get_instance_id()),
				"position": _vector_json(_first_property(enemy, ["global_position", "position"], Vector2.ZERO)),
				"velocity": _vector_json(_property(enemy, "linear_velocity", Vector2.ZERO)),
				"boss": bool(_property(enemy, "is_boss", false))
			})
	var player_state := _player_state(player)
	var wave_number := int(_first_property(root.get_node_or_null("RunData"), ["current_wave", "wave"], 0))
	return {
		"type": "raw_state",
		"session": _session_id,
		"tick": _tick,
		"published_at_ms": OS.get_ticks_msec(),
		"action": _latest_action,
		"action_sequence": _last_sequence,
		"phase": str(get_tree().current_scene.name) if get_tree().current_scene != null else "",
		"wave": wave_number,
		"player": player_state,
		"enemies": enemies
	}


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
	var projectile_nodes := []
	var pickups := []
	var attack_indicators := []
	var ui_actions := []
	var spawned_enemies := []
	var zone_service = root.get_node_or_null("ZoneService")
	var arena_size = _property(zone_service, "current_zone_max_position", Vector2(1920, 1080))
	if typeof(arena_size) != TYPE_VECTOR2:
		arena_size = Vector2(1920, 1080)

	if main != null:
		var spawner = main.get_node_or_null("EntitySpawner")
		spawned_enemies = _property(spawner, "enemies", [])
		if typeof(spawned_enemies) == TYPE_ARRAY:
			for enemy in spawned_enemies:
				if enemies.size() >= MAX_ENEMIES:
					break
				if is_instance_valid(enemy):
					_observe_enemy_death(enemy)
					enemies.append(_entity_state(enemy))
		_collect_projectiles(main, projectiles, MAX_PROJECTILES, projectile_nodes)
		_append_pickups(main.get_node_or_null("Items"), pickups, "item", MAX_PICKUPS)
		_append_pickups(main.get_node_or_null("Consumables"), pickups, "consumable", MAX_PICKUPS)

	var timer = _property(main, "_wave_timer", null)
	var run_data = root.get_node_or_null("RunData")
	var run_player_data = _run_player_data(run_data, player)
	var build_state := {}
	var combat_state := {}
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
	if phase == "shop":
		_capture_wave_restart_state(wave_number)
	if phase != "combat" and phase != "wave_end":
		build_state = _build_state_for_policy(run_data, run_player_data)
		if phase == "game_over" and _wave_restart_state != null and _wave_restart_number > 1:
			ui_actions.append({
				"id": BRIDGE_RESTART_WAVE_ACTION,
				"name": "BridgeRestartWave",
				"text": "Restart wave",
				"role": "restart",
				"enabled": true,
				"pressed": false
			})
		_collect_ui_actions(get_tree().current_scene, ui_actions, phase)
	elif phase == "combat":
		combat_state = _combat_summary(player, run_data)
		# Warning-node discovery is more expensive than the normal entity export.
		# Four scans/second at early-wave rate is responsive without worsening the
		# late-wave frame drops that motivated the adaptive state interval.
		if _tick - _last_indicator_scan_tick >= 6:
			_last_indicator_scan_tick = _tick
			_last_attack_indicators = _collect_attack_indicators(main, projectile_nodes)
		attack_indicators = _last_attack_indicators.duplicate(true)
	else:
		_last_attack_indicators = []
		_last_indicator_scan_tick = -999999
		_last_projectile_paths = {}
		_last_projectile_path_tick = -999999
		_last_arena_enemy_grid = []
		_last_arena_grid_tick = -999999
	if wave_number != _last_wave_number:
		_last_wave_number = wave_number
		_kills_this_wave = 0
	if _reset_kills_on_combat and phase == "combat":
		_kills_this_wave = 0
		_reset_kills_on_combat = false
	var arena_enemy_grid := _cached_arena_enemy_grid(
		spawned_enemies,
		arena_size,
		player_state
	)
	return {
		"type": "state",
		"session": _session_id,
		"tick": _tick,
		"published_at_ms": OS.get_ticks_msec(),
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
		"arena_grid": {
			"enemy": arena_enemy_grid
		},
		"projectiles": projectiles,
		"projectile_paths": _cached_projectile_path_state(
			projectile_nodes,
			spawned_enemies,
			player_state,
			combat_state,
			arena_size
		),
		"pickups": pickups,
		"attack_indicators": attack_indicators,
		"combat": combat_state,
		"human_action": _latest_human_action,
		"human_input_age_ms": max(0, OS.get_ticks_msec() - _last_human_input_ms),
		"build": build_state,
		"ui": {"actions": ui_actions},
		"dead": dead,
		"victory": run_won
	}


func _capture_wave_restart_state(wave_number: int) -> void:
	# Brotato stores the build entering the next wave in current_run_state while
	# the shop is open. Keep an in-memory deep copy because vanilla game-over
	# clears ProgressData before the trainer can press a retry button.
	var next_wave := wave_number + 1
	if next_wave <= 1:
		return
	if _wave_restart_number == next_wave and _wave_restart_state != null:
		return
	# Current Brotato keeps the authoritative shop-boundary state on disk and
	# may leave the singleton's current_run_state empty until explicitly loaded.
	# Loading it here changes only ProgressData; the live shop remains in RunData.
	ProgressData.load_game_file()
	var current_state = ProgressData.current_run_state
	if current_state == null:
		print("[BrotatoRLBridge] no saved run state before wave %d" % next_wave)
		return
	_wave_restart_state = current_state.duplicate(true)
	_wave_restart_number = next_wave
	print("[BrotatoRLBridge] captured shop before wave %d" % _wave_restart_number)


func _full_arena_enemy_grid(spawned_enemies, arena_size: Vector2, player_state: Dictionary) -> Array:
	# Unlike the detailed entity list, this inexpensive aggregate includes every
	# live enemy. It gives the policy whole-arena density and motion without
	# serializing hundreds of reflected objects in late waves.
	var output := []
	for _index in range(FULL_ARENA_GRID_COLUMNS * FULL_ARENA_GRID_ROWS * 4):
		output.append(0.0)
	if typeof(spawned_enemies) != TYPE_ARRAY:
		return output
	var width := max(1.0, arena_size.x)
	var height := max(1.0, arena_size.y)
	var max_health := max(1.0, float(player_state.get("max_health", 1.0)))
	for enemy in spawned_enemies:
		if enemy == null or not is_instance_valid(enemy):
			continue
		var position = _property(enemy, "position", Vector2.ZERO)
		if typeof(position) != TYPE_VECTOR2:
			continue
		var column := int(clamp(
			floor(position.x / width * FULL_ARENA_GRID_COLUMNS),
			0,
			FULL_ARENA_GRID_COLUMNS - 1
		))
		var row := int(clamp(
			floor(position.y / height * FULL_ARENA_GRID_ROWS),
			0,
			FULL_ARENA_GRID_ROWS - 1
		))
		var offset := (row * FULL_ARENA_GRID_COLUMNS + column) * 4
		var current_stats = _property(enemy, "current_stats", null)
		var static_data := _entity_static_data(enemy)
		var radius := max(1.0, float(static_data.get("radius", 40.0)))
		var damage := max(0.0, float(_first_property(
			current_stats,
			["damage", "contact_damage", "touch_damage"],
			_first_property(enemy, ["damage", "contact_damage", "touch_damage"], 0.0)
		)))
		var velocity = _property(enemy, "linear_velocity", Vector2.ZERO)
		if typeof(velocity) != TYPE_VECTOR2:
			velocity = Vector2.ZERO
		output[offset] += 1.0 / 16.0
		output[offset + 1] += min(1.0, radius / 300.0 + damage / max_health) / 8.0
		output[offset + 2] += clamp(velocity.x / 1000.0, -1.0, 1.0) / 8.0
		output[offset + 3] += clamp(velocity.y / 1000.0, -1.0, 1.0) / 8.0
	for index in range(output.size()):
		output[index] = clamp(float(output[index]), -1.0, 1.0)
	return output


func _cached_arena_enemy_grid(spawned_enemies, arena_size: Vector2, player_state: Dictionary) -> Array:
	if _tick - _last_arena_grid_tick >= ARENA_GRID_REFRESH_TICKS:
		_last_arena_enemy_grid = _full_arena_enemy_grid(
			spawned_enemies,
			arena_size,
			player_state
		)
		_last_arena_grid_tick = _tick
	return _last_arena_enemy_grid


func _collect_ui_actions(node, output: Array, phase: String) -> void:
	if node == null or output.size() >= MAX_UI_ACTIONS:
		return
	if node is BaseButton and node.is_visible_in_tree():
		var path := str(node.get_path())
		var text := str(_property(node, "text", "")).strip_edges()
		var role := _ui_role(node, phase, text)
		var action = {
			"id": path,
			"name": str(node.name),
			"text": text,
			"role": role,
			"enabled": not bool(_property(node, "disabled", false)),
			"pressed": bool(_property(node, "pressed", false))
		}
		var choice = _ui_choice_data(node, phase, role)
		if not choice.empty():
			action["choice"] = choice
			if role == "buy" and choice.has("affordable"):
				action["enabled"] = bool(action["enabled"]) and bool(choice["affordable"])
		# Inventory and category controls can exceed the export cap in late
		# shops, hiding GoButton even though it is visible. Only advertise roles
		# the automation protocol understands.
		if role != "other":
			output.append(action)
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
	if (
		token.find("merge") >= 0
		or token.find("combine") >= 0
		or token.find("fuse") >= 0
		or token.find("合并") >= 0
		or token.find("融合") >= 0
	):
		return "merge"
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


func _ui_choice_data(node, phase: String, role: String) -> Dictionary:
	var property_name := ""
	if role == "buy":
		property_name = "item_data"
	elif role == "upgrade_choice":
		property_name = "upgrade_data"
	elif role == "take_item" or role == "recycle_item":
		property_name = "item_data"
	else:
		return {}

	var cursor = node
	var depth := 0
	while cursor != null and depth < 12:
		var data = _property(cursor, property_name, null)
		if data != null:
			var price := int(_property(cursor, "value", _property(data, "value", 0)))
			var result := _item_policy_data(data, price)
			if role == "buy":
				result["affordable"] = price <= _current_materials()
				result["locked"] = bool(_property(cursor, "locked", false))
			return result
		cursor = cursor.get_parent()
		depth += 1
	return {}


func _item_policy_data(data, price: int = -1) -> Dictionary:
	if data == null:
		return {}
	var item_id := str(_property(data, "my_id", ""))
	var weapon_id := str(_property(data, "weapon_id", ""))
	var upgrade_id := str(_property(data, "upgrade_id", ""))
	var name_key := str(_property(data, "name", ""))
	var category := "item"
	if not weapon_id.empty():
		category = "weapon"
	elif not upgrade_id.empty():
		category = "upgrade"
	var effects := []
	var raw_effects = _property(data, "effects", [])
	if typeof(raw_effects) == TYPE_ARRAY:
		for effect in raw_effects:
			if effects.size() >= 32:
				break
			if effect != null:
				effects.append(_effect_policy_data(effect))
	var tags := []
	var raw_tags = _property(data, "tags", [])
	if typeof(raw_tags) == TYPE_ARRAY:
		for tag in raw_tags:
			tags.append(str(tag))
	var sets := []
	var raw_sets = _property(data, "sets", [])
	if typeof(raw_sets) == TYPE_ARRAY:
		for set_data in raw_sets:
			sets.append(str(_first_property(set_data, ["my_id", "set_id", "name"], "")))
	return {
		"id": item_id,
		"base_id": weapon_id if not weapon_id.empty() else upgrade_id,
		"name_key": name_key,
		"display_name": tr(name_key) if not name_key.empty() else item_id,
		"category": category,
		"tier": int(_property(data, "tier", 0)),
		"base_value": int(_property(data, "value", 0)),
		"price": price,
		"weapon_type": int(_property(data, "type", -1)) if category == "weapon" else -1,
		"tags": tags,
		"sets": sets,
		"effects": effects
	}


func _effect_policy_data(effect) -> Dictionary:
	var result := {}
	for property_name in [
		"key",
		"custom_key",
		"text_key",
		"value",
		"stat_scaled",
		"nb_stat_scaled",
		"to_stat",
		"to_value",
		"stat_name",
		"weapon_id"
	]:
		var value = _property(effect, property_name, null)
		if value != null and typeof(value) in [TYPE_BOOL, TYPE_INT, TYPE_REAL, TYPE_STRING]:
			result[property_name] = value
	return result


func _current_materials() -> int:
	var observed_player = null
	if _observed_player != null and is_instance_valid(_observed_player):
		observed_player = _observed_player
	var run_player_data = _run_player_data(RunData, observed_player)
	return int(_first_property(
		run_player_data,
		["gold", "materials"],
		_first_property(RunData, ["gold", "materials"], 0)
	))


func _build_state_for_policy(run_data, run_player_data) -> Dictionary:
	var weapons := []
	var raw_weapons = _first_property(
		run_player_data,
		["weapons"],
		_property(run_data, "weapons", [])
	)
	if typeof(raw_weapons) == TYPE_ARRAY:
		for weapon in raw_weapons:
			if weapons.size() >= MAX_BUILD_ITEMS:
				break
			weapons.append(_item_policy_data(weapon))
	var items := []
	var raw_items = _first_property(
		run_player_data,
		["items"],
		_property(run_data, "items", [])
	)
	if typeof(raw_items) == TYPE_ARRAY:
		for item in raw_items:
			if items.size() >= MAX_BUILD_ITEMS:
				break
			items.append(_item_policy_data(item))
	var stats := {}
	if run_data != null and run_data.has_method("get_stat"):
		for stat_key in BUILD_STAT_KEYS:
			stats[stat_key] = float(run_data.call("get_stat", stat_key))
	var character = _property(run_data, "current_character", null)
	return {
		"character_id": str(_property(character, "my_id", "")),
		"weapons": weapons,
		"items": items,
		"stats": stats
	}


func _combat_summary(player, run_data) -> Dictionary:
	if player == null:
		return {}
	var current_stats = _property(player, "current_stats", null)
	var weapons = _property(player, "current_weapons", [])
	var weapon_count := 0
	var melee_count := 0
	var ranged_count := 0
	var weapon_range := 170.0
	var range_seen := false
	var weapon_states := []
	if typeof(weapons) == TYPE_ARRAY:
		for weapon in weapons:
			if weapon == null:
				continue
			if typeof(weapon) == TYPE_OBJECT and not is_instance_valid(weapon):
				continue
			weapon_count += 1
			var weapon_stats = _property(weapon, "current_stats", null)
			var current_range := float(_first_property(
				weapon_stats,
				["max_range", "range"],
				170.0
			))
			if not range_seen or current_range < weapon_range:
				weapon_range = current_range
				range_seen = true
			var weapon_token := _script_token(weapon) + " " + _script_token(weapon_stats)
			if weapon_token.find("ranged") >= 0 or weapon_token.find("projectile") >= 0:
				ranged_count += 1
			else:
				melee_count += 1
			if weapon_states.size() < MAX_COMBAT_WEAPONS:
				weapon_states.append(_combat_weapon_state(weapon, weapon_stats))
	var armor := 0.0
	var attack_speed := 0.0
	var speed_stat := 0.0
	if run_data != null and run_data.has_method("get_stat"):
		armor = float(run_data.call("get_stat", "stat_armor"))
		attack_speed = float(run_data.call("get_stat", "stat_attack_speed"))
		speed_stat = float(run_data.call("get_stat", "stat_speed"))
	var move_speed := float(_first_property(
		player,
		["current_speed", "movement_speed", "move_speed", "speed"],
		300.0 * max(0.1, 1.0 + speed_stat / 100.0)
	))
	var character = _property(run_data, "current_character", null)
	return {
		"character_id": str(_property(character, "my_id", "")),
		"weapon_count": weapon_count,
		"melee_count": melee_count,
		"ranged_count": ranged_count,
		"weapon_range": weapon_range,
		"move_speed": move_speed,
		"armor": armor,
		"attack_speed": attack_speed,
		"dodge": float(_first_property(current_stats, ["dodge"], 0.0)),
		"weapons": weapon_states
	}


func _combat_weapon_state(weapon, weapon_stats) -> Dictionary:
	var weapon_data = _first_property(
		weapon,
		["_weapon_data", "weapon_data", "_item_data", "item_data", "_data", "data"],
		null
	)
	_log_semantic_probe("weapon", weapon, weapon_data)
	var token := _script_token(weapon) + " " + _script_token(weapon_data)
	var cooldown_timer = _first_property(
		weapon,
		["_cooldown_timer", "cooldown_timer", "_attack_timer", "attack_timer"],
		null
	)
	var reload_timer = _first_property(
		weapon,
		["_reload_timer", "reload_timer"],
		null
	)
	if cooldown_timer == null:
		cooldown_timer = _find_named_timer(weapon, ["cooldown", "attack"])
	if reload_timer == null:
		reload_timer = _find_named_timer(weapon, ["reload"])
	var cooldown_remaining := float(_first_property(
		cooldown_timer,
		["time_left"],
		_first_property(
			weapon,
			["_current_cooldown", "cooldown_remaining", "attack_cooldown", "_cooldown"],
			0.0
		)
	))
	var cooldown_duration := float(_first_property(
		cooldown_timer,
		["wait_time"],
		_first_property(weapon_stats, ["cooldown", "attack_cooldown", "cooldown_duration"], 0.0)
	))
	var reload_remaining := float(_first_property(reload_timer, ["time_left"], 0.0))
	var ammo := int(_first_property(
		weapon,
		["current_ammo", "ammo", "ammo_count", "shots_remaining"],
		_first_property(weapon_stats, ["current_ammo", "ammo", "ammo_count"], -1)
	))
	var ammo_capacity := int(_first_property(
		weapon,
		["max_ammo", "ammo_capacity", "magazine_size"],
		_first_property(weapon_stats, ["max_ammo", "ammo_capacity", "magazine_size"], -1)
	))
	var reloading := reload_remaining > 0.001 or bool(_first_property(
		weapon,
		["is_reloading", "reloading", "_is_reloading"],
		false
	))
	return {
		"id": _semantic_id(weapon, weapon_data),
		"attack_type": token,
		"range": float(_first_property(weapon_stats, ["max_range", "range"], 170.0)),
		"cooldown_remaining": max(0.0, cooldown_remaining),
		"cooldown_duration": max(0.0, cooldown_duration),
		"reload_remaining": max(0.0, reload_remaining),
		"ammo": ammo,
		"ammo_capacity": ammo_capacity,
		"is_reloading": reloading,
		"is_attacking": bool(_first_property(
			weapon,
			["_is_shooting", "is_shooting", "is_attacking", "attacking", "_is_attacking"],
			false
		)),
		"ready": cooldown_remaining <= 0.001 and not reloading and ammo != 0,
		"rotation": float(_property(weapon, "global_rotation", _property(weapon, "rotation", 0.0)))
	}


func _find_named_timer(node, terms: Array):
	if node == null:
		return null
	for child in node.get_children():
		if child is Timer:
			var token := str(child.name).to_lower()
			for term in terms:
				if token.find(str(term)) >= 0:
					return child
	return null


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
			"max_health": 1.0,
			"radius": 28.0,
			"width": 56.0,
			"height": 56.0,
			"shape": "unknown",
			"size_known": false
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
	var shape_data := _collision_shape_data(player)
	return {
		"position": _vector_json(_first_property(
			player,
			["global_position", "position"],
			Vector2.ZERO
		)),
		"velocity": _vector_json(_property(player, "linear_velocity", Vector2.ZERO)),
		"health": float(health),
		"max_health": max(1.0, float(max_health)),
		"radius": shape_data["radius"],
		"width": shape_data["width"],
		"height": shape_data["height"],
		"shape": shape_data["shape"],
		"size_known": shape_data["known"]
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
	var static_data := _entity_static_data(entity)
	var attack_behavior = _first_property(
		entity,
		["_current_attack_behavior", "current_attack_behavior", "_attack_behavior"],
		null
	)
	var attack_token := _script_token(attack_behavior)
	var movement_behavior = _first_property(
		entity,
		["_current_movement_behavior", "current_movement_behavior", "_movement_behavior"],
		null
	)
	var charge_direction = _first_property(
		attack_behavior,
		["_charge_direction", "charge_direction", "direction"],
		Vector2.ZERO
	)
	var target_position = _first_property(
		attack_behavior,
		["target_position", "_target_position", "attack_position", "_attack_position"],
		Vector2.ZERO
	)
	var cooldown_remaining := float(_first_property(
		attack_behavior,
		["cooldown_remaining", "time_left", "_cooldown", "attack_cooldown"],
		0.0
	))
	var attack_method := _infer_attack_method(attack_token)
	return {
		"id": static_data["id"],
		"runtime_id": str(entity.get_instance_id()),
		"type": static_data["type"],
		"position": _vector_json(_property(entity, "position", Vector2.ZERO)),
		"velocity": _vector_json(_property(entity, "linear_velocity", Vector2.ZERO)),
		"health": float(_property(current_stats, "health", 1.0)),
		"max_health": max(1.0, float(_property(max_stats, "health", 1.0))),
		"radius": static_data["radius"],
		"width": static_data["width"],
		"height": static_data["height"],
		"shape": static_data["shape"],
		"size_known": static_data["size_known"],
		"is_boss": static_data["is_boss"],
		"is_elite": static_data["is_elite"],
		"is_loot": static_data["is_loot"],
		"contact_damage": float(_first_property(
			current_stats,
			["damage", "contact_damage", "touch_damage"],
			_first_property(entity, ["damage", "contact_damage", "touch_damage"], 0.0)
		)),
		"is_charging": attack_token.find("charg") >= 0,
		"is_attacking": bool(_first_property(
			attack_behavior,
			["is_attacking", "attacking", "_is_attacking", "active"],
			false
		)),
		"charge_direction": _vector_json(charge_direction),
		"attack_target": _vector_json(target_position),
		"attack_cooldown_remaining": max(0.0, cooldown_remaining),
		"attack_type": attack_token,
		"attack_method": attack_method,
		"attack_method_confidence": 0.5 if attack_method != "unknown" else 0.0,
		"attack_method_source": "script_token_heuristic" if attack_method != "unknown" else "unknown",
		"movement_type": _script_token(movement_behavior)
	}


func _infer_attack_method(token: String) -> String:
	var value := token.to_lower()
	if value.find("charg") >= 0 or value.find("dash") >= 0:
		return "charge"
	if value.find("summon") >= 0 or value.find("spawn") >= 0:
		return "summon"
	if value.find("projectile") >= 0 or value.find("bullet") >= 0 or value.find("shoot") >= 0 or value.find("ranged") >= 0:
		return "projectile"
	if value.find("aoe") >= 0 or value.find("circle") >= 0 or value.find("slash") >= 0 or value.find("area") >= 0:
		return "area"
	if value.find("melee") >= 0 or value.find("contact") >= 0 or value.find("attack") >= 0:
		return "contact"
	return "unknown"


func _collect_projectiles(main, output: Array, maximum: int, all_nodes: Array) -> void:
	var seen := {}
	for path in [
		"Projectiles",
		"EnemyProjectiles",
		"Bullets",
		"Shots",
		"EntitySpawner/Projectiles",
		"EntitySpawner/EnemyProjectiles"
	]:
		_append_projectiles(main.get_node_or_null(path), output, maximum, seen, all_nodes)
	for group_name in ["projectiles", "enemy_projectiles", "bullets", "shots"]:
		for projectile in get_tree().get_nodes_in_group(group_name):
			_append_projectile(projectile, output, maximum, seen, all_nodes)
	# Some boss attacks are spawned beneath a generic entity container and are
	# neither in a projectile group nor under one of the conventional paths.
	# Discover those nodes by stable script/resource tokens as a bounded fallback.
	if output.empty():
		_projectile_scan_nodes = 0
		_collect_projectiles_recursive(main, output, maximum, seen, all_nodes, 0)


func _collect_projectiles_recursive(
	node,
	output: Array,
	maximum: int,
	seen: Dictionary,
	all_nodes: Array,
	depth: int
) -> void:
	if node == null or output.size() >= maximum or depth > 12 or _projectile_scan_nodes >= 800:
		return
	_projectile_scan_nodes += 1
	if _looks_like_projectile(_script_token(node)):
		_append_projectile(node, output, maximum, seen, all_nodes)
	if output.size() >= maximum:
		return
	for child in node.get_children():
		_collect_projectiles_recursive(child, output, maximum, seen, all_nodes, depth + 1)


func _looks_like_projectile(token: String) -> bool:
	for term in [
		"projectile", "enemy_bullet", "enemybullet", "bullet", "missile",
		"shot", "orb", "fireball", "rocket", "laser"
	]:
		if token.find(term) >= 0:
			return true
	return false


func _append_projectiles(
	container,
	output: Array,
	maximum: int,
	seen: Dictionary,
	all_nodes: Array
) -> void:
	if container == null:
		return
	for child in container.get_children():
		_append_projectile(child, output, maximum, seen, all_nodes)


func _append_projectile(
	projectile,
	output: Array,
	maximum: int,
	seen: Dictionary,
	all_nodes: Array
) -> void:
	if projectile == null or not is_instance_valid(projectile):
		return
	var instance_id := int(projectile.get_instance_id())
	if seen.has(instance_id):
		return
	seen[instance_id] = true
	all_nodes.append(projectile)
	if output.size() >= maximum:
		return
	var static_data := _projectile_static_data(projectile)
	var shape_data := _collision_shape_data(projectile)
	var velocity = _first_property(
		projectile,
		["linear_velocity", "velocity", "current_velocity"],
		Vector2.ZERO
	)
	if typeof(velocity) != TYPE_VECTOR2:
		velocity = Vector2.ZERO
	var direction = _first_property(projectile, ["direction", "_direction"], Vector2.ZERO)
	if velocity.length_squared() < 0.01 and typeof(direction) == TYPE_VECTOR2:
		velocity = direction * float(_first_property(projectile, ["speed", "current_speed"], 0.0))
	var source = _first_property(
		projectile,
		["source", "owner_unit", "shooter", "attacker"],
		null
	)
	output.append({
		"id": static_data["id"],
		"runtime_id": str(projectile.get_instance_id()),
		"owner_id": _semantic_id(source, null) if source != null else "",
		"owner_runtime_id": str(source.get_instance_id()) if typeof(source) == TYPE_OBJECT and is_instance_valid(source) else "",
		"position": _vector_json(_first_property(
			projectile, ["global_position", "position"], Vector2.ZERO
		)),
		"velocity": _vector_json(velocity),
		"rotation": float(_property(projectile, "global_rotation", _property(projectile, "rotation", 0.0))),
		"radius": static_data["radius"],
		"width": shape_data["width"],
		"height": shape_data["height"],
		"shape": shape_data["shape"],
		"size_known": shape_data["known"],
		"damage": static_data["damage"],
		"attack_type": static_data["attack_type"],
		"time_to_live": float(_first_property(
			projectile,
			["time_to_live", "lifetime_remaining", "duration_remaining"],
			-1.0
		)),
		"kind": "projectile"
	})


func _cached_projectile_path_state(
	projectile_nodes: Array,
	spawned_enemies,
	player_state: Dictionary,
	combat_state: Dictionary,
	arena_size: Vector2
) -> Dictionary:
	if _tick - _last_projectile_path_tick >= PROJECTILE_PATH_REFRESH_TICKS:
		_last_projectile_paths = _projectile_path_state(
			projectile_nodes,
			spawned_enemies,
			player_state,
			combat_state,
			arena_size
		)
		_last_projectile_path_tick = _tick
	return _last_projectile_paths


func _projectile_path_state(
	projectile_nodes: Array,
	spawned_enemies,
	player_state: Dictionary,
	combat_state: Dictionary,
	arena_size: Vector2
) -> Dictionary:
	var grid := []
	for _index in range(BULLET_GRID_COLUMNS * BULLET_GRID_ROWS * BULLET_GRID_CHANNELS):
		grid.append(0.0)
	var action_risk := []
	var enemy_action_risk := []
	var boundary_action_risk := []
	for _index in range(BULLET_ACTION_VECTORS.size()):
		action_risk.append(0.0)
		enemy_action_risk.append(0.0)
		boundary_action_risk.append(0.0)
	var player_position := _json_vector(player_state.get("position", {}))
	var max_health := max(1.0, float(player_state.get("max_health", 1.0)))
	var player_speed := max(150.0, float(combat_state.get("move_speed", 300.0)))
	var hostile_count := 0
	for projectile in projectile_nodes:
		if projectile == null or not is_instance_valid(projectile):
			continue
		if not _is_hostile_projectile(projectile):
			continue
		hostile_count += 1
		var position = _first_property(
			projectile,
			["global_position", "position"],
			Vector2.ZERO
		)
		if typeof(position) != TYPE_VECTOR2:
			continue
		var velocity = _first_property(
			projectile,
			["linear_velocity", "velocity", "current_velocity"],
			Vector2.ZERO
		)
		if typeof(velocity) != TYPE_VECTOR2:
			velocity = Vector2.ZERO
		var direction = _first_property(projectile, ["direction", "_direction"], Vector2.ZERO)
		if velocity.length_squared() < 0.01 and typeof(direction) == TYPE_VECTOR2:
			velocity = direction * float(_first_property(
				projectile,
				["speed", "current_speed"],
				0.0
			))
		var static_data := _projectile_static_data(projectile)
		var radius := max(4.0, float(static_data.get("radius", 12.0))) + 36.0
		var damage := max(0.0, float(static_data.get("damage", 0.0)))
		var relative: Vector2 = position - player_position
		for horizon_index in range(BULLET_PATH_HORIZONS.size()):
			var horizon := float(BULLET_PATH_HORIZONS[horizon_index])
			_splat_projectile_path(
				grid,
				relative + velocity * horizon,
				radius,
				horizon_index,
				velocity,
				damage / max_health
			)
			if horizon_index > 0:
				var previous_horizon := float(BULLET_PATH_HORIZONS[horizon_index - 1])
				_splat_projectile_path(
					grid,
					relative + velocity * ((previous_horizon + horizon) * 0.5),
					radius,
					horizon_index,
					velocity,
					damage / max_health
				)
		for action_index in range(BULLET_ACTION_VECTORS.size()):
			var player_velocity: Vector2 = BULLET_ACTION_VECTORS[action_index] * player_speed
			var relative_velocity: Vector2 = velocity - player_velocity
			var speed_squared := relative_velocity.length_squared()
			var closest_time := 0.0
			if speed_squared > 1.0:
				closest_time = clamp(
					-relative.dot(relative_velocity) / speed_squared,
					0.0,
					0.8
				)
			var miss_distance := (relative + relative_velocity * closest_time).length()
			if miss_distance < radius:
				action_risk[action_index] += (
					(radius - miss_distance) / radius
					* (1.0 + min(2.0, damage / max_health))
				)
	for index in range(grid.size()):
		grid[index] = clamp(float(grid[index]), 0.0, 1.0)
	for index in range(action_risk.size()):
		action_risk[index] = clamp(float(action_risk[index]) / 4.0, 0.0, 1.0)
	var enemy_count := 0
	if typeof(spawned_enemies) == TYPE_ARRAY:
		for enemy in spawned_enemies:
			if enemy == null or not is_instance_valid(enemy):
				continue
			enemy_count += 1
			var enemy_position = _first_property(
				enemy,
				["global_position", "position"],
				Vector2.ZERO
			)
			if typeof(enemy_position) != TYPE_VECTOR2:
				continue
			var enemy_velocity = _first_property(
				enemy,
				["linear_velocity", "velocity", "current_velocity"],
				Vector2.ZERO
			)
			if typeof(enemy_velocity) != TYPE_VECTOR2:
				enemy_velocity = Vector2.ZERO
			var enemy_static := _entity_static_data(enemy)
			var contact_radius := max(
				20.0,
				float(enemy_static.get("radius", 40.0)) + 36.0
			)
			var current_stats = _property(enemy, "current_stats", null)
			var contact_damage := max(0.0, float(_first_property(
				current_stats,
				["damage", "contact_damage", "touch_damage"],
				_first_property(enemy, ["damage", "contact_damage", "touch_damage"], 0.0)
			)))
			var enemy_relative: Vector2 = enemy_position - player_position
			# Several Brotato movement behaviours update position directly and leave
			# linear_velocity at zero.  Treat a nearby mobile enemy as approaching
			# the player in that case, so IDLE is not incorrectly advertised as safe.
			if enemy_velocity.length_squared() < 1.0 and enemy_relative.length_squared() > 1.0:
				var estimated_enemy_speed := max(60.0, float(_first_property(
					enemy,
					["current_speed", "movement_speed", "move_speed", "speed"],
					120.0
				)))
				enemy_velocity = -enemy_relative.normalized() * estimated_enemy_speed
			for action_index in range(BULLET_ACTION_VECTORS.size()):
				var player_velocity: Vector2 = BULLET_ACTION_VECTORS[action_index] * player_speed
				var relative_velocity: Vector2 = enemy_velocity - player_velocity
				var speed_squared := relative_velocity.length_squared()
				var closest_time := 0.0
				if speed_squared > 1.0:
					closest_time = clamp(
						-enemy_relative.dot(relative_velocity) / speed_squared,
						0.0,
						0.8
					)
				var miss_distance := (
					enemy_relative + relative_velocity * closest_time
				).length()
				if miss_distance < contact_radius:
					enemy_action_risk[action_index] += (
						(contact_radius - miss_distance) / contact_radius
						* (1.0 + min(2.0, contact_damage / max_health))
					)
	for index in range(enemy_action_risk.size()):
		enemy_action_risk[index] = clamp(
			float(enemy_action_risk[index]) / 4.0,
			0.0,
			1.0
		)
	# Predict half a second ahead. This exposes actions that keep pressing into
	# an arena wall, a frequent precursor to an avoidable contact death.
	var boundary_margin := 80.0
	for action_index in range(BULLET_ACTION_VECTORS.size()):
		var future_position: Vector2 = (
			player_position
			+ BULLET_ACTION_VECTORS[action_index] * player_speed * 0.5
		)
		var edge_clearance := min(
			min(future_position.x, future_position.y),
			min(arena_size.x - future_position.x, arena_size.y - future_position.y)
		)
		boundary_action_risk[action_index] = clamp(
			(boundary_margin - edge_clearance) / boundary_margin,
			0.0,
			1.0
		)
	return {
		"columns": BULLET_GRID_COLUMNS,
		"rows": BULLET_GRID_ROWS,
		"channels": BULLET_GRID_CHANNELS,
		"horizons": BULLET_PATH_HORIZONS,
		"grid": grid,
		"action_risk": action_risk,
		"enemy_action_risk": enemy_action_risk,
		"boundary_action_risk": boundary_action_risk,
		"count": hostile_count,
		"enemy_count": enemy_count
	}


func _splat_projectile_path(
	grid: Array,
	relative: Vector2,
	radius: float,
	horizon_channel: int,
	velocity: Vector2,
	damage_fraction: float
) -> void:
	var half_width := BULLET_GRID_WIDTH * 0.5
	var half_height := BULLET_GRID_HEIGHT * 0.5
	if (
		relative.x + radius < -half_width
		or relative.x - radius > half_width
		or relative.y + radius < -half_height
		or relative.y - radius > half_height
	):
		return
	var cell_width := BULLET_GRID_WIDTH / BULLET_GRID_COLUMNS
	var cell_height := BULLET_GRID_HEIGHT / BULLET_GRID_ROWS
	var min_column := int(clamp(
		floor((relative.x - radius + half_width) / cell_width),
		0,
		BULLET_GRID_COLUMNS - 1
	))
	var max_column := int(clamp(
		floor((relative.x + radius + half_width) / cell_width),
		0,
		BULLET_GRID_COLUMNS - 1
	))
	var min_row := int(clamp(
		floor((relative.y - radius + half_height) / cell_height),
		0,
		BULLET_GRID_ROWS - 1
	))
	var max_row := int(clamp(
		floor((relative.y + radius + half_height) / cell_height),
		0,
		BULLET_GRID_ROWS - 1
	))
	var direction_channel := 5
	if abs(velocity.x) >= abs(velocity.y):
		direction_channel = 5 if velocity.x >= 0.0 else 6
	else:
		direction_channel = 7 if velocity.y >= 0.0 else 8
	for row in range(min_row, max_row + 1):
		for column in range(min_column, max_column + 1):
			var offset := (
				(row * BULLET_GRID_COLUMNS + column) * BULLET_GRID_CHANNELS
			)
			grid[offset + horizon_channel] += 0.25
			grid[offset + direction_channel] += 0.125
			grid[offset + 9] += max(0.05, min(1.0, damage_fraction)) * 0.25


func _is_hostile_projectile(projectile) -> bool:
	if bool(_first_property(
		projectile,
		["is_player_projectile", "from_player", "player_projectile"],
		false
	)):
		return false
	var source = _first_property(
		projectile,
		["source", "owner_unit", "shooter", "attacker"],
		null
	)
	if source != null and source == TempStats.player:
		return false
	var token := _script_token(projectile)
	if token.find("player_projectile") >= 0 or token.find("player_bullet") >= 0:
		return false
	return true


func _append_pickups(container, output: Array, kind: String, maximum: int) -> void:
	if container == null:
		return
	for child in container.get_children():
		if output.size() >= maximum:
			break
		var static_data := _pickup_static_data(child, kind)
		output.append({
			"id": static_data["id"],
			"type": static_data["type"],
			"category": static_data["category"],
			"position": _vector_json(_property(child, "position", Vector2.ZERO)),
			"velocity": _vector_json(_property(child, "linear_velocity", Vector2.ZERO)),
			"radius": static_data["radius"],
			"width": static_data["width"],
			"height": static_data["height"],
			"shape": static_data["shape"],
			"size_known": static_data["size_known"],
			"healing": static_data["healing"],
			"material_value": static_data["material_value"],
			"crate_value": static_data["crate_value"],
			"kind": kind
		})


func _entity_static_data(entity) -> Dictionary:
	if _entity_static_cache.size() > 4096:
		_entity_static_cache.clear()
	var cache_key := int(entity.get_instance_id())
	if _entity_static_cache.has(cache_key):
		return _entity_static_cache[cache_key]
	var entity_data = _first_property(
		entity,
		["_enemy_data", "enemy_data", "_unit_data", "unit_data", "_data", "data"],
		null
	)
	_log_semantic_probe("enemy", entity, entity_data)
	var entity_token := _script_token(entity) + " " + _script_token(entity_data)
	var shape_data := _collision_shape_data(entity)
	var result := {
		"id": _semantic_id(entity, entity_data),
		"type": entity_token,
		"radius": shape_data["radius"],
		"width": shape_data["width"],
		"height": shape_data["height"],
		"shape": shape_data["shape"],
		"size_known": shape_data["known"],
		"is_boss": bool(_property(entity, "is_boss", false)) or entity_token.find("boss") >= 0,
		"is_elite": bool(_first_property(entity, ["is_elite", "elite"], false)) or entity_token.find("elite") >= 0,
		"is_loot": bool(_property(entity, "is_loot", false))
	}
	_entity_static_cache[cache_key] = result
	return result


func _pickup_static_data(pickup, kind: String) -> Dictionary:
	if _pickup_static_cache.size() > 4096:
		_pickup_static_cache.clear()
	var cache_key := int(pickup.get_instance_id())
	if _pickup_static_cache.has(cache_key):
		return _pickup_static_cache[cache_key]
	var data = _first_property(
		pickup,
		[
			"_consumable_data", "consumable_data", "_pickup_data", "pickup_data",
			"_item_data", "item_data", "_data", "data"
		],
		null
	)
	_log_semantic_probe("pickup", pickup, data)
	var token := _script_token(pickup) + " " + _script_token(data)
	var category := kind
	if token.find("fruit") >= 0 or token.find("heal") >= 0 or token.find("food") >= 0:
		category = "healing"
	elif token.find("crate") >= 0 or token.find("box") >= 0 or token.find("loot") >= 0:
		category = "crate"
	elif token.find("material") >= 0 or token.find("currency") >= 0 or token.find("gold") >= 0:
		category = "material"
	elif kind == "consumable":
		category = "consumable"
	var shape_data := _collision_shape_data(pickup)
	var result := {
		"id": _semantic_id(pickup, data),
		"type": token,
		"category": category,
		"radius": shape_data["radius"],
		"width": shape_data["width"],
		"height": shape_data["height"],
		"shape": shape_data["shape"],
		"size_known": shape_data["known"],
		"healing": float(_first_property(
			pickup,
			["healing", "heal_amount", "hp_restored", "health_restored"],
			_first_property(data, ["healing", "heal_amount", "hp_restored", "health_restored"], 0.0)
		)),
		"material_value": float(_first_property(
			pickup,
			["material_value", "materials", "currency_value", "value"],
			_first_property(data, ["material_value", "materials", "currency_value"], 0.0)
		)),
		"crate_value": 1.0 if category == "crate" else 0.0
	}
	_pickup_static_cache[cache_key] = result
	return result


func _projectile_static_data(projectile) -> Dictionary:
	if _projectile_static_cache.size() > 4096:
		_projectile_static_cache.clear()
	var cache_key := int(projectile.get_instance_id())
	if _projectile_static_cache.has(cache_key):
		return _projectile_static_cache[cache_key]
	var shape_data := _collision_shape_data(projectile)
	var result := {
		"id": _semantic_id(projectile, null),
		"radius": shape_data["radius"],
		"damage": float(_first_property(projectile, ["damage", "current_damage"], 0.0)),
		"attack_type": _script_token(projectile)
	}
	_projectile_static_cache[cache_key] = result
	return result


func _collect_attack_indicators(main, projectile_nodes: Array = []) -> Array:
	var output := []
	_indicator_scan_nodes = 0
	_indicator_scan_seen = {}
	for group_name in [
		"attack_indicators",
		"attack_warnings",
		"warning_areas",
		"danger_zones",
		"telegraphs",
		"target_indicators"
	]:
		for node in get_tree().get_nodes_in_group(group_name):
			_append_attack_indicator(node, output)
	_collect_attack_indicators_recursive(main, output)
	# Some Brotato builds render the red telegraphs as hostile projectile nodes
	# rather than a separate warning-area node. Mirror those nodes into the
	# indicator channel so the policy receives the same warning geometry a human
	# sees instead of an always-empty feature block.
	for projectile in projectile_nodes:
		if output.size() >= MAX_ATTACK_INDICATORS:
			break
		_append_projectile_attack_indicator(projectile, output)
	return output


func _collect_attack_indicators_recursive(node, output: Array) -> void:
	if node == null or output.size() >= MAX_ATTACK_INDICATORS:
		return
	if _indicator_scan_nodes >= MAX_INDICATOR_SCAN_NODES:
		return
	_indicator_scan_nodes += 1
	var token := _script_token(node)
	# Walk through generic Control containers as well.  Brotato's boss
	# telegraphs are often Sprite2D/Polygon2D descendants of a plain Control
	# node, so pruning unnamed controls hides the red-circle warning entirely.
	if node is CanvasItem and not node.is_visible_in_tree():
		return
	if _looks_like_attack_indicator(token):
		_append_attack_indicator(node, output)
	for child in node.get_children():
		if output.size() >= MAX_ATTACK_INDICATORS:
			break
		_collect_attack_indicators_recursive(child, output)


func _looks_like_attack_indicator(token: String) -> bool:
	for term in [
		"attack_indicator",
		"attackindicator",
		"attack_warning",
		"attackwarning",
		"warning_area",
		"warningarea",
		"hit_warning",
		"hitwarning",
		"danger_zone",
		"dangerzone",
		"telegraph",
		"target_indicator",
		"targetindicator",
		"aim_line",
		"aimline",
		"laser_warning",
		"aoe_warning",
		"area_warning",
		"spell_warning",
		"projectile_warning"
	]:
		if token.find(term) >= 0:
			return true
	return false


func _append_projectile_attack_indicator(projectile, output: Array) -> void:
	if projectile == null or output.size() >= MAX_ATTACK_INDICATORS:
		return
	if typeof(projectile) == TYPE_OBJECT and not is_instance_valid(projectile):
		return
	if not _is_hostile_projectile(projectile):
		return
	var static_data := _projectile_static_data(projectile)
	var shape_data := _collision_shape_data(projectile)
	var velocity = _first_property(
		projectile,
		["linear_velocity", "velocity", "current_velocity"],
		Vector2.ZERO
	)
	if typeof(velocity) != TYPE_VECTOR2:
		velocity = Vector2.ZERO
	var direction = _first_property(projectile, ["direction", "_direction"], Vector2.ZERO)
	if velocity.length_squared() > 0.01:
		direction = velocity.normalized()
	elif typeof(direction) != TYPE_VECTOR2:
		direction = Vector2.ZERO
	var lifetime := float(_first_property(
		projectile,
		["time_to_live", "lifetime_remaining", "duration_remaining"],
		0.0
	))
	var identity := str(static_data.get("id", "projectile"))
	var source = _first_property(
		projectile,
		["source", "owner_unit", "shooter", "attacker"],
		null
	)
	output.append({
		"id": identity + ":incoming_projectile",
		"type": "incoming_projectile " + str(static_data.get("attack_type", "")),
		"owner_id": _semantic_id(source, null) if source != null else "",
		"owner_runtime_id": str(source.get_instance_id()) if typeof(source) == TYPE_OBJECT and is_instance_valid(source) else "",
		"position": _vector_json(_first_property(
			projectile, ["global_position", "position"], Vector2.ZERO
		)),
		"direction": _vector_json(direction),
		"rotation": float(_property(
			projectile,
			"global_rotation",
			_property(projectile, "rotation", 0.0)
		)),
		"radius": shape_data["radius"],
		"width": shape_data["width"],
		"height": shape_data["height"],
		"shape": shape_data["shape"],
		"time_to_activate": 0.0,
		"duration": max(0.0, lifetime),
		"damage": float(static_data.get("damage", 0.0)),
		"active": true,
		"source": "projectile"
	})


func _append_attack_indicator(node, output: Array) -> void:
	if node == null or output.size() >= MAX_ATTACK_INDICATORS:
		return
	if typeof(node) == TYPE_OBJECT and not is_instance_valid(node):
		return
	var instance_id := int(node.get_instance_id())
	if _indicator_scan_seen.has(instance_id):
		return
	_indicator_scan_seen[instance_id] = true
	var token := _script_token(node)
	var shape_data := _collision_shape_data(node)
	var timer = _first_property(
		node,
		["_activation_timer", "activation_timer", "_attack_timer", "attack_timer", "timer"],
		null
	)
	if timer == null:
		timer = _find_named_timer(node, ["activation", "warning", "attack"])
	var direction = _first_property(
		node,
		["direction", "attack_direction", "aim_direction", "_direction"],
		Vector2.ZERO
	)
	var owner_node = _first_property(node, ["source", "attacker", "owner_unit", "enemy"], null)
	output.append({
		"id": _semantic_id(node, null),
		"type": token,
		"owner_id": _semantic_id(owner_node, null) if owner_node != null else "",
		"owner_runtime_id": str(owner_node.get_instance_id()) if typeof(owner_node) == TYPE_OBJECT and is_instance_valid(owner_node) else "",
		"position": _vector_json(_first_property(
			node,
			["global_position", "position"],
			Vector2.ZERO
		)),
		"direction": _vector_json(direction),
		"rotation": float(_property(node, "global_rotation", _property(node, "rotation", 0.0))),
		"radius": shape_data["radius"],
		"width": shape_data["width"],
		"height": shape_data["height"],
		"shape": shape_data["shape"],
		"time_to_activate": max(0.0, float(_first_property(
			timer,
			["time_left"],
			_first_property(node, ["time_to_activate", "warning_time", "delay"], 0.0)
		))),
		"duration": max(0.0, float(_first_property(
			timer,
			["wait_time"],
			_first_property(node, ["duration", "active_duration"], 0.0)
		))),
		"damage": float(_first_property(node, ["damage", "current_damage"], 0.0)),
		"active": bool(_first_property(node, ["active", "is_active", "damaging"], false))
	})


func _script_token(object) -> String:
	if object == null:
		return ""
	var raw_name := str(_first_property(object, ["my_id", "id", "name"], "")).to_lower()
	# Generated names such as @Enemy@1622 identify one instance, not an enemy
	# class. Excluding them makes observations stable between spawns and runs.
	var token := "" if raw_name.begins_with("@") else raw_name
	var resource_path := str(_first_property(
		object,
		["resource_path", "filename", "scene_file_path"],
		""
	)).to_lower()
	if not resource_path.empty():
		token += " " + resource_path
	if object is Object:
		var script = object.get_script()
		if script != null:
			token += " " + str(script.resource_path).to_lower()
	return token.strip_edges()


func _semantic_id(object, data) -> String:
	for candidate in [data, object]:
		if candidate == null:
			continue
		var identity := str(_first_property(
			candidate,
			[
				"weapon_id", "item_id", "consumable_id", "enemy_id", "unit_id",
				"my_id", "id", "resource_path", "filename", "scene_file_path"
			],
			""
		)).strip_edges().to_lower()
		if not identity.empty() and not identity.begins_with("@"):
			return identity
	var token := (_script_token(data) + " " + _script_token(object)).strip_edges()
	return token if not token.empty() else "unknown"


func _log_semantic_probe(kind: String, object, data) -> void:
	if _logged_semantic_probes.has(kind) or object == null:
		return
	_logged_semantic_probes[kind] = true
	var properties := []
	for descriptor in object.get_property_list():
		properties.append(str(descriptor.get("name", "")))
	var children := []
	for child in object.get_children():
		if children.size() >= 24:
			break
		children.append(_script_token(child))
	print("[BrotatoRLBridge] semantic_probe kind=%s data=%s properties=%s children=%s" % [
		kind,
		_script_token(data),
		properties,
		children
	])


func _collision_radius(object) -> float:
	return float(_collision_shape_data(object)["radius"])


func _collision_shape_data(object) -> Dictionary:
	var hitbox = _first_property(object, ["_hitbox", "hitbox", "_hurtbox", "hurtbox"], null)
	var collision = _first_property(hitbox, ["_collision", "collision", "collision_shape"], null)
	if collision == null:
		collision = _first_property(object, ["_collision", "collision", "collision_shape"], null)
	if collision == null and object is Node:
		for child in object.get_children():
			if child is CollisionShape2D:
				collision = child
				break
	var shape = _property(collision, "shape", null)
	if shape == null:
		return {
			"radius": 40.0,
			"width": 80.0,
			"height": 80.0,
			"shape": "unknown",
			"known": false
		}
	var radius = _property(shape, "radius", null)
	if radius != null:
		var diameter := max(2.0, float(radius) * 2.0)
		return {
			"radius": diameter / 2.0,
			"width": diameter,
			"height": diameter,
			"shape": "circle",
			"known": true
		}
	var extents = _property(shape, "extents", null)
	if typeof(extents) == TYPE_VECTOR2:
		var width := max(2.0, float(extents.x) * 2.0)
		var height := max(2.0, float(extents.y) * 2.0)
		return {
			"radius": max(width, height) / 2.0,
			"width": width,
			"height": height,
			"shape": "rectangle",
			"known": true
		}
	return {
		"radius": 40.0,
		"width": 80.0,
		"height": 80.0,
		"shape": "unknown",
		"known": false
	}


func _property(object, property_name: String, fallback):
	if object == null:
		return fallback
	if typeof(object) == TYPE_DICTIONARY:
		var dictionary: Dictionary = object
		var dictionary_value = dictionary.get(property_name, fallback)
		return fallback if dictionary_value == null else dictionary_value
	if typeof(object) != TYPE_OBJECT or not is_instance_valid(object):
		return fallback
	var schema_key := str(object.get_class())
	var script = object.get_script()
	if script != null and not str(script.resource_path).empty():
		schema_key += ":" + str(script.resource_path)
	if not _property_name_cache.has(schema_key):
		var names := {}
		for descriptor in object.get_property_list():
			if descriptor.has("name"):
				names[str(descriptor["name"])] = true
		_property_name_cache[schema_key] = names
	var property_names: Dictionary = _property_name_cache[schema_key]
	if property_names.has(property_name):
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


func _json_vector(value) -> Vector2:
	if typeof(value) != TYPE_DICTIONARY:
		return Vector2.ZERO
	return Vector2(float(value.get("x", 0.0)), float(value.get("y", 0.0)))


func _movement_to_action(value) -> int:
	if typeof(value) != TYPE_VECTOR2:
		return 0
	var horizontal := 0
	var vertical := 0
	if value.x < -0.25:
		horizontal = -1
	elif value.x > 0.25:
		horizontal = 1
	if value.y < -0.25:
		vertical = -1
	elif value.y > 0.25:
		vertical = 1
	if horizontal == 0 and vertical == -1:
		return 1
	if horizontal == 0 and vertical == 1:
		return 2
	if horizontal == -1 and vertical == 0:
		return 3
	if horizontal == 1 and vertical == 0:
		return 4
	if horizontal == -1 and vertical == -1:
		return 5
	if horizontal == 1 and vertical == -1:
		return 6
	if horizontal == -1 and vertical == 1:
		return 7
	if horizontal == 1 and vertical == 1:
		return 8
	return 0


func _send_hello() -> void:
	_send({
		"type": "hello",
		"session": _session_id,
		"mod_version": MOD_VERSION,
		"game_version": str(ProjectSettings.get_setting("application/config/version")),
		"capabilities": [
			"structured_state",
			"movement",
			"realtime_control",
			"ui_actions",
			"combat_build_summary",
			"threat_geometry",
			"semantic_entities_v2",
			"pickup_semantics",
			"weapon_readiness",
			"attack_indicators",
			"full_arena_grid_v1",
			"projectile_path_grid_v1",
			"training_pause_v1",
			"configurable_state_rate",
			"human_input_observation"
		]
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
