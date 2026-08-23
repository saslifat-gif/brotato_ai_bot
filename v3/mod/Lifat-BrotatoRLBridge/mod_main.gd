extends Node

const MOD_DIR_NAME := "Lifat-BrotatoRLBridge"
const MOD_LOG := "Lifat-BrotatoRLBridge:Main"
const BRIDGE_NAME := "BrotatoRLBridge"

var _bridge = null


func _init() -> void:
	var mod_dir_path := ModLoaderMod.get_unpacked_dir().plus_file(MOD_DIR_NAME)
	var movement_extension := mod_dir_path.plus_file(
		"extensions/entities/units/movement_behaviors/player_movement_behavior.gd"
	)
	var main_extension := mod_dir_path.plus_file("extensions/main.gd")
	ModLoaderMod.install_script_extension(main_extension)
	ModLoaderMod.install_script_extension(movement_extension)
	ModLoaderLog.info("Installed game-state and movement extensions", MOD_LOG)


func _ready() -> void:
	var root := get_tree().get_root()
	_bridge = root.get_node_or_null(BRIDGE_NAME)
	if _bridge == null:
		var bridge_path = ModLoaderMod.get_unpacked_dir().plus_file(
			MOD_DIR_NAME
		).plus_file("bridge.gd")
		var bridge_script = load(bridge_path)
		if bridge_script == null or not bridge_script.can_instance():
			ModLoaderLog.error("Bridge script failed to load: %s" % bridge_path, MOD_LOG)
			return
		var bridge_instance = bridge_script.new()
		if bridge_instance == null or not (bridge_instance is Node):
			ModLoaderLog.error("Bridge script did not create a Node", MOD_LOG)
			return
		_bridge = bridge_instance
		_bridge.name = BRIDGE_NAME
		call_deferred("_attach_bridge", root, _bridge)
		return
	ModLoaderLog.info("Bridge already ready on 127.0.0.1:4242", MOD_LOG)


func _attach_bridge(root, bridge) -> void:
	if not is_instance_valid(root) or not is_instance_valid(bridge):
		ModLoaderLog.error("Bridge attachment target became invalid", MOD_LOG)
		return
	if bridge.get_parent() == null:
		root.add_child(bridge)
	ModLoaderLog.info("Bridge ready on 127.0.0.1:4242", MOD_LOG)


func _exit_tree() -> void:
	if is_instance_valid(_bridge):
		_bridge.queue_free()
