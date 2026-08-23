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
		var bridge_script = load(
			ModLoaderMod.get_unpacked_dir().plus_file(MOD_DIR_NAME).plus_file("bridge.gd")
		)
		_bridge = bridge_script.new()
		_bridge.name = BRIDGE_NAME
		root.add_child(_bridge)
	ModLoaderLog.info("Bridge ready on 127.0.0.1:4242", MOD_LOG)


func _exit_tree() -> void:
	if is_instance_valid(_bridge):
		_bridge.queue_free()
