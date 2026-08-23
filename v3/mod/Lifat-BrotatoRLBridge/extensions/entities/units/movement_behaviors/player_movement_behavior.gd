extends "res://entities/units/movement_behaviors/player_movement_behavior.gd"


func get_movement() -> Vector2:
	var bridge = get_node_or_null("/root/BrotatoRLBridge")
	if bridge != null and bridge.should_control():
		return bridge.get_movement()
	return .get_movement()
