extends "res://main.gd"


func _on_enemy_died(enemy) -> void:
	._on_enemy_died(enemy)
	var bridge = get_node_or_null("/root/BrotatoRLBridge")
	if bridge != null:
		bridge.record_enemy_death()


func _on_player_died(player) -> void:
	._on_player_died(player)
	var bridge = get_node_or_null("/root/BrotatoRLBridge")
	if bridge != null:
		bridge.record_player_death()
