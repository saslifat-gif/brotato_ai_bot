extends "res://main.gd"


func _on_enemy_died(enemy, death_data) -> void:
	# Brotato 1.1.15.x emits both the enemy and its death/drop data. Forward
	# both arguments first so vanilla material and consumable drops still run.
	._on_enemy_died(enemy, death_data)
	var bridge = get_node_or_null("/root/BrotatoRLBridge")
	if bridge != null:
		bridge.record_enemy_death()
