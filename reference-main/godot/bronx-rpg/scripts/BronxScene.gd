# Main scene: wires WorldLoader + AgentBridge + agent sprites + HUD.
extends Node2D

const AGENT_SCENE := preload("res://scenes/AgentCharacter.tscn")

@onready var world = $World
@onready var bridge = $AgentBridge
@onready var agents_root: Node2D = $Agents
@onready var hud = $HUD

var _agents: Dictionary = {}
var _icon_textures: Dictionary = {}
var _atlas: Dictionary = {}
var _just_streamed: Dictionary = {}  # agents whose stream ended; skip next chat_response


func _ready() -> void:
	_load_atlas()
	_load_icon_textures()
	bridge.ready_event.connect(_on_ready_event)
	bridge.stats_event.connect(_on_stats)
	bridge.quest_start.connect(_on_quest_start)
	bridge.quest_done.connect(_on_quest_done)
	bridge.quest_fail.connect(_on_quest_fail)
	bridge.cooldown_event.connect(_on_cooldown)
	bridge.idle_action.connect(_on_idle_action)
	bridge.walk_to_event.connect(_on_walk_to)
	bridge.chat_response_event.connect(_on_chat_response)
	bridge.chat_stream_event.connect(_on_chat_stream)
	bridge.chat_chunk_event.connect(_on_chat_chunk)
	bridge.system_event.connect(_on_system)
	bridge.log_event.connect(_on_log)
	bridge.reports_snapshot_event.connect(_on_reports_snapshot)
	hud.chat_submitted.connect(_on_chat_submitted)
	hud.cmd_patrol.connect(func(): bridge.send_command({"cmd": "patrol"}))
	hud.cmd_auto.connect(func(): bridge.send_command({"cmd": "auto"}))
	hud.cmd_report.connect(func(): _request_reports())
	hud.cmd_sage_scan.connect(func(): bridge.send_command({"cmd": "sage_scan"}))

	# Fetch reports every 60 s so the REPORTS tab stays current.
	var refresh_timer := get_tree().create_timer(60.0)
	refresh_timer.timeout.connect(_on_report_timer)

	_spawn_camera()


func _request_reports() -> void:
	bridge.send_command({"cmd": "report"})
	bridge.send_command({"cmd": "reports"})


func _on_report_timer() -> void:
	bridge.send_command({"cmd": "reports"})
	# Reset the timer for the next cycle.
	get_tree().create_timer(60.0).timeout.connect(_on_report_timer)


func _on_reports_snapshot(data: Dictionary) -> void:
	hud.call("update_reports", data)


func _spawn_camera() -> void:
	# Auto-fit camera to world bounds read from data/world.json so the view
	# tracks map size instead of being hardcoded to legacy Bronx dimensions.
	var world_json := _load_json_file("res://data/world.json")
	var tile_size: int = int(world_json.get("tile_size", 32))
	var map_w: int = int(world_json.get("map_w", 31))
	var map_h: int = int(world_json.get("map_h", 17))

	var px_w: float = float(map_w * tile_size)
	var px_h: float = float(map_h * tile_size)

	# Reserve ~840px of HUD chrome (two right panels ~420 each); fit what's left.
	var vp_size: Vector2 = get_viewport().get_visible_rect().size
	var usable: Vector2 = Vector2(max(1.0, vp_size.x - 840.0), max(1.0, vp_size.y - 80.0))
	var zoom: float = min(usable.x / px_w, usable.y / px_h)
	# Clamp to readable range — very small maps shouldn't blow up past 6x.
	zoom = clamp(zoom, 1.0, 6.0)

	var cam := Camera2D.new()
	cam.position = Vector2(px_w * 0.5, px_h * 0.5)
	cam.zoom = Vector2(zoom, zoom)
	add_child(cam)
	cam.make_current()


func _load_json_file(res_path: String) -> Dictionary:
	var f := FileAccess.open(res_path, FileAccess.READ)
	if f == null:
		return {}
	var parsed = JSON.parse_string(f.get_as_text())
	if typeof(parsed) != TYPE_DICTIONARY:
		return {}
	return parsed as Dictionary


func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("dispatch_patrol"):
		bridge.send_command({"cmd": "patrol"})
	elif event.is_action_pressed("toggle_auto"):
		bridge.send_command({"cmd": "auto"})
	elif event.is_action_pressed("request_report"):
		bridge.send_command({"cmd": "report"})


func _load_atlas() -> void:
	var f := FileAccess.open("res://assets/atlas.json", FileAccess.READ)
	if f == null:
		push_error("atlas.json missing")
		return
	_atlas = JSON.parse_string(f.get_as_text())


func _load_png(res_path: String) -> Texture2D:
	var abs_path: String = ProjectSettings.globalize_path(res_path)
	var img := Image.new()
	var err := img.load(abs_path)
	if err != OK:
		push_error("Failed to load: %s" % abs_path)
		return null
	return ImageTexture.create_from_image(img)


func _load_icon_textures() -> void:
	var icons: Dictionary = _atlas.get("icons", {})
	for key in icons.keys():
		var entry: Dictionary = icons[key]
		_icon_textures[key] = _load_png("res://assets/" + String(entry.get("path", "")))


func _load_character_textures(agent_name: String) -> Array:
	var chars: Dictionary = _atlas.get("characters", {})
	var frames: Dictionary = chars.get(agent_name, {})
	var out: Array = [[], [], [], []]
	for d in range(4):
		for f in range(2):
			var key := "d%d_f%d" % [d, f]
			var entry: Dictionary = frames.get(key, {})
			var tex: Texture2D = _load_png("res://assets/" + String(entry.get("path", "")))
			out[d].append(tex)
	return out


func _get_agent(agent_name: String) -> Node:
	if _agents.has(agent_name):
		return _agents[agent_name] as Node
	return null


func _event_name(data: Dictionary) -> String:
	return String(data.get("name", ""))


func _on_ready_event(data: Dictionary) -> void:
	var names: Array = data.get("agents", [])
	var profiles: Dictionary = data.get("profiles", {})
	for entry in names:
		var agent_name: String = String(entry)
		var prof: Dictionary = profiles.get(agent_name, {})
		var col := Color(String(prof.get("color", "#888888")))
		var textures := _load_character_textures(agent_name)
		var node := AGENT_SCENE.instantiate()
		agents_root.add_child(node)
		node.call("setup", agent_name, col, textures, _icon_textures)
		var home_variant: Variant = world.call("agent_home", agent_name)
		var home: Vector2 = home_variant if home_variant is Vector2 else Vector2.ZERO
		node.position = home
		node.call("walk_to_tile", home)
		_agents[agent_name] = node
		hud.call("add_agent", agent_name, prof)


func _on_stats(data: Dictionary) -> void:
	hud.call("update_stats", data)


func _on_quest_start(data: Dictionary) -> void:
	var n := _event_name(data)
	var ag := _get_agent(n)
	if ag != null:
		ag.call("set_state", "working")
	var quest_text: String = String(data.get("quest", "?"))
	hud.call("log_line", n, "starts [%s]" % quest_text, "action")
	hud.call("add_task", quest_text, n)


func _on_quest_done(data: Dictionary) -> void:
	var n := _event_name(data)
	var ag := _get_agent(n)
	if ag != null:
		ag.call("set_state", "idle")
		ag.call("say", "Done: %s" % String(data.get("quest", "?")), 4.0)
	var quest_text: String = String(data.get("quest", "?"))
	var xp_val: int = int(data.get("xp", 0))
	hud.call("log_line", n, "completed [%s] +%dXP" % [quest_text, xp_val], "action")
	hud.call("quest_done_entry", n, quest_text, xp_val)


func _on_quest_fail(data: Dictionary) -> void:
	var n := _event_name(data)
	var ag := _get_agent(n)
	if ag != null:
		ag.call("set_state", "idle")
	var quest_text: String = String(data.get("quest", "?"))
	hud.call("log_line", n, "FAILED [%s]" % quest_text, "error")
	hud.call("quest_fail_entry", n, quest_text)


func _on_cooldown(data: Dictionary) -> void:
	var n := _event_name(data)
	var ag := _get_agent(n)
	if ag != null:
		ag.call("set_state", "cooldown")
	hud.call("log_line", n, "cooldown: %s" % String(data.get("reason", "?")), "error")


func _on_idle_action(data: Dictionary) -> void:
	hud.call("log_line", _event_name(data), String(data.get("result", "idle")), "action")


func _on_walk_to(data: Dictionary) -> void:
	var n := _event_name(data)
	var ag := _get_agent(n)
	if ag != null:
		var tx: float = float(data.get("x", 0)) * 32.0
		var ty: float = float(data.get("y", 0)) * 32.0
		ag.call("walk_to_tile", Vector2(tx, ty))


func _on_chat_response(data: Dictionary) -> void:
	var from_agent: String = String(data.get("from", "ORACLE"))
	var msg: String = String(data.get("msg", ""))
	# If a stream just finished for this agent, the chunks already painted
	# the bubble — don't duplicate it. Consume the flag and skip.
	if _just_streamed.erase(from_agent):
		hud.call("log_line", from_agent, msg, "action")
		return
	if not hud.call("chat_stream_active", from_agent):
		hud.call("add_chat_response", from_agent, msg)
	hud.call("log_line", from_agent, msg, "action")


func _on_chat_stream(data: Dictionary) -> void:
	var from_agent: String = String(data.get("from", "ORACLE"))
	var event: String = String(data.get("event", ""))
	if event == "start":
		hud.call("chat_stream_start", from_agent)
	elif event == "end":
		hud.call("chat_stream_end", from_agent)
		_just_streamed[from_agent] = true  # mark: next chat_response is a duplicate


func _on_chat_chunk(data: Dictionary) -> void:
	var from_agent: String = String(data.get("from", "ORACLE"))
	var delta: String = String(data.get("delta", ""))
	hud.call("chat_stream_chunk", from_agent, delta)


func _on_system(data: Dictionary) -> void:
	hud.call("log_line", "SYSTEM", String(data.get("msg", "")), "system")
	# Also show system messages in chat as responses
	hud.call("add_chat_response", "SYSTEM", String(data.get("msg", "")))


func _on_log(data: Dictionary) -> void:
	hud.call("log_line", _event_name(data), String(data.get("msg", "")),
		String(data.get("category", "action")))


func _on_chat_submitted(text: String) -> void:
	# Parse chat text into bridge commands
	var lower := text.to_lower().strip_edges()
	if lower == "patrol" or lower == "dispatch":
		bridge.send_command({"cmd": "patrol"})
	elif lower == "auto on":
		bridge.send_command({"cmd": "auto", "on": true})
	elif lower == "auto off":
		bridge.send_command({"cmd": "auto", "on": false})
	elif lower == "auto":
		bridge.send_command({"cmd": "auto"})
	elif lower == "report" or lower == "status":
		bridge.send_command({"cmd": "report"})
	elif lower == "help":
		hud.call("add_chat_response", "SYSTEM",
			"Commands: patrol, auto [on/off], report, status, help\nOr type any question for ORACLE.")
	else:
		# Send to ORACLE for discourse-based response
		bridge.send_command({"cmd": "chat", "text": text})
		hud.call("add_chat_response", "YOU", text)
