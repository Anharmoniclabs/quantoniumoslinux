# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# HUD overlay -- tabbed: COMMS | QUESTS | REPORTS | LOG
extends CanvasLayer

signal cmd_patrol
signal cmd_auto
signal cmd_report
signal cmd_sage_scan
signal cmd_sage_experiments
signal chat_submitted(text: String)

const TAB_COMMS   := 0
const TAB_QUESTS  := 1
const TAB_REPORTS := 2
const TAB_LOG     := 3

const MAX_LOG_LINES   := 300
const MAX_COMMS_LINES := 120
const MAX_QUEST_LINES := 200

# -- Node refs -------------------------------------------------
@onready var cards_vbox:    VBoxContainer = $CardsPanel/Scroll/Cards
@onready var title_label:   Label         = $CardsPanel/Title
@onready var tabs:          TabContainer  = $MainPanel/VBox/Tabs

@onready var comms_rtl:     RichTextLabel = $MainPanel/VBox/Tabs/COMMS/CommsRTL
@onready var chat_input:    LineEdit      = $MainPanel/VBox/Tabs/COMMS/InputRow/ChatInput
@onready var send_btn:      Button        = $MainPanel/VBox/Tabs/COMMS/InputRow/SendBtn

@onready var quest_rtl:     RichTextLabel = $MainPanel/VBox/Tabs/QUESTS/QuestRTL
@onready var reports_rtl:   RichTextLabel = $MainPanel/VBox/Tabs/REPORTS/ReportsRTL
@onready var log_rtl:       RichTextLabel = $MainPanel/VBox/Tabs/LOG/LogRTL

@onready var patrol_btn:    Button        = $MainPanel/VBox/TopRow/PatrolBtn
@onready var auto_btn:      Button        = $MainPanel/VBox/TopRow/AutoBtn
@onready var report_btn:    Button        = $MainPanel/VBox/TopRow/ReportBtn
@onready var sage_scan_btn: Button        = $MainPanel/VBox/TopRow/SageScanBtn

# -- State -----------------------------------------------------
var _cards:          Dictionary = {}
var _auto_on:        bool       = false
var _streaming_from: String     = ""
var _quest_active:   Dictionary = {}
var _quest_history:  Array      = []
const MAX_HISTORY := 60
var _agent_colors:   Dictionary = {}

# -- Init ------------------------------------------------------
func _ready() -> void:
	title_label.add_theme_font_size_override("font_size", 20)
	title_label.add_theme_color_override("font_color", Color(1.0, 0.84, 0.0))

	_style_panel($CardsPanel, Color(0.04, 0.04, 0.06, 0.96))
	_style_panel($MainPanel,  Color(0.04, 0.04, 0.06, 0.96))

	_style_button(patrol_btn,    Color(0.18, 0.45, 0.9))
	_style_button(auto_btn,      Color(0.18, 0.65, 0.28))
	_style_button(report_btn,    Color(0.75, 0.55, 0.1))
	_style_button(sage_scan_btn, Color(0.55, 0.18, 0.82))

	for rtl: RichTextLabel in [comms_rtl, quest_rtl, reports_rtl, log_rtl]:
		rtl.add_theme_font_size_override("normal_font_size", 15)

	chat_input.add_theme_font_size_override("font_size", 17)
	send_btn.add_theme_font_size_override("font_size", 17)
	chat_input.placeholder_text = "Message ORACLE..."

	patrol_btn.pressed.connect(func(): cmd_patrol.emit())
	auto_btn.pressed.connect(_toggle_auto)
	report_btn.pressed.connect(func(): cmd_report.emit())
	sage_scan_btn.pressed.connect(_on_sage_scan_pressed)
	send_btn.pressed.connect(_submit_chat)
	chat_input.text_submitted.connect(func(_t): _submit_chat())

	_rtl_append(comms_rtl, "[color=#556677]QUE Guild online. Type a message or press PATROL.[/color]\n", MAX_COMMS_LINES)
	_rtl_append(quest_rtl,   "[color=#556677]Waiting for first patrol dispatch...[/color]\n", MAX_QUEST_LINES)
	_rtl_append(reports_rtl, "[color=#556677]No reports yet - press REPORT or wait for agents.[/color]\n", MAX_LOG_LINES)
	_rtl_append(log_rtl,     "[color=#556677]Event log ready.[/color]\n", MAX_LOG_LINES)


# -- Public API ------------------------------------------------

func add_agent(agent_name: String, profile: Dictionary) -> void:
	if _cards.has(agent_name):
		return
	var col := Color(String(profile.get("color", "#888888")))
	_agent_colors[agent_name] = col

	var card := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.08, 0.08, 0.10, 0.95)
	style.border_color = col
	style.border_width_left = 4
	style.corner_radius_top_left = 3
	style.corner_radius_bottom_left = 3
	style.content_margin_left = 10
	style.content_margin_right = 10
	style.content_margin_top = 6
	style.content_margin_bottom = 6
	card.add_theme_stylebox_override("panel", style)

	var vb := VBoxContainer.new()
	vb.add_theme_constant_override("separation", 3)
	card.add_child(vb)

	var title_row := HBoxContainer.new()
	vb.add_child(title_row)

	var name_lbl := Label.new()
	name_lbl.text = "%s  %s" % [agent_name, String(profile.get("rpg_class", ""))]
	name_lbl.add_theme_color_override("font_color", col)
	name_lbl.add_theme_font_size_override("font_size", 18)
	name_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	title_row.add_child(name_lbl)

	var level_lbl := Label.new()
	level_lbl.text = "Lv.1"
	level_lbl.add_theme_color_override("font_color", Color(0.7, 0.85, 0.6))
	level_lbl.add_theme_font_size_override("font_size", 15)
	title_row.add_child(level_lbl)

	var state_lbl := Label.new()
	state_lbl.text = "IDLE"
	state_lbl.add_theme_color_override("font_color", Color(0.9, 0.7, 0.1))
	state_lbl.add_theme_font_size_override("font_size", 14)
	state_lbl.custom_minimum_size.x = 72
	state_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	title_row.add_child(state_lbl)

	var quest_lbl := Label.new()
	quest_lbl.text = ""
	quest_lbl.add_theme_color_override("font_color", Color(0.45, 0.55, 0.65))
	quest_lbl.add_theme_font_size_override("font_size", 13)
	quest_lbl.clip_text = true
	vb.add_child(quest_lbl)

	var hp_row := HBoxContainer.new()
	hp_row.add_theme_constant_override("separation", 6)
	var hp_lbl := Label.new()
	hp_lbl.text = "HP"
	hp_lbl.custom_minimum_size.x = 26
	hp_lbl.add_theme_font_size_override("font_size", 12)
	hp_lbl.add_theme_color_override("font_color", Color(0.3, 0.85, 0.35))
	hp_row.add_child(hp_lbl)
	var hp := ProgressBar.new()
	hp.max_value = 100
	hp.value = 100
	hp.show_percentage = false
	hp.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	hp.custom_minimum_size = Vector2(0, 10)
	hp_row.add_child(hp)
	vb.add_child(hp_row)

	var xp_row := HBoxContainer.new()
	xp_row.add_theme_constant_override("separation", 6)
	var xp_lbl := Label.new()
	xp_lbl.text = "XP"
	xp_lbl.custom_minimum_size.x = 26
	xp_lbl.add_theme_font_size_override("font_size", 12)
	xp_lbl.add_theme_color_override("font_color", Color(0.45, 0.55, 0.9))
	xp_row.add_child(xp_lbl)
	var xp := ProgressBar.new()
	xp.max_value = 100
	xp.value = 0
	xp.show_percentage = false
	xp.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	xp.custom_minimum_size = Vector2(0, 10)
	xp_row.add_child(xp)
	vb.add_child(xp_row)

	var quests_lbl := Label.new()
	quests_lbl.text = "Quests: 0"
	quests_lbl.add_theme_color_override("font_color", Color(0.5, 0.5, 0.55))
	quests_lbl.add_theme_font_size_override("font_size", 12)
	vb.add_child(quests_lbl)

	cards_vbox.add_child(card)
	_cards[agent_name] = {
		"name_lbl": name_lbl, "level_lbl": level_lbl, "state_lbl": state_lbl,
		"quest_lbl": quest_lbl, "hp": hp, "xp": xp, "quests_lbl": quests_lbl,
	}


func update_stats(data: Dictionary) -> void:
	var n: String = String(data.get("name", ""))
	if not _cards.has(n):
		return
	var c: Dictionary = _cards[n]
	c["level_lbl"].text = "Lv.%d" % int(data.get("level", 1))
	c["hp"].max_value = 100
	c["hp"].value = int(data.get("hp", 100))
	var xp_max: int = max(1, int(data.get("xp_to_next", 100)))
	c["xp"].max_value = xp_max
	c["xp"].value = int(data.get("xp", 0))
	c["quests_lbl"].text = "Quests: %d" % int(data.get("quests_completed", 0))
	_set_state_label(c["state_lbl"], String(data.get("state", "idle")).to_upper())


func log_line(who: String, msg: String, category: String) -> void:
	var ts := _ts()
	var ac := _agent_hex(who)
	var line: String
	match category:
		"error":
			line = "[color=#445566][%s][/color] [color=%s][b]%s[/b][/color] [color=#dd4444]%s[/color]\n" % [ts, ac, who, _esc(msg)]
		"level_up":
			line = "[color=#445566][%s][/color] [color=%s][b]%s[/b][/color] [color=#ffd700]%s[/color]\n" % [ts, ac, who, _esc(msg)]
		"system":
			line = "[color=#445566][%s][/color] [color=#667788][b]SYS[/b][/color] [color=#8899aa]%s[/color]\n" % [ts, _esc(msg)]
		_:
			line = "[color=#445566][%s][/color] [color=%s]%s[/color] [color=#aabbcc]%s[/color]\n" % [ts, ac, who, _esc(msg)]
	_rtl_append(log_rtl, line, MAX_LOG_LINES)


func add_chat_response(who: String, msg: String) -> void:
	var col := _agent_hex(who)
	var ts  := _ts()
	var line: String
	if who == "YOU":
		line = "[color=#445566][%s][/color] [color=#88bbcc][b]YOU >[/b][/color] [color=#ccddee]%s[/color]\n" % [ts, _esc(msg)]
	elif who == "SYSTEM":
		line = "[color=#445566][%s][/color] [color=#667788][b]SYS[/b][/color] [color=#8899aa]%s[/color]\n" % [ts, _esc(msg)]
	else:
		line = "[color=#445566][%s][/color] [color=%s][b]%s >[/b][/color] [color=#e8dba0]%s[/color]\n" % [ts, col, who, _esc(msg)]
	_rtl_append(comms_rtl, line, MAX_COMMS_LINES)
	# Non-system, non-user response (i.e. Oracle replied) — unbusy the input
	if who != "SYSTEM" and who != "YOU":
		tabs.current_tab = TAB_COMMS
		_oracle_unbusy()


func add_task(task_text: String, agent_name: String) -> void:
	var key := "%s::%s" % [agent_name, task_text]
	_quest_active[key] = {"agent": agent_name, "title": task_text, "ts": _ts()}
	if _cards.has(agent_name):
		_cards[agent_name]["quest_lbl"].text = task_text.substr(0, 40)
		_set_state_label(_cards[agent_name]["state_lbl"], "WORKING")
	_refresh_quest_tab()
	tabs.current_tab = TAB_QUESTS


func quest_done_entry(agent_name: String, quest_title: String, xp: int) -> void:
	_quest_active.erase("%s::%s" % [agent_name, quest_title])
	_quest_history.push_front({"agent": agent_name, "title": quest_title, "xp": xp, "result": "DONE", "ts": _ts()})
	if _quest_history.size() > MAX_HISTORY:
		_quest_history.resize(MAX_HISTORY)
	if _cards.has(agent_name):
		_cards[agent_name]["quest_lbl"].text = ""
	_refresh_quest_tab()


func quest_fail_entry(agent_name: String, quest_title: String) -> void:
	_quest_active.erase("%s::%s" % [agent_name, quest_title])
	_quest_history.push_front({"agent": agent_name, "title": quest_title, "xp": 0, "result": "FAIL", "ts": _ts()})
	if _quest_history.size() > MAX_HISTORY:
		_quest_history.resize(MAX_HISTORY)
	if _cards.has(agent_name):
		_cards[agent_name]["quest_lbl"].text = ""
	_refresh_quest_tab()


func update_reports(data: Dictionary) -> void:
	var reports: Dictionary = data.get("reports", {})
	if reports.is_empty():
		_rtl_set(reports_rtl, "[color=#667799]No reports available yet.[/color]\n")
		return
	var sb: PackedStringArray = []
	sb.append("[color=#ffd700][b]=== AGENT REPORTS ===[/b][/color]\n")
	sb.append("[color=#445566]%s[/color]\n\n" % _ts_full())
	var keys: Array = reports.keys()
	keys.sort()
	for rkey: String in keys:
		var r = reports[rkey]
		if typeof(r) != TYPE_DICTIONARY:
			continue
		var agent: String = String(r.get("agent", rkey.split("_")[0].to_upper()))
		var col     := _agent_hex(agent)
		var health  := String(r.get("health", r.get("overall_health", "?")))
		var hcol    := _health_color(health)
		var ts_str  := String(r.get("timestamp", "")).replace("T", " ").replace("Z", "")
		sb.append("[color=%s][b]-- %s[/b][/color]  [color=%s][b]%s[/b][/color]  [color=#445566]%s[/color]\n" % [col, agent, hcol, health, ts_str])
		var skip: Array = ["agent", "timestamp", "health", "overall_health",
			"agent_summaries", "artifacts", "broken_link_files", "smells_by_type"]
		for field: String in r.keys():
			if skip.has(field):
				continue
			var val = r[field]
			if typeof(val) == TYPE_DICTIONARY or typeof(val) == TYPE_ARRAY:
				continue
			var fd: String = field.replace("_", " ").capitalize()
			sb.append("   [color=#556677]%-28s[/color] [color=#ccd8e0]%s[/color]\n" % [fd, str(val)])
		if r.has("agent_summaries") and typeof(r["agent_summaries"]) == TYPE_DICTIONARY:
			for sub_a: String in (r["agent_summaries"] as Dictionary).keys():
				var sub = r["agent_summaries"][sub_a]
				if typeof(sub) != TYPE_DICTIONARY:
					continue
				sb.append("   [color=#445566]%s[/color] > [color=%s]%s[/color]\n" % [
					sub_a, _health_color(String(sub.get("health", "?"))), String(sub.get("health", "?"))
				])
		sb.append("\n")
	_rtl_set(reports_rtl, "".join(sb))


# -- Streaming Oracle chat -------------------------------------

func chat_stream_active(_from_agent: String) -> bool:
	return _streaming_from != ""


func chat_stream_start(from_agent: String) -> void:
	while comms_rtl.get_paragraph_count() > MAX_COMMS_LINES:
		comms_rtl.remove_paragraph(0)
	_streaming_from = from_agent
	send_btn.disabled = true
	send_btn.text = "..."
	chat_input.placeholder_text = "ORACLE is typing..."
	comms_rtl.append_text("[color=#445566][%s][/color] [color=%s][b]%s >[/b][/color] [color=#e8dba0]" % [_ts(), _agent_hex(from_agent), from_agent])
	tabs.current_tab = TAB_COMMS


func chat_stream_chunk(from_agent: String, delta: String) -> void:
	if _streaming_from == from_agent:
		comms_rtl.add_text(delta)


func chat_stream_end(from_agent: String) -> void:
	if _streaming_from == from_agent:
		comms_rtl.append_text("[/color]\n")
		_streaming_from = ""
		_oracle_unbusy()


func _oracle_unbusy() -> void:
	send_btn.disabled = false
	send_btn.text = "SEND"
	chat_input.placeholder_text = "Message ORACLE..."
	chat_input.grab_focus()


# -- Private helpers -------------------------------------------

func _ts() -> String:
	var t := Time.get_datetime_dict_from_system()
	return "%02d:%02d:%02d" % [t["hour"], t["minute"], t["second"]]


func _ts_full() -> String:
	var t := Time.get_datetime_dict_from_system()
	return "%04d-%02d-%02d %02d:%02d:%02d" % [t["year"], t["month"], t["day"], t["hour"], t["minute"], t["second"]]


func _rtl_append(rtl: RichTextLabel, bbcode: String, max_lines: int) -> void:
	rtl.append_text(bbcode)
	while rtl.get_paragraph_count() > max_lines:
		rtl.remove_paragraph(0)


func _rtl_set(rtl: RichTextLabel, bbcode: String) -> void:
	rtl.clear()
	rtl.append_text(bbcode)


func _esc(s: String) -> String:
	return s.replace("[", "(").replace("]", ")")


func _agent_hex(name: String) -> String:
	if _agent_colors.has(name):
		return "#%s" % _agent_colors[name].to_html(false)
	var d := {"FORGE": "#e8a020", "VIAL": "#c060d0", "WARD": "#3080d0",
		"LORE": "#40b060", "SPARK": "#e8d040", "TIDE": "#30c0c0",
		"ORACLE": "#d060e0", "SAGE": "#6080e0", "YOU": "#88bbcc", "SYSTEM": "#667788"}
	return d.get(name, "#8899aa")


func _health_color(health: String) -> String:
	match health.to_upper():
		"GREEN":  return "#44cc66"
		"YELLOW": return "#ccaa22"
		"RED":    return "#cc3333"
		_:        return "#778899"


func _set_state_label(lbl: Label, state: String) -> void:
	lbl.text = state
	match state:
		"WORKING":  lbl.add_theme_color_override("font_color", Color(0.2, 0.9, 0.4))
		"COOLDOWN": lbl.add_theme_color_override("font_color", Color(0.5, 0.5, 0.65))
		_:          lbl.add_theme_color_override("font_color", Color(0.9, 0.7, 0.1))


func _refresh_quest_tab() -> void:
	var sb: PackedStringArray = []
	sb.append("[color=#ffd700][b]=== ACTIVE ===[/b][/color]\n\n")
	if _quest_active.is_empty():
		sb.append("[color=#556677]  No active quests.[/color]\n")
	else:
		for key: String in _quest_active.keys():
			var q: Dictionary = _quest_active[key]
			sb.append("[color=%s][b]%-8s[/b][/color]  [color=#ccddee]%s[/color]\n" % [
				_agent_hex(q["agent"]), q["agent"], _esc(q["title"])
			])
	sb.append("\n[color=#ffd700][b]=== RECENT (%d) ===[/b][/color]\n\n" % min(_quest_history.size(), MAX_HISTORY))
	for i in range(min(_quest_history.size(), MAX_HISTORY)):
		var q: Dictionary = _quest_history[i]
		var result: String = q.get("result", "DONE")
		var xp:     int    = int(q.get("xp", 0))
		var rcol := "#44cc66" if result == "DONE" else "#cc4444"
		var icon := "v" if result == "DONE" else "x"
		var xps  := ("  [color=#ccaa44]+%dXP[/color]" % xp) if xp > 0 else ""
		sb.append("[color=#445566]%s[/color]  [color=%s]%s[/color]  [color=%s][b]%-7s[/b][/color]  [color=#8899aa]%s[/color]%s\n" % [
			q.get("ts", ""), rcol, icon, _agent_hex(q["agent"]), q["agent"], _esc(q["title"]), xps
		])
	_rtl_set(quest_rtl, "".join(sb))


# -- Button handlers -------------------------------------------

func _toggle_auto() -> void:
	_auto_on = not _auto_on
	if _auto_on:
		auto_btn.text = "AUTO: ON"
		_style_button(auto_btn, Color(0.1, 0.85, 0.2))
	else:
		auto_btn.text = "AUTO"
		_style_button(auto_btn, Color(0.18, 0.65, 0.28))
	cmd_auto.emit()


func _on_sage_scan_pressed() -> void:
	sage_scan_btn.disabled = true
	sage_scan_btn.text = "SAGE SCANNING..."
	log_line("SYS", "SAGE deep scan starting - 14b, report to data/agent_reports/", "system")
	cmd_sage_scan.emit()
	tabs.current_tab = TAB_LOG
	get_tree().create_timer(360.0).timeout.connect(func():
		sage_scan_btn.disabled = false
		sage_scan_btn.text = "SAGE SCAN (14b)"
	)


func _submit_chat() -> void:
	var text := chat_input.text.strip_edges()
	if text.is_empty() or send_btn.disabled:
		return
	chat_input.text = ""
	# Jump to COMMS so the user sees their message and Oracle's streaming reply
	tabs.current_tab = TAB_COMMS
	add_chat_response("YOU", text)
	# Lock input until Oracle finishes responding
	send_btn.disabled = true
	send_btn.text = "..."
	chat_input.placeholder_text = "ORACLE is responding..."
	chat_submitted.emit(text)


# -- Styling helpers -------------------------------------------

func _style_panel(panel: Panel, bg: Color) -> void:
	var sb := StyleBoxFlat.new()
	sb.bg_color = bg
	sb.border_color = Color(0.18, 0.20, 0.25)
	sb.border_width_left = 1; sb.border_width_top = 1
	sb.border_width_right = 1; sb.border_width_bottom = 1
	sb.corner_radius_top_left = 3; sb.corner_radius_top_right = 3
	sb.corner_radius_bottom_left = 3; sb.corner_radius_bottom_right = 3
	panel.add_theme_stylebox_override("panel", sb)


func _style_button(btn: Button, col: Color) -> void:
	btn.add_theme_font_size_override("font_size", 15)
	var normal := StyleBoxFlat.new()
	normal.bg_color = col.darkened(0.55)
	normal.border_color = col
	normal.border_width_bottom = 2; normal.border_width_left = 1
	normal.border_width_right = 1; normal.border_width_top = 1
	normal.corner_radius_top_left = 4; normal.corner_radius_top_right = 4
	normal.corner_radius_bottom_left = 4; normal.corner_radius_bottom_right = 4
	normal.content_margin_left = 10; normal.content_margin_right = 10
	normal.content_margin_top = 5; normal.content_margin_bottom = 5
	btn.add_theme_stylebox_override("normal", normal)
	var hover := normal.duplicate()
	hover.bg_color = col.darkened(0.3)
	btn.add_theme_stylebox_override("hover", hover)
	var pressed_sb := normal.duplicate()
	pressed_sb.bg_color = col.darkened(0.1)
	btn.add_theme_stylebox_override("pressed", pressed_sb)
# SPDX-License-Identifier: AGPL-3.0-only# Copyright (C) 2026 Luis M. Minier / quantoniumos# HUD overlay -- tabbed: COMMS | QUESTS | REPORTS | LOGextends CanvasLayersignal cmd_patrolsignal cmd_autosignal cmd_reportsignal cmd_sage_scansignal chat_submitted(text: String)const TAB_COMMS   := 0const TAB_QUESTS  := 1const TAB_REPORTS := 2const TAB_LOG     := 3const MAX_LOG_LINES   := 300const MAX_COMMS_LINES := 120const MAX_QUEST_LINES := 200# -- Node refs -------------------------------------------------@onready var cards_vbox:    VBoxContainer = $CardsPanel/Scroll/Cards@onready var title_label:   Label         = $CardsPanel/Title@onready var tabs:          TabContainer  = $MainPanel/VBox/Tabs@onready var comms_rtl:     RichTextLabel = $MainPanel/VBox/Tabs/COMMS/CommsRTL@onready var chat_input:    LineEdit      = $MainPanel/VBox/Tabs/COMMS/InputRow/ChatInput@onready var send_btn:      Button        = $MainPanel/VBox/Tabs/COMMS/InputRow/SendBtn@onready var quest_rtl:     RichTextLabel = $MainPanel/VBox/Tabs/QUESTS/QuestRTL@onready var reports_rtl:   RichTextLabel = $MainPanel/VBox/Tabs/REPORTS/ReportsRTL@onready var log_rtl:       RichTextLabel = $MainPanel/VBox/Tabs/LOG/LogRTL@onready var patrol_btn:    Button        = $MainPanel/VBox/TopRow/PatrolBtn@onready var auto_btn:      Button        = $MainPanel/VBox/TopRow/AutoBtn@onready var report_btn:    Button        = $MainPanel/VBox/TopRow/ReportBtn@onready var sage_scan_btn: Button        = $MainPanel/VBox/TopRow/SageScanBtn# -- State -----------------------------------------------------var _cards:          Dictionary = {}var _auto_on:        bool       = falsevar _streaming_from: String     = ""var _quest_active:   Dictionary = {}var _quest_history:  Array      = []const MAX_HISTORY := 60var _agent_colors:   Dictionary = {}# -- Init ------------------------------------------------------func _ready() -> void:	title_label.add_theme_font_size_override("font_size", 20)	title_label.add_theme_color_override("font_color", Color(1.0, 0.84, 0.0))	_style_panel($CardsPanel, Color(0.04, 0.04, 0.06, 0.96))	_style_panel($MainPanel,  Color(0.04, 0.04, 0.06, 0.96))	_style_button(patrol_btn,    Color(0.18, 0.45, 0.9))	_style_button(auto_btn,      Color(0.18, 0.65, 0.28))	_style_button(report_btn,    Color(0.75, 0.55, 0.1))	_style_button(sage_scan_btn, Color(0.55, 0.18, 0.82))	for rtl: RichTextLabel in [comms_rtl, quest_rtl, reports_rtl, log_rtl]:		rtl.add_theme_font_size_override("normal_font_size", 15)	chat_input.add_theme_font_size_override("font_size", 17)	send_btn.add_theme_font_size_override("font_size", 17)	patrol_btn.pressed.connect(func(): cmd_patrol.emit())	auto_btn.pressed.connect(_toggle_auto)	report_btn.pressed.connect(func(): cmd_report.emit())	sage_scan_btn.pressed.connect(_on_sage_scan_pressed)	send_btn.pressed.connect(_submit_chat)	chat_input.text_submitted.connect(func(_t): _submit_chat())	_rtl_append(comms_rtl, "[color=#556677]QUE Guild online. Type a message or press PATROL.[/color]\n", MAX_COMMS_LINES)	_rtl_append(quest_rtl,   "[color=#556677]Waiting for first patrol dispatch...[/color]\n", MAX_QUEST_LINES)	_rtl_append(reports_rtl, "[color=#556677]No reports yet - press REPORT or wait for agents.[/color]\n", MAX_LOG_LINES)	_rtl_append(log_rtl,     "[color=#556677]Event log ready.[/color]\n", MAX_LOG_LINES)# -- Public API (called via hud.call() from BronxScene) --------func add_agent(agent_name: String, profile: Dictionary) -> void:	if _cards.has(agent_name):		return	var col := Color(String(profile.get("color", "#888888")))	_agent_colors[agent_name] = col	var card := PanelContainer.new()	var style := StyleBoxFlat.new()	style.bg_color = Color(0.08, 0.08, 0.10, 0.95)	style.border_color = col	style.border_width_left = 4	style.corner_radius_top_left = 3	style.corner_radius_bottom_left = 3	style.content_margin_left = 10	style.content_margin_right = 10	style.content_margin_top = 6	style.content_margin_bottom = 6	card.add_theme_stylebox_override("panel", style)	var vb := VBoxContainer.new()	vb.add_theme_constant_override("separation", 3)	card.add_child(vb)	var title_row := HBoxContainer.new()	vb.add_child(title_row)	var name_lbl := Label.new()	name_lbl.text = "%s  %s" % [agent_name, String(profile.get("rpg_class", ""))]	name_lbl.add_theme_color_override("font_color", col)	name_lbl.add_theme_font_size_override("font_size", 18)	name_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL	title_row.add_child(name_lbl)	var level_lbl := Label.new()	level_lbl.text = "Lv.1"	level_lbl.add_theme_color_override("font_color", Color(0.7, 0.85, 0.6))	level_lbl.add_theme_font_size_override("font_size", 15)	title_row.add_child(level_lbl)	var state_lbl := Label.new()	state_lbl.text = "IDLE"	state_lbl.add_theme_color_override("font_color", Color(0.9, 0.7, 0.1))	state_lbl.add_theme_font_size_override("font_size", 14)	state_lbl.custom_minimum_size.x = 72	state_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT	title_row.add_child(state_lbl)	var quest_lbl := Label.new()	quest_lbl.text = ""	quest_lbl.add_theme_color_override("font_color", Color(0.45, 0.55, 0.65))	quest_lbl.add_theme_font_size_override("font_size", 13)	quest_lbl.clip_text = true	vb.add_child(quest_lbl)	var hp_row := HBoxContainer.new()	hp_row.add_theme_constant_override("separation", 6)	var hp_lbl := Label.new()	hp_lbl.text = "HP"	hp_lbl.custom_minimum_size.x = 26	hp_lbl.add_theme_font_size_override("font_size", 12)	hp_lbl.add_theme_color_override("font_color", Color(0.3, 0.85, 0.35))	hp_row.add_child(hp_lbl)	var hp := ProgressBar.new()	hp.max_value = 100	hp.value = 100	hp.show_percentage = false	hp.size_flags_horizontal = Control.SIZE_EXPAND_FILL	hp.custom_minimum_size = Vector2(0, 10)	hp_row.add_child(hp)	vb.add_child(hp_row)	var xp_row := HBoxContainer.new()	xp_row.add_theme_constant_override("separation", 6)	var xp_lbl := Label.new()	xp_lbl.text = "XP"	xp_lbl.custom_minimum_size.x = 26	xp_lbl.add_theme_font_size_override("font_size", 12)	xp_lbl.add_theme_color_override("font_color", Color(0.45, 0.55, 0.9))	xp_row.add_child(xp_lbl)	var xp := ProgressBar.new()	xp.max_value = 100	xp.value = 0	xp.show_percentage = false	xp.size_flags_horizontal = Control.SIZE_EXPAND_FILL	xp.custom_minimum_size = Vector2(0, 10)	xp_row.add_child(xp)	vb.add_child(xp_row)	var quests_lbl := Label.new()	quests_lbl.text = "Quests: 0"	quests_lbl.add_theme_color_override("font_color", Color(0.5, 0.5, 0.55))	quests_lbl.add_theme_font_size_override("font_size", 12)	vb.add_child(quests_lbl)	cards_vbox.add_child(card)	_cards[agent_name] = {		"name_lbl": name_lbl, "level_lbl": level_lbl, "state_lbl": state_lbl,		"quest_lbl": quest_lbl, "hp": hp, "xp": xp, "quests_lbl": quests_lbl,	}func update_stats(data: Dictionary) -> void:	var n: String = String(data.get("name", ""))	if not _cards.has(n):		return	var c: Dictionary = _cards[n]	c["level_lbl"].text = "Lv.%d" % int(data.get("level", 1))	c["hp"].max_value = 100	c["hp"].value = int(data.get("hp", 100))	var xp_max: int = max(1, int(data.get("xp_to_next", 100)))	c["xp"].max_value = xp_max	c["xp"].value = int(data.get("xp", 0))	c["quests_lbl"].text = "Quests: %d" % int(data.get("quests_completed", 0))	_set_state_label(c["state_lbl"], String(data.get("state", "idle")).to_upper())func log_line(who: String, msg: String, category: String) -> void:	var ts := _ts()	var ac := _agent_hex(who)	var line: String	match category:		"error":			line = "[color=#445566][%s][/color] [color=%s][b]%s[/b][/color] [color=#dd4444]%s[/color]\n" % [ts, ac, who, _esc(msg)]		"level_up":			line = "[color=#445566][%s][/color] [color=%s][b]%s[/b][/color] [color=#ffd700]%s[/color]\n" % [ts, ac, who, _esc(msg)]		"system":			line = "[color=#445566][%s][/color] [color=#667788][b]SYS[/b][/color] [color=#8899aa]%s[/color]\n" % [ts, _esc(msg)]		_:			line = "[color=#445566][%s][/color] [color=%s]%s[/color] [color=#aabbcc]%s[/color]\n" % [ts, ac, who, _esc(msg)]	_rtl_append(log_rtl, line, MAX_LOG_LINES)func add_chat_response(who: String, msg: String) -> void:	var col := _agent_hex(who)	var ts  := _ts()	var line: String	if who == "YOU":		line = "[color=#445566][%s][/color] [color=#88bbcc][b]YOU >[/b][/color] [color=#ccddee]%s[/color]\n" % [ts, _esc(msg)]	elif who == "SYSTEM":		line = "[color=#445566][%s][/color] [color=#667788][b]SYS[/b][/color] [color=#8899aa]%s[/color]\n" % [ts, _esc(msg)]	else:		line = "[color=#445566][%s][/color] [color=%s][b]%s >[/b][/color] [color=#e8dba0]%s[/color]\n" % [ts, col, who, _esc(msg)]	_rtl_append(comms_rtl, line, MAX_COMMS_LINES)	if who != "SYSTEM" and who != "YOU":		tabs.current_tab = TAB_COMMS		_oracle_unbusy()func add_task(task_text: String, agent_name: String) -> void:	var key := "%s::%s" % [agent_name, task_text]	_quest_active[key] = {"agent": agent_name, "title": task_text, "ts": _ts()}	if _cards.has(agent_name):		_cards[agent_name]["quest_lbl"].text = task_text.substr(0, 40)		_set_state_label(_cards[agent_name]["state_lbl"], "WORKING")	_refresh_quest_tab()	tabs.current_tab = TAB_QUESTSfunc quest_done_entry(agent_name: String, quest_title: String, xp: int) -> void:	_quest_active.erase("%s::%s" % [agent_name, quest_title])	_quest_history.push_front({"agent": agent_name, "title": quest_title, "xp": xp, "result": "DONE", "ts": _ts()})	if _quest_history.size() > MAX_HISTORY:		_quest_history.resize(MAX_HISTORY)	if _cards.has(agent_name):		_cards[agent_name]["quest_lbl"].text = ""	_refresh_quest_tab()func quest_fail_entry(agent_name: String, quest_title: String) -> void:	_quest_active.erase("%s::%s" % [agent_name, quest_title])	_quest_history.push_front({"agent": agent_name, "title": quest_title, "xp": 0, "result": "FAIL", "ts": _ts()})	if _quest_history.size() > MAX_HISTORY:		_quest_history.resize(MAX_HISTORY)	if _cards.has(agent_name):		_cards[agent_name]["quest_lbl"].text = ""	_refresh_quest_tab()func update_reports(data: Dictionary) -> void:	var reports: Dictionary = data.get("reports", {})	if reports.is_empty():		_rtl_set(reports_rtl, "[color=#667799]No reports available yet.[/color]\n")		return	var sb: PackedStringArray = []	sb.append("[color=#ffd700][b]=== AGENT REPORTS ===[/b][/color]\n")	sb.append("[color=#445566]%s[/color]\n\n" % _ts_full())	var keys: Array = reports.keys()	keys.sort()	for rkey: String in keys:		var r = reports[rkey]		if typeof(r) != TYPE_DICTIONARY:			continue		var agent: String = String(r.get("agent", rkey.split("_")[0].to_upper()))		var col     := _agent_hex(agent)		var health  := String(r.get("health", r.get("overall_health", "?")))		var hcol    := _health_color(health)		var ts_str  := String(r.get("timestamp", "")).replace("T", " ").replace("Z", "")		sb.append("[color=%s][b]-- %s[/b][/color]  [color=%s][b]%s[/b][/color]  [color=#445566]%s[/color]\n" % [col, agent, hcol, health, ts_str])		var skip: Array = ["agent", "timestamp", "health", "overall_health",						   "agent_summaries", "artifacts", "broken_link_files", "smells_by_type"]		for field: String in r.keys():			if skip.has(field):				continue			var val = r[field]			if typeof(val) == TYPE_DICTIONARY or typeof(val) == TYPE_ARRAY:				continue			var fd: String = field.replace("_", " ").capitalize()			sb.append("   [color=#556677]%-28s[/color] [color=#ccd8e0]%s[/color]\n" % [fd, str(val)])		if r.has("agent_summaries") and typeof(r["agent_summaries"]) == TYPE_DICTIONARY:			for sub_a: String in (r["agent_summaries"] as Dictionary).keys():				var sub = r["agent_summaries"][sub_a]				if typeof(sub) != TYPE_DICTIONARY:					continue				sb.append("   [color=#445566]%s[/color] > [color=%s]%s[/color]\n" % [					sub_a, _health_color(String(sub.get("health", "?"))), String(sub.get("health", "?"))				])		sb.append("\n")	_rtl_set(reports_rtl, "".join(sb))func chat_stream_active(_from_agent: String) -> bool:	return _streaming_from != ""func chat_stream_start(from_agent: String) -> void:	while comms_rtl.get_paragraph_count() > MAX_COMMS_LINES:		comms_rtl.remove_paragraph(0)	_streaming_from = from_agent	send_btn.disabled = true	send_btn.text = "..."	chat_input.placeholder_text = "ORACLE is typing..."	comms_rtl.append_text("[color=#445566][%s][/color] [color=%s][b]%s >[/b][/color] [color=#e8dba0]" % [_ts(), _agent_hex(from_agent), from_agent])	tabs.current_tab = TAB_COMMSfunc chat_stream_chunk(from_agent: String, delta: String) -> void:	if _streaming_from == from_agent:		comms_rtl.add_text(delta)func chat_stream_end(from_agent: String) -> void:	if _streaming_from == from_agent:		comms_rtl.append_text("[/color]\n")		_streaming_from = ""		_oracle_unbusy()func _oracle_unbusy() -> void:	send_btn.disabled = false	send_btn.text = "SEND"	chat_input.placeholder_text = "Message ORACLE..."	chat_input.grab_focus()# -- Private helpers -------------------------------------------func _ts() -> String:	var t := Time.get_datetime_dict_from_system()	return "%02d:%02d:%02d" % [t["hour"], t["minute"], t["second"]]func _ts_full() -> String:	var t := Time.get_datetime_dict_from_system()	return "%04d-%02d-%02d %02d:%02d:%02d" % [t["year"], t["month"], t["day"], t["hour"], t["minute"], t["second"]]func _rtl_append(rtl: RichTextLabel, bbcode: String, max_lines: int) -> void:	rtl.append_text(bbcode)	while rtl.get_paragraph_count() > max_lines:		rtl.remove_paragraph(0)func _rtl_set(rtl: RichTextLabel, bbcode: String) -> void:	rtl.clear()	rtl.append_text(bbcode)func _esc(s: String) -> String:	return s.replace("[", "(").replace("]", ")")func _agent_hex(name: String) -> String:	if _agent_colors.has(name):		return "#%s" % _agent_colors[name].to_html(false)	var d := {"FORGE": "#e8a020", "VIAL": "#c060d0", "WARD": "#3080d0",		"LORE": "#40b060", "SPARK": "#e8d040", "TIDE": "#30c0c0",		"ORACLE": "#d060e0", "SAGE": "#6080e0", "YOU": "#88bbcc", "SYSTEM": "#667788"}	return d.get(name, "#8899aa")func _health_color(health: String) -> String:	match health.to_upper():		"GREEN":  return "#44cc66"		"YELLOW": return "#ccaa22"		"RED":    return "#cc3333"		_:        return "#778899"func _set_state_label(lbl: Label, state: String) -> void:	lbl.text = state	match state:		"WORKING": lbl.add_theme_color_override("font_color", Color(0.2, 0.9, 0.4))		"COOLDOWN": lbl.add_theme_color_override("font_color", Color(0.5, 0.5, 0.65))		_: lbl.add_theme_color_override("font_color", Color(0.9, 0.7, 0.1))func _refresh_quest_tab() -> void:	var sb: PackedStringArray = []	sb.append("[color=#ffd700][b]=== ACTIVE ===[/b][/color]\n\n")	if _quest_active.is_empty():		sb.append("[color=#556677]  No active quests.[/color]\n")	else:		for key: String in _quest_active.keys():			var q: Dictionary = _quest_active[key]			sb.append("[color=%s][b]%-8s[/b][/color]  [color=#ccddee]%s[/color]\n" % [				_agent_hex(q["agent"]), q["agent"], _esc(q["title"])			])	sb.append("\n[color=#ffd700][b]=== RECENT (%d) ===[/b][/color]\n\n" % min(_quest_history.size(), MAX_HISTORY))	for i in range(min(_quest_history.size(), MAX_HISTORY)):		var q: Dictionary = _quest_history[i]		var result: String = q.get("result", "DONE")		var xp:     int    = int(q.get("xp", 0))		var rcol := "#44cc66" if result == "DONE" else "#cc4444"		var icon := "v" if result == "DONE" else "x"		var xps  := ("  [color=#ccaa44]+%dXP[/color]" % xp) if xp > 0 else ""		sb.append("[color=#445566]%s[/color]  [color=%s]%s[/color]  [color=%s][b]%-7s[/b][/color]  [color=#8899aa]%s[/color]%s\n" % [			q.get("ts", ""), rcol, icon, _agent_hex(q["agent"]), q["agent"], _esc(q["title"]), xps		])	_rtl_set(quest_rtl, "".join(sb))# -- Button handlers -------------------------------------------func _toggle_auto() -> void:	_auto_on = not _auto_on	if _auto_on:		auto_btn.text = "AUTO: ON"		_style_button(auto_btn, Color(0.1, 0.85, 0.2))	else:		auto_btn.text = "AUTO"		_style_button(auto_btn, Color(0.18, 0.65, 0.28))	cmd_auto.emit()func _on_sage_scan_pressed() -> void:	sage_scan_btn.disabled = true	sage_scan_btn.text = "SAGE SCANNING..."	log_line("SYS", "SAGE deep scan starting - 14b, report to data/agent_reports/", "system")	cmd_sage_scan.emit()	tabs.current_tab = TAB_LOG	get_tree().create_timer(360.0).timeout.connect(func():		sage_scan_btn.disabled = false		sage_scan_btn.text = "SAGE SCAN (14b)"	)func _submit_chat() -> void:	var text := chat_input.text.strip_edges()	if text.is_empty() or send_btn.disabled:		return	chat_input.text = ""	# Show COMMS immediately so user sees their message + Oracle's reply	tabs.current_tab = TAB_COMMS	add_chat_response("YOU", text)	# Disable input until Oracle responds	send_btn.disabled = true	send_btn.text = "..."	chat_input.placeholder_text = "ORACLE is responding..."	chat_submitted.emit(text)# -- Styling helpers -------------------------------------------func _style_panel(panel: Panel, bg: Color) -> void:	var sb := StyleBoxFlat.new()	sb.bg_color = bg	sb.border_color = Color(0.18, 0.20, 0.25)	sb.border_width_left = 1; sb.border_width_top = 1	sb.border_width_right = 1; sb.border_width_bottom = 1	sb.corner_radius_top_left = 3; sb.corner_radius_top_right = 3	sb.corner_radius_bottom_left = 3; sb.corner_radius_bottom_right = 3	panel.add_theme_stylebox_override("panel", sb)func _style_button(btn: Button, col: Color) -> void:	btn.add_theme_font_size_override("font_size", 15)	var normal := StyleBoxFlat.new()	normal.bg_color = col.darkened(0.55)	normal.border_color = col	normal.border_width_bottom = 2; normal.border_width_left = 1	normal.border_width_right = 1; normal.border_width_top = 1	normal.corner_radius_top_left = 4; normal.corner_radius_top_right = 4	normal.corner_radius_bottom_left = 4; normal.corner_radius_bottom_right = 4	normal.content_margin_left = 10; normal.content_margin_right = 10	normal.content_margin_top = 5; normal.content_margin_bottom = 5	btn.add_theme_stylebox_override("normal", normal)	var hover := normal.duplicate()	hover.bg_color = col.darkened(0.3)	btn.add_theme_stylebox_override("hover", hover)	var pressed_sb := normal.duplicate()	pressed_sb.bg_color = col.darkened(0.1)	btn.add_theme_stylebox_override("pressed", pressed_sb)