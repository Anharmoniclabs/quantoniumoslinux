# QUE Guild — Python bridge.
# Spawns rpg_headless.py as a child process, reads JSON events from
# stdout on a background Thread, sends JSON commands on stdin.
extends Node

signal ready_event(data: Dictionary)
signal stats_event(data: Dictionary)
signal quest_start(data: Dictionary)
signal quest_done(data: Dictionary)
signal quest_fail(data: Dictionary)
signal cooldown_event(data: Dictionary)
signal idle_action(data: Dictionary)
signal walk_to_event(data: Dictionary)
signal chat_response_event(data: Dictionary)
signal chat_stream_event(data: Dictionary)
signal chat_chunk_event(data: Dictionary)
signal log_event(data: Dictionary)
signal system_event(data: Dictionary)
signal reports_snapshot_event(data: Dictionary)

const PYTHON_EXE := "python"

var _pid: int = -1
var _proc: Dictionary = {}
var _stdio: FileAccess = null
var _stderr: FileAccess = null

var _reader_thread: Thread = null
var _stderr_thread: Thread = null
var _running: bool = false


func _ready() -> void:
	_start_python()


func _exit_tree() -> void:
	_running = false
	send_command({"cmd": "quit"})
	if _pid > 0:
		OS.kill(_pid)
	if _reader_thread != null and _reader_thread.is_started():
		_reader_thread.wait_to_finish()
	if _stderr_thread != null and _stderr_thread.is_started():
		_stderr_thread.wait_to_finish()


func _start_python() -> void:
	# res:// is godot/bronx-rpg/; repo root is two levels up.
	# globalize_path returns a trailing slash, so strip it before going up.
	var res_base: String = ProjectSettings.globalize_path("res://")
	if res_base.ends_with("/") or res_base.ends_with("\\"):
		res_base = res_base.substr(0, res_base.length() - 1)
	var godot_dir: String = res_base.get_base_dir()       # .../godot
	var project_root: String = godot_dir.get_base_dir()   # repo root
	var venv_py: String = project_root.path_join(".venv/Scripts/python.exe")
	var py: String = venv_py if FileAccess.file_exists(venv_py) else PYTHON_EXE
	var script_path: String = project_root.path_join("src/rpg_headless.py")

	if not FileAccess.file_exists(script_path):
		push_error("AgentBridge: script missing: %s" % script_path)
		return

	var args: PackedStringArray = PackedStringArray(["-u", script_path])
	_proc = OS.execute_with_pipe(py, args)
	if _proc.is_empty():
		push_error("AgentBridge: failed to spawn python: %s" % py)
		return
	_pid = int(_proc.get("pid", -1))
	_stdio = _proc.get("stdio", null) as FileAccess
	_stderr = _proc.get("stderr", null) as FileAccess
	print("AgentBridge: spawned %s (pid=%d)" % [py, _pid])
	print("AgentBridge: script=%s" % script_path)

	_running = true
	_reader_thread = Thread.new()
	_reader_thread.start(_reader_loop)
	if _stderr != null:
		_stderr_thread = Thread.new()
		_stderr_thread.start(_stderr_loop)


func _reader_loop() -> void:
	while _running and _stdio != null:
		var line: String = _stdio.get_line()
		if line == "":
			if _stdio.get_error() != OK:
				break
			continue
		call_deferred("_handle_line", line)
	# Reader exited — if we weren't asked to shut down, Python died on us.
	if _running:
		call_deferred("_handle_line",
			'{"type":"system","msg":"⚠ backend disconnected — Python headless exited"}')


func _stderr_loop() -> void:
	while _running and _stderr != null:
		var line: String = _stderr.get_line()
		if line == "":
			if _stderr.get_error() != OK:
				break
			continue
		call_deferred("_handle_stderr", line)


func _handle_stderr(line: String) -> void:
	# Don't push_warning — under bridge mode Python stderr carries human-readable
	# event mirrors (when TTY) or genuine logger output. Print only, keep the
	# warnings panel clean.
	print("[py.err] %s" % line)


func _handle_line(line: String) -> void:
	var parsed_v: Variant = JSON.parse_string(line)
	if typeof(parsed_v) != TYPE_DICTIONARY:
		push_warning("AgentBridge: non-dict line: %s" % line)
		return
	var parsed: Dictionary = parsed_v
	var kind: String = String(parsed.get("type", ""))
	match kind:
		"ready": ready_event.emit(parsed)
		"stats": stats_event.emit(parsed)
		"quest_start": quest_start.emit(parsed)
		"quest_done": quest_done.emit(parsed)
		"quest_fail": quest_fail.emit(parsed)
		"cooldown": cooldown_event.emit(parsed)
		"idle_action": idle_action.emit(parsed)
		"walk_to": walk_to_event.emit(parsed)
		"chat_response": chat_response_event.emit(parsed)
		"chat_stream": chat_stream_event.emit(parsed)
		"chat_chunk": chat_chunk_event.emit(parsed)
		"log": log_event.emit(parsed)
		"system": system_event.emit(parsed)
		"error": log_event.emit(parsed)
		"blocked": log_event.emit(parsed)
		"reports_snapshot": reports_snapshot_event.emit(parsed)
		_: push_warning("AgentBridge: unknown event: %s" % kind)


func send_command(cmd: Dictionary) -> void:
	if _stdio == null:
		return
	_stdio.store_line(JSON.stringify(cmd))
	_stdio.flush()
