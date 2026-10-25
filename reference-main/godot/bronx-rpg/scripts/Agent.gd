# One guild agent — sprite with 4 directions × 2 walk frames + state icon + bubble.
extends Node2D

const TILE_SIZE := 32
const CH_W := 24
const CH_H := 40
const WALK_SPEED := 64.0  # px/sec

var agent_name: String = ""
var agent_color: Color = Color.WHITE
var agent_state: String = "idle"

var _target: Vector2
var _direction: int = 0  # 0=down 1=left 2=right 3=up
var _frame: int = 0
var _frame_timer: float = 0.0

var _sprite: Sprite2D
var _icon: Sprite2D
var _label: Label
var _bubble: Label
var _bubble_timer: float = 0.0

# [ [down_f0, down_f1], [left_f0, left_f1], [right_f0, right_f1], [up_f0, up_f1] ]
var _textures: Array = []
var _icon_tex: Dictionary = {}


func _texture_for(direction: int, frame: int) -> Texture2D:
	if direction < 0 or direction >= _textures.size():
		return null
	var row = _textures[direction]
	if row == null or not (row is Array):
		return null
	var arr: Array = row
	if frame < 0 or frame >= arr.size():
		return null
	var item = arr[frame]
	return item as Texture2D


func setup(p_name: String, p_color: Color, textures: Array, icons: Dictionary) -> void:
	agent_name = p_name
	agent_color = p_color
	_textures = textures
	_icon_tex = icons

	_sprite = Sprite2D.new()
	_sprite.centered = false
	_sprite.texture = _texture_for(0, 0)
	add_child(_sprite)

	_icon = Sprite2D.new()
	_icon.centered = false
	_icon.position = Vector2(CH_W - 6, -10)
	var zzz = icons.get("zzz", null)
	if zzz != null:
		_icon.texture = zzz as Texture2D
	add_child(_icon)

	_label = Label.new()
	_label.text = p_name
	_label.add_theme_font_size_override("font_size", 11)
	_label.add_theme_color_override("font_color", p_color)
	_label.position = Vector2(-4, -18)
	add_child(_label)

	_bubble = Label.new()
	_bubble.add_theme_font_size_override("font_size", 11)
	_bubble.add_theme_color_override("font_color", Color(0.1, 0.1, 0.1))
	_bubble.position = Vector2(-10, -32)
	_bubble.visible = false
	add_child(_bubble)

	_target = position


func _process(delta: float) -> void:
	var to_target: Vector2 = _target - position
	var dist: float = to_target.length()
	if dist > 0.5:
		var step: float = min(WALK_SPEED * delta, dist)
		position += to_target.normalized() * step
		if abs(to_target.x) > abs(to_target.y):
			_direction = 2 if to_target.x > 0 else 1
		else:
			_direction = 0 if to_target.y > 0 else 3
		_frame_timer += delta
		if _frame_timer > 0.25:
			_frame = 1 - _frame
			_frame_timer = 0.0
			_sprite.texture = _texture_for(_direction, _frame)
	else:
		_frame = 0
		_sprite.texture = _texture_for(_direction, 0)

	if _bubble != null and _bubble.visible:
		_bubble_timer -= delta
		if _bubble_timer <= 0.0:
			_bubble.visible = false


func walk_to_tile(tile_pos: Vector2) -> void:
	_target = tile_pos


func set_state(s: String) -> void:
	agent_state = s
	var key: String = "zzz"
	match s:
		"working": key = "working"
		"cooldown": key = "alert"
		_: key = "zzz"
	if _icon_tex.has(key):
		var t = _icon_tex[key]
		if t != null:
			_icon.texture = t as Texture2D


func say(text: String, duration: float = 4.0) -> void:
	_bubble.text = text
	_bubble.visible = true
	_bubble_timer = duration
