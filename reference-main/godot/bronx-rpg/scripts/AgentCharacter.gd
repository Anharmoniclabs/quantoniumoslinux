# Enhanced agent character — walk bob, idle sway, procedural shadow,
# outline shader, premium label styling.  Same API as Agent.gd.
extends Node2D

const TILE_SIZE := 32
const CH_W := 24
const CH_H := 40
const WALK_SPEED := 64.0

# Animation constants
const BOB_AMPLITUDE := 2.0
const BOB_FREQUENCY := 8.0
const SWAY_AMPLITUDE := 0.3
const SWAY_FREQUENCY := 0.6
const SWAY_ROTATION_DEG := 0.5
const SHADOW_W := 18
const SHADOW_H := 6
const SHADOW_ALPHA := 0.35

var agent_name: String = ""
var agent_color: Color = Color.WHITE
var agent_state: String = "idle"

var _target: Vector2
var _direction: int = 0
var _frame: int = 0
var _frame_timer: float = 0.0
var _time: float = 0.0
var _is_moving: bool = false

# Scene node references
@onready var _visual_root: Node2D = $VisualRoot
@onready var _shadow: Sprite2D = $VisualRoot/Shadow
@onready var _sprite: Sprite2D = $VisualRoot/Body
@onready var _icon: Sprite2D = $VisualRoot/Icon
@onready var _label: Label = $VisualRoot/NameLabel
@onready var _bubble: Label = $VisualRoot/Bubble

var _textures: Array = []
var _icon_tex: Dictionary = {}
var _bubble_timer: float = 0.0

# Shared across all AgentCharacter instances — built once, reused by all 8.
static var _SHARED_SHADOW_TEX: Texture2D = null


func _ready() -> void:
	if _SHARED_SHADOW_TEX == null:
		_SHARED_SHADOW_TEX = _build_shadow_texture()
	_shadow.texture = _SHARED_SHADOW_TEX


static func _build_shadow_texture() -> Texture2D:
	var img := Image.create(SHADOW_W, SHADOW_H, false, Image.FORMAT_RGBA8)
	var rx := float(SHADOW_W) / 2.0
	var ry := float(SHADOW_H) / 2.0
	for py in range(SHADOW_H):
		for px in range(SHADOW_W):
			var dx := (float(px) + 0.5 - rx) / rx
			var dy := (float(py) + 0.5 - ry) / ry
			var d := dx * dx + dy * dy
			if d <= 1.0:
				var a := SHADOW_ALPHA * (1.0 - d * 0.5)
				img.set_pixel(px, py, Color(0.0, 0.0, 0.0, a))
			else:
				img.set_pixel(px, py, Color(0.0, 0.0, 0.0, 0.0))
	return ImageTexture.create_from_image(img)


func _texture_for(direction: int, frame: int) -> Texture2D:
	if direction < 0 or direction >= _textures.size():
		return null
	var row = _textures[direction]
	if row == null or not (row is Array):
		return null
	var arr: Array = row
	if frame < 0 or frame >= arr.size():
		return null
	return arr[frame] as Texture2D


func setup(p_name: String, p_color: Color, textures: Array, icons: Dictionary) -> void:
	agent_name = p_name
	agent_color = p_color
	_textures = textures
	_icon_tex = icons

	_sprite.texture = _texture_for(0, 0)

	var zzz = icons.get("zzz", null)
	if zzz != null:
		_icon.texture = zzz as Texture2D

	# Name label — small, above head
	_label.text = p_name
	_label.add_theme_font_size_override("font_size", 8)
	_label.add_theme_color_override("font_color", Color(1.0, 0.95, 0.7))
	_label.add_theme_color_override("font_shadow_color", Color(0.0, 0.0, 0.0, 0.9))
	_label.add_theme_constant_override("shadow_offset_x", 1)
	_label.add_theme_constant_override("shadow_offset_y", 1)
	_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER

	# Speech bubble styling
	_bubble.add_theme_font_size_override("font_size", 14)
	_bubble.add_theme_color_override("font_color", Color(0.9, 0.9, 0.9))
	_bubble.add_theme_color_override("font_shadow_color", Color(0.0, 0.0, 0.0, 0.6))
	_bubble.add_theme_constant_override("shadow_offset_x", 1)
	_bubble.add_theme_constant_override("shadow_offset_y", 1)

	_target = position


func _process(delta: float) -> void:
	_time += delta

	# ── Movement ──
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
		_is_moving = true
	else:
		if _is_moving:
			_frame = 0
			_sprite.texture = _texture_for(_direction, 0)
		_is_moving = false

	# ── Visual animation ──
	if _is_moving:
		# Walk bob
		var bob: float = sin(_time * BOB_FREQUENCY * TAU) * BOB_AMPLITUDE
		_visual_root.position.y = bob
		_visual_root.rotation = 0.0
		# Shadow squash in sync
		_shadow.scale.x = 1.0 + sin(_time * BOB_FREQUENCY * TAU) * 0.1
	else:
		# Idle sway
		_visual_root.position.y = sin(_time * SWAY_FREQUENCY * TAU) * SWAY_AMPLITUDE
		_visual_root.rotation = sin(_time * SWAY_FREQUENCY * 0.7 * TAU) * deg_to_rad(SWAY_ROTATION_DEG)
		_shadow.scale.x = 1.0

	# ── Bubble timer ──
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
