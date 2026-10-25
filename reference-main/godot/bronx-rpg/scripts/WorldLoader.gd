# Loads world.json + atlas.json and builds the Bronx world. Tiles are packed
# into a runtime atlas and drawn via a single TileMapLayer (1–2 draw calls)
# instead of hundreds of Sprite2D nodes. Decorations and labels stay as
# nodes — there are only a few dozen of each.
extends Node2D

const TILE_SIZE := 32
const ATLAS_COLS := 16   # runtime atlas: 16 columns × N rows of 32x32 tiles

var _atlas: Dictionary = {}
var _world: Dictionary = {}

var _tile_layer: TileMapLayer
var _decos_root: Node2D
var _labels_root: Node2D

var _deco_tex_cache: Dictionary = {}
var _tile_source_id: int = -1
# tile_key (String) -> Vector2i atlas coord within the TileSetAtlasSource
var _tile_atlas_coords: Dictionary = {}


func _ready() -> void:
	_atlas = _load_json("res://assets/atlas.json")
	_world = _load_json("res://data/world.json")

	_tile_layer = TileMapLayer.new()
	_tile_layer.name = "Tiles"
	add_child(_tile_layer)

	_decos_root = Node2D.new()
	_decos_root.name = "Decos"
	add_child(_decos_root)

	_labels_root = Node2D.new()
	_labels_root.name = "Labels"
	add_child(_labels_root)

	_build_runtime_tileset()
	_build()


func _load_json(path: String) -> Dictionary:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("missing %s" % path)
		return {}
	var parsed = JSON.parse_string(f.get_as_text())
	if typeof(parsed) != TYPE_DICTIONARY:
		push_error("bad json: %s" % path)
		return {}
	return parsed as Dictionary


func _load_image(res_rel: String) -> Image:
	var abs_path: String = ProjectSettings.globalize_path("res://assets/" + res_rel)
	var img := Image.new()
	if img.load(abs_path) != OK:
		push_error("Failed to load image: %s" % abs_path)
		return null
	if img.get_format() != Image.FORMAT_RGBA8:
		img.convert(Image.FORMAT_RGBA8)
	return img


# ── Runtime TileSet: pack all tile PNGs into one atlas image ──

func _build_runtime_tileset() -> void:
	var tiles_v: Variant = _atlas.get("tiles", {})
	if typeof(tiles_v) != TYPE_DICTIONARY:
		push_error("atlas.json: no tiles section")
		return
	var tiles_dict: Dictionary = tiles_v

	var keys: Array = tiles_dict.keys()
	keys.sort()
	var count: int = keys.size()
	if count == 0:
		return

	var cols: int = ATLAS_COLS
	var rows: int = int(ceil(float(count) / float(cols)))
	var atlas_w: int = cols * TILE_SIZE
	var atlas_h: int = rows * TILE_SIZE
	var atlas_img := Image.create(atlas_w, atlas_h, false, Image.FORMAT_RGBA8)

	var coords_map: Dictionary = {}
	var idx: int = 0
	for k in keys:
		var key: String = String(k)
		var entry_v: Variant = tiles_dict[key]
		if typeof(entry_v) != TYPE_DICTIONARY:
			idx += 1
			continue
		var entry: Dictionary = entry_v
		var rel: String = String(entry.get("path", ""))
		if rel == "":
			idx += 1
			continue
		var tile_img: Image = _load_image(rel)
		if tile_img == null:
			idx += 1
			continue
		var col: int = idx % cols
		var row: int = idx / cols
		var dst := Vector2i(col * TILE_SIZE, row * TILE_SIZE)
		var src_rect := Rect2i(0, 0, TILE_SIZE, TILE_SIZE)
		atlas_img.blit_rect(tile_img, src_rect, dst)
		coords_map[key] = Vector2i(col, row)
		idx += 1

	var atlas_tex := ImageTexture.create_from_image(atlas_img)

	var source := TileSetAtlasSource.new()
	source.texture = atlas_tex
	source.texture_region_size = Vector2i(TILE_SIZE, TILE_SIZE)
	# Create one tile definition per packed cell.
	for key in coords_map.keys():
		var ac: Vector2i = coords_map[key]
		source.create_tile(ac)

	var ts := TileSet.new()
	ts.tile_size = Vector2i(TILE_SIZE, TILE_SIZE)
	_tile_source_id = ts.add_source(source)

	_tile_layer.tile_set = ts
	_tile_atlas_coords = coords_map


# ── Build the world ──

func _as_dict(v: Variant) -> Dictionary:
	if typeof(v) == TYPE_DICTIONARY:
		return v
	return {}


func _deco_tex(key: String) -> Texture2D:
	if _deco_tex_cache.has(key):
		return _deco_tex_cache[key] as Texture2D
	var decos: Dictionary = _atlas.get("decos", {})
	var icons: Dictionary = _atlas.get("icons", {})
	var entry_v: Variant = null
	if decos.has(key):
		entry_v = decos[key]
	elif icons.has(key):
		entry_v = icons[key]
	if entry_v == null or typeof(entry_v) != TYPE_DICTIONARY:
		return null
	var entry: Dictionary = entry_v
	var rel: String = String(entry.get("path", ""))
	var img: Image = _load_image(rel)
	if img == null:
		return null
	var tex := ImageTexture.create_from_image(img)
	_deco_tex_cache[key] = tex
	return tex


func _build() -> void:
	var tiles_v: Variant = _world.get("tiles", [])
	if typeof(tiles_v) != TYPE_ARRAY:
		return
	var tiles: Array = tiles_v

	for gy in range(tiles.size()):
		var row_v = tiles[gy]
		if typeof(row_v) != TYPE_ARRAY:
			continue
		var row: Array = row_v
		for gx in range(row.size()):
			var cell: Dictionary = _as_dict(row[gx])
			var t: String = String(cell.get("t", "sidewalk"))
			var v: int = int(cell.get("v", 0))
			var key: String = _tile_key(t, v, cell)
			if not _tile_atlas_coords.has(key):
				continue
			var ac: Vector2i = _tile_atlas_coords[key]
			_tile_layer.set_cell(Vector2i(gx, gy), _tile_source_id, ac)

	var decos_v: Variant = _world.get("decorations", [])
	if typeof(decos_v) == TYPE_ARRAY:
		var decos: Array = decos_v
		for deco_v in decos:
			var d: Dictionary = _as_dict(deco_v)
			var dtype: String = String(d.get("type", ""))
			var dvariant: int = int(d.get("variant", 0))
			var dkey: String = "%s_%d" % [dtype, dvariant]
			if dtype in ["hydrant", "signpost", "bball_hoop",
					"office_chair", "whiteboard", "coffee_machine",
					"water_cooler", "server_rack", "bookshelf"]:
				dkey = dtype
			var tex: Texture2D = _deco_tex(dkey)
			if tex == null:
				continue
			var s := Sprite2D.new()
			s.texture = tex
			s.centered = false
			s.position = Vector2(int(d.get("x", 0)) * TILE_SIZE, int(d.get("y", 0)) * TILE_SIZE)
			_decos_root.add_child(s)

	var buildings_v: Variant = _world.get("buildings", [])
	if typeof(buildings_v) == TYPE_ARRAY:
		var buildings: Array = buildings_v
		for b_v in buildings:
			var bd: Dictionary = _as_dict(b_v)
			var lbl := Label.new()
			lbl.text = String(bd.get("label", ""))
			lbl.add_theme_font_size_override("font_size", 11)
			lbl.add_theme_color_override("font_color", Color(0.86, 0.86, 0.86))
			var bx: float = float(bd.get("x", 0))
			var by: float = float(bd.get("y", 0))
			var bw: float = float(bd.get("w", 1))
			var cx: float = (bx + bw / 2.0) * TILE_SIZE
			var cy: float = by * TILE_SIZE - 14.0
			lbl.position = Vector2(cx - 40, cy)
			_labels_root.add_child(lbl)


func _tile_key(t: String, v: int, cell: Dictionary) -> String:
	match t:
		# Office tiles
		"hardwood", "carpet":
			return "%s_%d" % [t, v % 8]
		"glass_wall", "marble", "office_wall", "office_door", "conf_carpet", "elevator", "elevator_door":
			return t
		# Legacy Bronx tiles (backward compat)
		"sidewalk", "grass":
			return "%s_%d" % [t, v % 8]
		"asphalt":
			return "asphalt_%d" % (v % 5)
		"water":
			return "water_0"
		"window":
			return "brownstone_window"
		"brownstone", "door", "roof":
			return t
		"storefront":
			var bkey: String = String(cell.get("building", ""))
			if bkey != "":
				return "storefront_%s" % bkey
			return "storefront_community"
		_:
			return "hardwood_0"


func agent_home(p_name: String) -> Vector2:
	var homes_v: Variant = _world.get("agent_homes", {})
	if typeof(homes_v) != TYPE_DICTIONARY:
		return Vector2(15 * TILE_SIZE, 8 * TILE_SIZE)
	var homes: Dictionary = homes_v
	var h: Dictionary = _as_dict(homes.get(p_name, {"x": 15, "y": 8}))
	var hx: float = float(h.get("x", 15))
	var hy: float = float(h.get("y", 8))
	return Vector2(hx * TILE_SIZE, hy * TILE_SIZE)
