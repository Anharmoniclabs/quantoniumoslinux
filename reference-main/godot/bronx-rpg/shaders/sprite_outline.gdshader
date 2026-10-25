shader_type canvas_item;

uniform vec4 outline_color : source_color = vec4(0.05, 0.05, 0.08, 0.9);
uniform float outline_width : hint_range(0.0, 3.0) = 1.0;
uniform float saturation_boost : hint_range(0.0, 1.0) = 0.15;
uniform float value_boost : hint_range(0.0, 1.0) = 0.10;

void fragment() {
	vec2 off = TEXTURE_PIXEL_SIZE * outline_width;
	vec4 c = texture(TEXTURE, UV);

	// Sample 4 cardinal neighbors for outline detection
	float a_up    = texture(TEXTURE, UV + vec2(0.0, -off.y)).a;
	float a_down  = texture(TEXTURE, UV + vec2(0.0,  off.y)).a;
	float a_left  = texture(TEXTURE, UV + vec2(-off.x, 0.0)).a;
	float a_right = texture(TEXTURE, UV + vec2( off.x, 0.0)).a;
	float neighbor_max = max(max(a_up, a_down), max(a_left, a_right));

	if (c.a < 0.1 && neighbor_max > 0.5) {
		// Outline: transparent pixel with opaque neighbor
		COLOR = outline_color;
	} else if (c.a < 0.1) {
		// Transparent — pass through
		COLOR = c;
	} else {
		// Boost saturation and value via inline RGB->HSV->RGB
		vec3 rgb = c.rgb;
		float cmax = max(rgb.r, max(rgb.g, rgb.b));
		float cmin = min(rgb.r, min(rgb.g, rgb.b));
		float delta = cmax - cmin;

		float sat = (cmax > 0.0) ? delta / cmax : 0.0;
		float val = cmax;

		// Apply boosts
		sat = min(sat + saturation_boost, 1.0);
		val = min(val + value_boost, 1.0);

		// Reconstruct — scale color to new value, then blend toward full saturation
		vec3 boosted = (cmax > 0.001) ? rgb * (val / cmax) : vec3(val);

		// Re-saturate: lerp between gray(val) and the boosted color based on new sat
		float new_delta = val * sat;
		float old_delta_ratio = (delta > 0.001) ? new_delta / delta : 0.0;
		vec3 gray = vec3(val);
		vec3 final_rgb = gray + (boosted - gray) * min(old_delta_ratio, 1.5);
		final_rgb = clamp(final_rgb, 0.0, 1.0);

		COLOR = vec4(final_rgb, c.a);
	}
}
