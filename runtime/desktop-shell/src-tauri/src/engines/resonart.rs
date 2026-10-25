// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// Resonart Rust-side scene planner and native AetherCanvas-style image renderer.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use image::imageops::{blur, resize, FilterType};
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use rand::{Rng, SeedableRng};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::rft_kernel::RftEngine;
use super::types::{Complex, RftFlags, RftVariant};

type Rgb = [u8; 3];
type RgbaColor = [u8; 4];

const CAR_TOKENS: &[&str] = &[
    "car",
    "sportscar",
    "vehicle",
    "supercar",
    "coupe",
    "road",
    "street",
    "highway",
    "track",
    "race",
    "racing",
];

const WATER_TOKENS: &[&str] = &[
    "ocean",
    "sea",
    "lake",
    "river",
    "beach",
    "shore",
    "boat",
    "ship",
    "sailboat",
    "harbor",
    "dock",
];

const MOUNTAIN_TOKENS: &[&str] = &[
    "mountain",
    "mountains",
    "peak",
    "peaks",
    "ridge",
    "alpine",
    "cliff",
];

const CASTLE_TOKENS: &[&str] = &["castle", "fortress", "citadel", "palace", "tower"];

const BOAT_TOKENS: &[&str] = &["boat", "ship", "sailboat", "sailing", "harbor", "dock"];

const FLOWER_TOKENS: &[&str] = &[
    "flower",
    "flowers",
    "rose",
    "roses",
    "tulip",
    "tulips",
    "blossom",
    "blossoms",
    "garden",
];

const BIRD_TOKENS: &[&str] = &["bird", "birds", "eagle", "hawk", "sparrow", "crow"];
#[derive(Debug, Clone, Serialize)]
pub struct ResonartScenePlan {
    pub tokens: Vec<String>,
    pub blocked_tokens: Vec<String>,
    pub palette_hint: String,
    pub scene_hint: String,
    pub spectral_complexity: f64,
}

#[derive(Debug, Clone)]
pub struct ResonartRenderRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub steps: u32,
    pub style: String,
    pub width: u32,
    pub height: u32,
    pub quality: String,
    pub scene_hint: String,
    pub palette_hint: String,
    pub spectral_complexity: f64,
    pub seed: Option<i64>,
    pub output_dir: String,
}

#[derive(Debug, Clone)]
pub struct ResonartRenderResult {
    pub path: String,
    pub prompt: String,
    pub style: String,
    pub seed: i64,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy)]
struct StyleProfile {
    blur: f32,
    contrast: f32,
    saturation: f32,
    sharpen: f32,
    pixelate: u32,
    vignette: f32,
    neon: bool,
    greyscale: bool,
    edge: bool,
    paint: bool,
    grain: f32,
}

pub fn plan_scene(prompt: &str, negative_prompt: &str) -> ResonartScenePlan {
    let tokens = tokenize(prompt);
    let blocked_tokens = tokenize(negative_prompt);

    let palette_hint = pick_palette_hint(&tokens, &blocked_tokens);
    let scene_hint = pick_scene_hint(&tokens, &blocked_tokens);
    let spectral_complexity = spectral_complexity(prompt);

    ResonartScenePlan {
        tokens,
        blocked_tokens,
        palette_hint,
        scene_hint,
        spectral_complexity,
    }
}

pub fn generate_image(req: ResonartRenderRequest) -> Result<ResonartRenderResult, String> {
    let seed = req.seed.unwrap_or_else(|| prompt_seed(&req.prompt) as i64);
    let style = normalize_style(&req.style).to_string();
    let quality = normalize_quality(&req.quality);
    let width = req.width.clamp(256, 4096);
    let height = req.height.clamp(256, 4096);
    let render_scale = render_scale_for(&style, quality);
    let render_w = width.saturating_mul(render_scale).clamp(256, 8192);
    let render_h = height.saturating_mul(render_scale).clamp(256, 8192);

    let mut tokens = token_set(&req.prompt);
    apply_scene_hint_tokens(&mut tokens, &req.scene_hint);
    let negative_tokens = token_set(&req.negative_prompt);
    let palette = extract_palette(&req.prompt, &negative_tokens, &req.palette_hint);
    let is_portrait = is_portrait(&tokens, &req.scene_hint);

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
    let mut img = spectral_background_image(
        &mut rng,
        render_w,
        render_h,
        &palette,
        req.steps.clamp(8, 60),
        &tokens,
        req.spectral_complexity,
    )?;

    paint_environment(&mut img, &mut rng, &tokens, &negative_tokens, &palette);
    render_semantic_subject(&mut img, &mut rng, &tokens, &negative_tokens, &req.prompt, &palette, &style);

    if is_portrait && matches!(style.as_str(), "photo" | "oil_paint") {
        img = portrait_dof(&img);
        portrait_lighting(&mut img);
    }

    add_depth_and_lighting(&mut img, &style);
    img = postprocess(img, style_profile(&style), seed as u64);

    if render_scale > 1 {
        img = resize(&img, width, height, FilterType::Lanczos3);
    }

    let output_dir = PathBuf::from(&req.output_dir);
    std::fs::create_dir_all(&output_dir).map_err(|e| format!("Failed to create output dir: {e}"))?;
    let filename = format!("{}_{}.png", unix_timestamp(), slugify(&req.prompt));
    let out_path = output_dir.join(filename);
    DynamicImage::ImageRgba8(img)
        .save_with_format(&out_path, ImageFormat::Png)
        .map_err(|e| format!("PNG save error: {e}"))?;

    Ok(ResonartRenderResult {
        path: out_path.to_string_lossy().to_string(),
        prompt: req.prompt,
        style,
        seed,
        width,
        height,
    })
}

/// Legacy compatibility helper. The desktop app used to let Rust render only
/// the background and hand the rest to Python. Keeping this around preserves
/// the scene-plan preview command and remains handy for tests.
#[allow(dead_code)]
pub fn render_background_bmp(
    out_path: &str,
    width: u32,
    height: u32,
    plan: &ResonartScenePlan,
    steps: u32,
    seed: u64,
) -> Result<(), String> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let palette = palette_from_hint(&plan.palette_hint);
    let tokens = token_set_from_slice(&plan.tokens);
    let img = spectral_background_image(
        &mut rng,
        width.clamp(64, 4096),
        height.clamp(64, 4096),
        &palette,
        steps.clamp(8, 60),
        &tokens,
        plan.spectral_complexity,
    )?;

    let path = Path::new(out_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Failed to create BMP dir: {e}"))?;
    }
    DynamicImage::ImageRgba8(img)
        .save_with_format(path, ImageFormat::Bmp)
        .map_err(|e| format!("BMP write error: {e}"))?;
    Ok(())
}

fn tokenize(text: &str) -> Vec<String> {
    let mut out: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .collect();
    out.sort();
    out.dedup();
    out
}

fn token_set(text: &str) -> BTreeSet<String> {
    token_set_from_slice(&tokenize(text))
}

fn token_set_from_slice(tokens: &[String]) -> BTreeSet<String> {
    tokens.iter().cloned().collect()
}

fn has_token(tokens: &BTreeSet<String>, needle: &str) -> bool {
    tokens.contains(needle)
}

fn has_any(tokens: &BTreeSet<String>, needles: &[&str]) -> bool {
    needles.iter().any(|needle| has_token(tokens, needle))
}

fn pick_palette_hint(tokens: &[String], blocked_tokens: &[String]) -> String {
    let token_set = token_set_from_slice(tokens);
    let blocked_set = token_set_from_slice(blocked_tokens);

    let checks = [
        ("sportscar", "racing"),
        ("supercar", "racing"),
        ("car", "racing"),
        ("coupe", "racing"),
        ("vehicle", "racing"),
        ("boat", "ocean"),
        ("ship", "ocean"),
        ("sailboat", "ocean"),
        ("lake", "ocean"),
        ("sea", "ocean"),
        ("beach", "ocean"),
        ("mountain", "forest"),
        ("winter", "snow"),
        ("snow", "snow"),
        ("moon", "midnight"),
        ("castle", "midnight"),
        ("flower", "pastel"),
        ("flowers", "pastel"),
        ("neon", "neon"),
        ("cyberpunk", "cyberpunk"),
        ("ocean", "ocean"),
        ("forest", "forest"),
        ("sunset", "sunset"),
        ("city", "city"),
        ("space", "space"),
        ("golden", "golden"),
        ("cat", "cat"),
    ];

    for (token, palette) in checks {
        if has_token(&token_set, token) && !has_token(&blocked_set, token) {
            return palette.to_string();
        }
    }

    "default".to_string()
}

fn pick_scene_hint(tokens: &[String], blocked_tokens: &[String]) -> String {
    let token_set = token_set_from_slice(tokens);
    let blocked_set = token_set_from_slice(blocked_tokens);

    if has_any(&token_set, &["dog", "puppy", "retriever", "canine"])
        && !has_any(&blocked_set, &["dog", "puppy", "retriever", "canine"])
    {
        if has_any(&token_set, &["portrait", "headshot", "closeup", "close"]) {
            return "dog_portrait".to_string();
        }
        return "dog".to_string();
    }

    if has_any(&token_set, &["cat", "kitten", "feline"]) && !has_any(&blocked_set, &["cat", "kitten", "feline"]) {
        return "cat".to_string();
    }

    if has_any(&token_set, CASTLE_TOKENS) && !has_any(&blocked_set, CASTLE_TOKENS) {
        return "castle".to_string();
    }

    if has_any(&token_set, BOAT_TOKENS) && !has_any(&blocked_set, BOAT_TOKENS) {
        return "sailboat".to_string();
    }

    if has_any(&token_set, MOUNTAIN_TOKENS) && !has_any(&blocked_set, MOUNTAIN_TOKENS) {
        return "mountain_landscape".to_string();
    }

    if has_any(&token_set, FLOWER_TOKENS) && !has_any(&blocked_set, FLOWER_TOKENS) {
        return "flowers".to_string();
    }

    if has_any(&token_set, &["tree", "forest"]) && !has_any(&blocked_set, &["tree", "forest"]) {
        return "tree".to_string();
    }

    if has_any(&token_set, &["sportscar", "supercar", "car", "vehicle", "coupe"])
        && !has_any(&blocked_set, &["sportscar", "supercar", "car", "vehicle", "coupe"])
    {
        return "sportscar".to_string();
    }

    if has_any(&token_set, &["city", "skyline", "building", "urban"])
        && !has_any(&blocked_set, &["city", "skyline", "building", "urban"])
    {
        return "cityscape".to_string();
    }

    "environment".to_string()
}

fn spectral_complexity(prompt: &str) -> f64 {
    const N: usize = 16;

    let mut input = vec![Complex::ZERO; N];
    for (i, b) in prompt.bytes().take(N).enumerate() {
        let v = (b as f64 / 255.0) * 2.0 - 1.0;
        input[i] = Complex::new(v, 0.0);
    }

    let Ok(engine) = RftEngine::new_with_variant(N, RftFlags::DEFAULT, RftVariant::EntropyGuided) else {
        return 0.0;
    };
    let Ok(freq) = engine.forward(&input) else {
        return 0.0;
    };

    let total: f64 = freq.iter().map(|c| c.norm_sq()).sum();
    (total / N as f64).clamp(0.0, 4.0)
}

fn palette_from_hint(hint: &str) -> Vec<Rgb> {
    match hint {
        "forest" => vec![[34, 85, 34], [60, 120, 40], [100, 160, 60], [180, 210, 100], [240, 230, 180]],
        "ocean" => vec![[10, 30, 80], [20, 60, 140], [40, 120, 180], [80, 180, 200], [200, 240, 255]],
        "desert" => vec![[180, 130, 60], [220, 170, 90], [240, 200, 120], [255, 230, 160], [255, 245, 210]],
        "snow" => vec![[200, 220, 240], [220, 235, 250], [240, 245, 255], [255, 255, 255], [180, 200, 230]],
        "volcano" => vec![[80, 10, 0], [160, 30, 10], [220, 80, 20], [255, 140, 40], [255, 220, 100]],
        "neon" => vec![[255, 0, 128], [0, 255, 180], [180, 0, 255], [255, 220, 0], [0, 200, 255]],
        "cyberpunk" => vec![[20, 0, 40], [60, 0, 120], [0, 200, 255], [255, 50, 150], [255, 200, 0]],
        "retro" => vec![[255, 100, 50], [255, 200, 50], [50, 200, 100], [50, 100, 200], [200, 50, 200]],
        "pastel" => vec![[255, 200, 210], [200, 220, 255], [200, 255, 220], [255, 255, 200], [230, 200, 255]],
        "dark" => vec![[10, 10, 20], [30, 20, 40], [50, 40, 60], [80, 60, 80], [120, 100, 120]],
        "midnight" => vec![[5, 5, 20], [15, 10, 40], [30, 20, 80], [50, 40, 120], [100, 80, 160]],
        "city" => vec![[40, 40, 60], [80, 80, 100], [160, 160, 180], [200, 180, 120], [255, 200, 80]],
        "space" => vec![[0, 0, 10], [10, 5, 30], [20, 10, 60], [80, 40, 120], [255, 255, 255]],
        "sky" => vec![[100, 160, 240], [140, 190, 255], [180, 220, 255], [220, 240, 255], [255, 255, 255]],
        "sunset" => vec![[255, 80, 30], [255, 140, 60], [255, 180, 80], [255, 220, 120], [200, 100, 180]],
        "racing" => vec![[110, 12, 12], [190, 24, 24], [238, 60, 48], [245, 180, 72], [250, 236, 220]],
        "dog" => vec![[126, 92, 60], [158, 122, 76], [201, 168, 119], [230, 210, 173], [98, 72, 48]],
        "golden" => vec![[166, 124, 48], [202, 156, 76], [226, 188, 116], [244, 220, 170], [92, 60, 30]],
        "cat" => vec![[100, 100, 110], [145, 145, 152], [190, 182, 170], [240, 226, 210], [70, 68, 76]],
        _ => vec![[60, 80, 140], [100, 130, 180], [160, 180, 220], [200, 210, 230], [240, 245, 255]],
    }
}

fn extract_palette(prompt: &str, negative_tokens: &BTreeSet<String>, palette_hint: &str) -> Vec<Rgb> {
    let tokens = token_set(prompt);
    if !palette_hint.is_empty() && !has_token(negative_tokens, palette_hint) {
        let hinted = palette_from_hint(palette_hint);
        if !palette_hint.eq_ignore_ascii_case("default") || has_token(&tokens, "default") {
            return hinted;
        }
    }

    let keys = [
        "forest",
        "ocean",
        "desert",
        "snow",
        "volcano",
        "neon",
        "cyberpunk",
        "retro",
        "pastel",
        "dark",
        "midnight",
        "city",
        "space",
        "sky",
        "sunset",
        "racing",
        "dog",
        "golden",
        "cat",
    ];

    for key in keys {
        if has_token(&tokens, key) && !has_token(negative_tokens, key) {
            return palette_from_hint(key);
        }
    }

    palette_from_hint("default")
}

fn spectral_background_image(
    rng: &mut rand::rngs::StdRng,
    width: u32,
    height: u32,
    palette: &[Rgb],
    steps: u32,
    tokens: &BTreeSet<String>,
    spectral_complexity: f64,
) -> Result<RgbaImage, String> {
    let width_usize = width as usize;
    let height_usize = height as usize;
    let mut canvas = vec![0.0f32; width_usize * height_usize * 3];

    let octaves = ((steps / 3) + 4).clamp(4, 8) as usize;
    let freq_bias = if has_any(tokens, &["portrait", "dog", "cat", "headshot", "closeup"]) {
        0.5f32
    } else {
        1.0f32
    };
    let richness = (1.0 + spectral_complexity.clamp(0.0, 4.0) as f32 * 0.08).clamp(1.0, 1.32);

    let mut frequency = freq_bias;
    let mut amplitude = 1.0f32;
    let gain = 0.5f32;
    let mut total_amplitude = 0.0f32;
    let mut angle = 0.0f32;

    for octave in 0..octaves {
        let c1 = palette[octave % palette.len()];
        let c2 = palette[(octave + 1) % palette.len()];
        let phase_x = rng.gen_range(0.0f32..(2.0 * std::f32::consts::PI));
        let phase_y = rng.gen_range(0.0f32..(2.0 * std::f32::consts::PI));
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        for y in 0..height_usize {
            let yy = y as f32 / height_usize as f32 * 4.0 * std::f32::consts::PI;
            let vertical = y as f32 / height_usize as f32;

            for x in 0..width_usize {
                let xx = x as f32 / width_usize as f32 * 4.0 * std::f32::consts::PI;
                let rx = cos_a * xx - sin_a * yy;
                let ry = sin_a * xx + cos_a * yy;
                let wave = (frequency * rx + phase_x).sin() * (frequency * ry + phase_y).cos();
                let t = (wave + 1.0) * 0.5;
                let smooth = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
                let fog = 1.0 - (vertical - 0.5).abs() * 0.18;
                let alpha = (smooth * fog.clamp(0.82, 1.0)).clamp(0.0, 1.0);

                let idx = (y * width_usize + x) * 3;
                canvas[idx] += amplitude * ((1.0 - alpha) * c1[0] as f32 + alpha * c2[0] as f32);
                canvas[idx + 1] += amplitude * ((1.0 - alpha) * c1[1] as f32 + alpha * c2[1] as f32);
                canvas[idx + 2] += amplitude * ((1.0 - alpha) * c1[2] as f32 + alpha * c2[2] as f32);
            }
        }

        total_amplitude += amplitude;
        frequency *= 2.01 * richness;
        amplitude *= gain;
        angle += 0.3926991;
    }

    if total_amplitude <= f32::EPSILON {
        return Err("Spectral background normalization failed".to_string());
    }

    for value in &mut canvas {
        *value /= total_amplitude;
    }

    let top = palette[0];
    let bottom = palette[(2).min(palette.len() - 1)];
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;

    for y in 0..height_usize {
        let vertical = y as f32 / height_usize as f32;
        let grad_r = (1.0 - vertical) * top[0] as f32 + vertical * bottom[0] as f32;
        let grad_g = (1.0 - vertical) * top[1] as f32 + vertical * bottom[1] as f32;
        let grad_b = (1.0 - vertical) * top[2] as f32 + vertical * bottom[2] as f32;

        for x in 0..width_usize {
            let idx = (y * width_usize + x) * 3;
            canvas[idx] = canvas[idx] * 0.55 + grad_r * 0.45;
            canvas[idx + 1] = canvas[idx + 1] * 0.55 + grad_g * 0.45;
            canvas[idx + 2] = canvas[idx + 2] * 0.55 + grad_b * 0.45;
            lo = lo.min(canvas[idx]).min(canvas[idx + 1]).min(canvas[idx + 2]);
            hi = hi.max(canvas[idx]).max(canvas[idx + 1]).max(canvas[idx + 2]);
        }
    }

    let mut img = RgbaImage::new(width, height);
    for y in 0..height_usize {
        for x in 0..width_usize {
            let idx = (y * width_usize + x) * 3;
            let r = normalize_channel(canvas[idx], lo, hi);
            let g = normalize_channel(canvas[idx + 1], lo, hi);
            let b = normalize_channel(canvas[idx + 2], lo, hi);
            img.put_pixel(x as u32, y as u32, Rgba([r, g, b, 255]));
        }
    }

    Ok(img)
}

fn paint_environment(
    img: &mut RgbaImage,
    rng: &mut rand::rngs::StdRng,
    tokens: &BTreeSet<String>,
    negative_tokens: &BTreeSet<String>,
    palette: &[Rgb],
) {
    let width = img.width() as i32;
    let height = img.height() as i32;
    let portrait = has_any(tokens, &["dog", "cat", "portrait"]);
    let car_scene = has_any(tokens, CAR_TOKENS);
    let horizon = if portrait {
        (height as f32 * 0.62) as i32
    } else if car_scene {
        (height as f32 * 0.6) as i32
    } else {
        (height as f32 * 0.55) as i32
    };

    if has_any(tokens, &["field", "grass", "park", "meadow", "dog", "cat"])
        && !has_any(negative_tokens, &["field", "grass", "park", "meadow", "dog", "cat"])
    {
        let ground = [palette[(2).min(palette.len() - 1)][0] / 2, 130, 72, 255];
        draw_rect(img, 0, horizon, width, height, ground);
        let blades = (width / 8).max(32);
        for _ in 0..blades {
            let x = rng.gen_range(0..width);
            let blade_h = rng.gen_range((height / 30).max(4)..(height / 12).max(8));
            draw_line(
                img,
                x,
                height - 1,
                x + rng.gen_range(-3..=3),
                height - blade_h,
                [90, 175, 70, 170],
                1,
            );
        }
    }

    if has_any(tokens, WATER_TOKENS) && !has_any(negative_tokens, WATER_TOKENS) {
        paint_water(img, width, height, horizon, tokens, palette);
    }

    if has_any(tokens, MOUNTAIN_TOKENS) && !has_any(negative_tokens, MOUNTAIN_TOKENS) {
        paint_mountains(img, width, height, tokens);
    }

    if has_any(tokens, &["city", "skyline", "building", "urban"])
        && !has_any(negative_tokens, &["city", "skyline", "building", "urban"])
    {
        let base_y = (height as f32 * 0.68) as i32;
        let step = (width / 24).max(18);
        let mut x = 0;
        while x < width {
            let w = rng.gen_range((width / 28).max(16)..(width / 14).max(26));
            let h = rng.gen_range((height / 8).max(24)..(height / 2).max(48));
            draw_rect(img, x, base_y - h, x + w, base_y, [24, 28, 46, 215]);

            let mut wx = x + 4;
            while wx < (x + w - 2).min(width - 2) {
                let mut wy = base_y - h + 4;
                while wy < base_y - 4 {
                    if rng.gen_bool(0.45) {
                        draw_rect(img, wx, wy, wx + 3, wy + 4, [255, 214, 120, 200]);
                    }
                    wy += 10;
                }
                wx += 8;
            }
            x += step;
        }
    }

    if car_scene && !has_any(negative_tokens, CAR_TOKENS) {
        let road_top = (height as f32 * 0.6) as i32;
        draw_polygon(
            img,
            &[
                (0, height),
                (width, height),
                ((width as f32 * 0.72) as i32, road_top),
                ((width as f32 * 0.28) as i32, road_top),
            ],
            [42, 44, 52, 238],
        );
        draw_line(
            img,
            (width as f32 * 0.29) as i32,
            road_top,
            (width as f32 * 0.71) as i32,
            road_top,
            [115, 120, 132, 110],
            (width / 280).max(2),
        );
        draw_line(
            img,
            (width as f32 * 0.15) as i32,
            height - 2,
            (width as f32 * 0.32) as i32,
            road_top,
            [84, 88, 96, 160],
            (width / 180).max(2),
        );
        draw_line(
            img,
            (width as f32 * 0.85) as i32,
            height - 2,
            (width as f32 * 0.68) as i32,
            road_top,
            [84, 88, 96, 160],
            (width / 180).max(2),
        );

        let stripe_w = (width / 140).max(4);
        let lane_x = width / 2;
        for idx in 0..5 {
            let t0 = idx as f32 / 5.0;
            let t1 = ((idx as f32) + 0.52) / 5.0;
            let y0 = road_top + ((height - road_top) as f32 * t0) as i32;
            let y1 = road_top + ((height - road_top) as f32 * t1.min(1.0)) as i32;
            let perspective = 1.0 + idx as f32 * 0.28;
            let w = (stripe_w as f32 * perspective) as i32;
            draw_polygon(
                img,
                &[
                    (lane_x - w, y1),
                    (lane_x + w, y1),
                    (lane_x + (w / 2).max(1), y0),
                    (lane_x - (w / 2).max(1), y0),
                ],
                [248, 232, 168, 210],
            );
        }

        for light_x in [(width as f32 * 0.18) as i32, (width as f32 * 0.82) as i32] {
            for ring in (1..=3).rev() {
                let rr = ring * (width / 100).max(8);
                draw_ellipse(
                    img,
                    light_x,
                    road_top,
                    rr,
                    (rr / 2).max(1),
                    [255, 180, 96, (22 * ring) as u8],
                );
            }
        }
    }

    if has_any(tokens, &["space", "night", "stars", "galaxy", "cosmic"])
        && !has_any(negative_tokens, &["space", "night", "stars", "galaxy", "cosmic"])
    {
        for _ in 0..(width / 3).max(48) {
            let x = rng.gen_range(0..width);
            let y = rng.gen_range(0..((height as f32 * 0.75) as i32).max(1));
            let r = rng.gen_range(1..=2);
            draw_circle(img, x, y, r, [255, 255, 255, rng.gen_range(90..=210) as u8]);
        }
    }

    if has_any(tokens, &["moon", "luna", "lunar"])
        && !has_any(negative_tokens, &["moon", "luna", "lunar"])
    {
        paint_moon(img, width, height);
    }

    if has_any(tokens, &["sun", "glow", "sunset", "golden"])
        && !has_any(negative_tokens, &["sun", "glow", "sunset", "golden"])
    {
        let radius = width / 10;
        let cx = (width as f32 * 0.75) as i32;
        let cy = (height as f32 * 0.22) as i32;
        for ring in (1..=5).rev() {
            let rr = radius * ring / 2;
            draw_circle(img, cx, cy, rr, [255, 220, 140, (18 * ring) as u8]);
        }
    }

    if has_any(tokens, &["cloud", "clouds", "overcast", "mist", "fog", "storm"])
        && !has_any(negative_tokens, &["cloud", "clouds", "overcast", "mist", "fog", "storm"])
    {
        paint_clouds(img, rng, width, height, tokens);
    }

    if has_any(tokens, &["rain", "rainy", "storm", "drizzle"])
        && !has_any(negative_tokens, &["rain", "rainy", "storm", "drizzle"])
    {
        paint_rain(img, rng, width, height);
    }

    if has_any(tokens, &["snow", "snowy", "winter", "blizzard"])
        && !has_any(negative_tokens, &["snow", "snowy", "winter", "blizzard"])
    {
        paint_snow(img, rng, width, height);
    }

    if has_any(tokens, BIRD_TOKENS) && !has_any(negative_tokens, BIRD_TOKENS) {
        paint_birds(img, rng, width, height);
    }

    if has_any(tokens, FLOWER_TOKENS) && !has_any(negative_tokens, FLOWER_TOKENS) {
        paint_flowers(img, rng, width, height, tokens);
    }
}

fn paint_water(
    img: &mut RgbaImage,
    width: i32,
    height: i32,
    horizon: i32,
    tokens: &BTreeSet<String>,
    palette: &[Rgb],
) {
    let base = palette[(2).min(palette.len() - 1)];
    let deep = [base[0] / 2, base[1].saturating_add(18), base[2].saturating_add(34), 220];
    let water_top = if has_any(tokens, MOUNTAIN_TOKENS) {
        (height as f32 * 0.70) as i32
    } else {
        (height as f32 * 0.64) as i32
    }
    .max(horizon - height / 16);

    draw_rect(img, 0, water_top, width, height, deep);

    for band in 0..8 {
        let y = water_top + band * ((height - water_top) / 10).max(6);
        let alpha = (78 - band * 8).clamp(18, 78) as u8;
        draw_line(img, 0, y, width, y + band / 2, [220, 236, 255, alpha], 1);
    }

    if has_any(tokens, &["beach", "shore", "sand"]) {
        let beach_top = water_top - (height / 18).max(10);
        draw_rect(img, 0, beach_top, width, water_top, [214, 192, 132, 198]);
    }
}

fn paint_mountains(img: &mut RgbaImage, width: i32, height: i32, tokens: &BTreeSet<String>) {
    let mut layer = transparent_layer_like(img);
    let ridge_y = (height as f32 * 0.68) as i32;

    draw_polygon(
        &mut layer,
        &[
            (-width / 10, ridge_y + height / 12),
            (width / 7, (height as f32 * 0.46) as i32),
            (width / 3, ridge_y + height / 18),
            (width / 2, (height as f32 * 0.40) as i32),
            ((width as f32 * 0.72) as i32, ridge_y + height / 20),
            ((width as f32 * 0.88) as i32, (height as f32 * 0.44) as i32),
            (width + width / 10, ridge_y + height / 10),
        ],
        [56, 72, 98, 215],
    );

    draw_polygon(
        &mut layer,
        &[
            (-width / 10, ridge_y + height / 8),
            (width / 9, (height as f32 * 0.54) as i32),
            ((width as f32 * 0.28) as i32, ridge_y + height / 12),
            ((width as f32 * 0.46) as i32, (height as f32 * 0.48) as i32),
            ((width as f32 * 0.64) as i32, ridge_y + height / 14),
            ((width as f32 * 0.82) as i32, (height as f32 * 0.52) as i32),
            (width + width / 10, ridge_y + height / 8),
        ],
        [92, 108, 128, 200],
    );

    if has_any(tokens, &["snow", "snowy", "winter", "alpine"]) {
        let peak_y = (height as f32 * 0.40) as i32;
        draw_polygon(
            &mut layer,
            &[
                (width / 2 - width / 20, peak_y + height / 30),
                (width / 2, peak_y),
                (width / 2 + width / 20, peak_y + height / 30),
                (width / 2 + width / 36, peak_y + height / 18),
                (width / 2 - width / 36, peak_y + height / 16),
            ],
            [245, 248, 255, 220],
        );
    }

    alpha_composite(img, &layer);
}

fn paint_clouds(
    img: &mut RgbaImage,
    rng: &mut rand::rngs::StdRng,
    width: i32,
    height: i32,
    tokens: &BTreeSet<String>,
) {
    let mut layer = transparent_layer_like(img);
    let stormy = has_any(tokens, &["storm", "overcast"]);
    let color = if stormy {
        [132, 140, 154, 96]
    } else {
        [244, 246, 252, 112]
    };

    for idx in 0..5 {
        let cx = (width / 8) + idx * (width / 5) + rng.gen_range(-(width / 24)..=(width / 24));
        let cy = (height / 8) + rng.gen_range(-(height / 40)..=(height / 28));
        let rx = (width / 12).max(26) + rng.gen_range(0..=(width / 18).max(6));
        let ry = (height / 20).max(12) + rng.gen_range(0..=(height / 30).max(4));
        draw_ellipse(&mut layer, cx, cy, rx, ry, color);
        draw_ellipse(&mut layer, cx - rx / 2, cy + ry / 6, rx / 2, (ry * 4 / 5).max(8), color);
        draw_ellipse(&mut layer, cx + rx / 2, cy + ry / 8, rx / 2, (ry * 3 / 4).max(8), color);
    }

    alpha_composite(img, &layer);
}

fn paint_moon(img: &mut RgbaImage, width: i32, height: i32) {
    let mut layer = transparent_layer_like(img);
    let cx = (width as f32 * 0.78) as i32;
    let cy = (height as f32 * 0.20) as i32;
    let radius = (width / 18).max(18);

    for ring in (1..=4).rev() {
        draw_circle(
            &mut layer,
            cx,
            cy,
            radius + ring * (width / 120).max(6),
            [196, 214, 255, (20 * ring) as u8],
        );
    }
    draw_circle(&mut layer, cx, cy, radius, [240, 244, 250, 225]);
    draw_circle(&mut layer, cx + radius / 3, cy - radius / 6, radius - radius / 3, [0, 0, 0, 90]);

    alpha_composite(img, &layer);
}

fn paint_rain(img: &mut RgbaImage, rng: &mut rand::rngs::StdRng, width: i32, height: i32) {
    for _ in 0..(width / 10).max(40) {
        let x = rng.gen_range(0..width);
        let y = rng.gen_range(0..height);
        let len = rng.gen_range((height / 26).max(10)..=(height / 18).max(18));
        draw_line(img, x, y, x - len / 3, y + len, [196, 216, 255, 86], 1);
    }
}

fn paint_snow(img: &mut RgbaImage, rng: &mut rand::rngs::StdRng, width: i32, height: i32) {
    for _ in 0..(width / 6).max(64) {
        let x = rng.gen_range(0..width);
        let y = rng.gen_range(0..height);
        let r = rng.gen_range(1..=3);
        draw_circle(img, x, y, r, [248, 250, 255, rng.gen_range(140..=230) as u8]);
    }
}

fn paint_birds(img: &mut RgbaImage, rng: &mut rand::rngs::StdRng, width: i32, height: i32) {
    let mut layer = transparent_layer_like(img);
    for idx in 0..rng.gen_range(5..=9) {
        let x = width / 6 + idx * (width / 10) + rng.gen_range(-(width / 24)..=(width / 24));
        let y = height / 5 + rng.gen_range(-(height / 24)..=(height / 18));
        let span = (width / 45).max(8) + rng.gen_range(0..=(width / 80).max(3));
        draw_polyline(&mut layer, &[(x - span, y + span / 4), (x, y), (x + span, y + span / 4)], [28, 28, 36, 180], 2);
    }
    alpha_composite(img, &layer);
}

fn paint_flowers(
    img: &mut RgbaImage,
    rng: &mut rand::rngs::StdRng,
    width: i32,
    height: i32,
    tokens: &BTreeSet<String>,
) {
    let mut layer = transparent_layer_like(img);
    let warm = has_any(tokens, &["rose", "roses", "sunset", "golden"]);
    let petal = if warm {
        [244, 124, 148, 210]
    } else {
        [232, 188, 238, 210]
    };

    for _ in 0..(width / 18).max(16) {
        let x = rng.gen_range(0..width);
        let stem_h = rng.gen_range((height / 12).max(18)..=(height / 7).max(28));
        let y0 = height - 1;
        let y1 = height - stem_h;
        draw_line(&mut layer, x, y0, x + rng.gen_range(-6..=6), y1, [74, 150, 76, 185], 2);
        let head_x = x + rng.gen_range(-6..=6);
        let head_y = y1;
        draw_circle(&mut layer, head_x, head_y, (width / 160).max(5), petal);
        draw_circle(&mut layer, head_x - 5, head_y + 2, (width / 240).max(3), petal);
        draw_circle(&mut layer, head_x + 5, head_y + 2, (width / 240).max(3), petal);
        draw_circle(&mut layer, head_x, head_y + 5, (width / 260).max(2), [248, 224, 132, 220]);
    }

    alpha_composite(img, &layer);
}

fn render_semantic_subject(
    img: &mut RgbaImage,
    rng: &mut rand::rngs::StdRng,
    tokens: &BTreeSet<String>,
    negative_tokens: &BTreeSet<String>,
    prompt: &str,
    palette: &[Rgb],
    style: &str,
) {
    if has_any(tokens, &["dog", "puppy", "retriever", "canine"])
        && !has_any(negative_tokens, &["dog", "puppy", "retriever", "canine"])
    {
        render_dog(img, rng, tokens, prompt, style);
        return;
    }
    if has_any(tokens, &["cat", "kitten", "feline"])
        && !has_any(negative_tokens, &["cat", "kitten", "feline"])
    {
        render_cat(img, tokens);
        return;
    }
    if has_any(tokens, &["car", "sportscar", "vehicle", "supercar", "coupe"])
        && !has_any(negative_tokens, &["car", "sportscar", "vehicle", "supercar", "coupe"])
    {
        render_sportscar(img, rng, tokens, style);
        return;
    }
    if has_any(tokens, BOAT_TOKENS) && !has_any(negative_tokens, BOAT_TOKENS) {
        render_sailboat(img, tokens, style);
        return;
    }
    if has_any(tokens, CASTLE_TOKENS) && !has_any(negative_tokens, CASTLE_TOKENS) {
        render_castle(img, tokens, style);
        return;
    }
    if has_any(tokens, &["tree", "forest"]) && !has_any(negative_tokens, &["tree", "forest"]) {
        render_tree(img, rng, palette);
    }
}

fn render_sportscar(
    img: &mut RgbaImage,
    _rng: &mut rand::rngs::StdRng,
    tokens: &BTreeSet<String>,
    style: &str,
) {
    let width = img.width() as i32;
    let height = img.height() as i32;
    let mut layer = transparent_layer_like(img);
    let mut glow = transparent_layer_like(img);

    let body: RgbaColor = if has_any(tokens, &["yellow", "gold", "golden"]) {
        [238, 194, 48, 255]
    } else if has_any(tokens, &["blue", "azure", "cobalt"]) {
        [42, 112, 220, 255]
    } else if has_any(tokens, &["white", "silver", "chrome"]) {
        [224, 228, 236, 255]
    } else if has_any(tokens, &["black", "midnight", "dark"]) {
        [34, 34, 42, 255]
    } else if has_any(tokens, &["green", "emerald"]) {
        [32, 138, 96, 255]
    } else {
        [214, 42, 38, 255]
    };
    let shadow = [
        body[0].saturating_sub(70),
        body[1].saturating_sub(70),
        body[2].saturating_sub(70),
        255,
    ];
    let mid = [
        ((body[0] as f32 * 0.82) + 24.0).clamp(0.0, 255.0) as u8,
        ((body[1] as f32 * 0.82) + 24.0).clamp(0.0, 255.0) as u8,
        ((body[2] as f32 * 0.82) + 24.0).clamp(0.0, 255.0) as u8,
        255,
    ];
    let highlight = [
        body[0].saturating_add(55),
        body[1].saturating_add(55),
        body[2].saturating_add(55),
        210,
    ];

    let ground = (height as f32 * 0.78) as i32;
    let car_w = (width as f32 * 0.54) as i32;
    let car_h = (height as f32 * 0.16) as i32;
    let cx = width / 2;
    let x0 = cx - car_w / 2;
    let x1 = cx + car_w / 2;
    let roof_y = ground - (car_h as f32 * 1.45) as i32;
    let nose_y = ground - (car_h as f32 * 0.78) as i32;
    let deck_y = ground - (car_h as f32 * 1.02) as i32;
    let rear_x = x0 + (car_w as f32 * 0.26) as i32;
    let front_x = x0 + (car_w as f32 * 0.77) as i32;
    let wheel_r = (car_h / 2).max(20);
    let wheel_y = ground - wheel_r / 4;

    draw_ellipse(&mut layer, (x0 + x1) / 2, ground + 10, ((x1 - x0) / 2).max(32), 18, [18, 16, 18, 95]);
    draw_polygon(
        &mut layer,
        &[
            (x0 + (car_w as f32 * 0.03) as i32, ground - (car_h as f32 * 0.18) as i32),
            (x0 + (car_w as f32 * 0.10) as i32, deck_y),
            (x0 + (car_w as f32 * 0.27) as i32, roof_y + (car_h as f32 * 0.06) as i32),
            (x0 + (car_w as f32 * 0.47) as i32, roof_y),
            (x0 + (car_w as f32 * 0.66) as i32, roof_y + (car_h as f32 * 0.12) as i32),
            (x0 + (car_w as f32 * 0.83) as i32, nose_y),
            (x1 - (car_w as f32 * 0.02) as i32, ground - (car_h as f32 * 0.20) as i32),
            (x1 - (car_w as f32 * 0.08) as i32, ground - (car_h as f32 * 0.05) as i32),
            (x0 + (car_w as f32 * 0.11) as i32, ground - (car_h as f32 * 0.03) as i32),
        ],
        body,
    );
    draw_polygon(
        &mut layer,
        &[
            (x0 + (car_w as f32 * 0.40) as i32, ground - (car_h as f32 * 0.48) as i32),
            (x0 + (car_w as f32 * 0.63) as i32, ground - (car_h as f32 * 0.50) as i32),
            (x0 + (car_w as f32 * 0.58) as i32, ground - (car_h as f32 * 0.25) as i32),
            (x0 + (car_w as f32 * 0.38) as i32, ground - (car_h as f32 * 0.22) as i32),
        ],
        shadow,
    );
    draw_polygon(
        &mut layer,
        &[
            (x0 + (car_w as f32 * 0.06) as i32, ground - (car_h as f32 * 0.12) as i32),
            (x1 - (car_w as f32 * 0.08) as i32, ground - (car_h as f32 * 0.12) as i32),
            (x1 - (car_w as f32 * 0.12) as i32, ground - (car_h as f32 * 0.02) as i32),
            (x0 + (car_w as f32 * 0.12) as i32, ground - (car_h as f32 * 0.02) as i32),
        ],
        mid,
    );
    draw_polygon(
        &mut layer,
        &[
            (x0 + (car_w as f32 * 0.27) as i32, roof_y + (car_h as f32 * 0.10) as i32),
            (x0 + (car_w as f32 * 0.46) as i32, roof_y + (car_h as f32 * 0.02) as i32),
            (x0 + (car_w as f32 * 0.65) as i32, roof_y + (car_h as f32 * 0.18) as i32),
            (x0 + (car_w as f32 * 0.71) as i32, ground - (car_h as f32 * 0.63) as i32),
            (x0 + (car_w as f32 * 0.39) as i32, ground - (car_h as f32 * 0.60) as i32),
        ],
        [44, 60, 84, 235],
    );
    draw_line(
        &mut layer,
        x0 + (car_w as f32 * 0.50) as i32,
        roof_y + (car_h as f32 * 0.05) as i32,
        x0 + (car_w as f32 * 0.47) as i32,
        ground - (car_h as f32 * 0.61) as i32,
        [190, 210, 232, 150],
        (width / 320).max(2),
    );
    draw_polyline(
        &mut layer,
        &[
            (x0 + (car_w as f32 * 0.14) as i32, ground - (car_h as f32 * 0.74) as i32),
            (x0 + (car_w as f32 * 0.43) as i32, roof_y + (car_h as f32 * 0.08) as i32),
            (x0 + (car_w as f32 * 0.79) as i32, nose_y + (car_h as f32 * 0.02) as i32),
        ],
        highlight,
        (width / 300).max(2),
    );
    draw_polyline(
        &mut layer,
        &[
            (x0 + (car_w as f32 * 0.19) as i32, ground - (car_h as f32 * 0.42) as i32),
            (x0 + (car_w as f32 * 0.45) as i32, ground - (car_h as f32 * 0.56) as i32),
            (x0 + (car_w as f32 * 0.75) as i32, ground - (car_h as f32 * 0.44) as i32),
        ],
        [255, 255, 255, 90],
        (width / 420).max(1),
    );

    for wheel_x in [rear_x, front_x] {
        draw_circle(&mut layer, wheel_x, wheel_y, wheel_r, [20, 20, 24, 255]);
        let rim_r = ((wheel_r as f32) * 0.58) as i32;
        draw_circle(&mut layer, wheel_x, wheel_y, rim_r, [172, 176, 186, 255]);
        let hub_r = (wheel_r / 6).max(4);
        draw_circle(&mut layer, wheel_x, wheel_y, hub_r, [110, 112, 118, 255]);
        for spoke in 0..5 {
            let angle = spoke as f32 * (std::f32::consts::PI / 2.5) + 0.18;
            let sx = (angle.cos() * (rim_r - 3) as f32) as i32;
            let sy = (angle.sin() * (rim_r - 3) as f32) as i32;
            draw_line(&mut layer, wheel_x, wheel_y, wheel_x + sx, wheel_y + sy, [228, 230, 236, 220], (width / 600).max(1));
        }
    }

    draw_arc_band(
        &mut layer,
        rear_x,
        wheel_y - 8,
        wheel_r + 8,
        188.0,
        354.0,
        shadow,
        (width / 170).max(5),
    );
    draw_arc_band(
        &mut layer,
        front_x,
        wheel_y - 10,
        wheel_r + 10,
        186.0,
        354.0,
        shadow,
        (width / 170).max(5),
    );

    let headlight = (x1 - (car_w as f32 * 0.03) as i32, ground - (car_h as f32 * 0.28) as i32);
    let taillight = (x0 + (car_w as f32 * 0.05) as i32, ground - (car_h as f32 * 0.35) as i32);
    draw_ellipse(&mut layer, headlight.0 - 2, headlight.1, 12, 6, [245, 250, 255, 240]);
    draw_ellipse(&mut layer, taillight.0, taillight.1, 8, 5, [255, 70, 52, 220]);

    for ring in (1..=4).rev() {
        let glow_r = ring * (width / 130).max(8);
        draw_ellipse(
            &mut glow,
            headlight.0,
            headlight.1,
            glow_r,
            (glow_r / 2).max(1),
            [255, 244, 220, (20 * ring) as u8],
        );
    }
    if has_any(tokens, &["neon", "night", "city", "cyberpunk"]) {
        for ring in (1..=3).rev() {
            let rr = ring * (width / 160).max(6);
            draw_ellipse(
                &mut glow,
                taillight.0,
                taillight.1,
                rr,
                (rr / 2).max(1),
                [255, 64, 80, (18 * ring) as u8],
            );
        }
    }

    if matches!(style, "photo" | "neon" | "oil_paint") {
        let streak_y = ground - (car_h as f32 * 0.18) as i32;
        for idx in 0..6 {
            let streak_x = x0 - (car_w as f32 * 0.08) as i32 - idx * (width / 80).max(10);
            let streak_len = (car_w as f32 * (0.08 + idx as f32 * 0.02)) as i32;
            draw_line(
                &mut layer,
                streak_x,
                streak_y + idx * 3,
                streak_x + streak_len,
                streak_y + idx,
                [255, 255, 255, (82 - idx * 10).max(24) as u8],
                (width / 700).max(1),
            );
        }
    }

    alpha_composite(img, &layer);
    let bloom = blur(&glow, 6.0);
    alpha_composite(img, &bloom);
}

fn render_dog(
    img: &mut RgbaImage,
    rng: &mut rand::rngs::StdRng,
    tokens: &BTreeSet<String>,
    _prompt: &str,
    style: &str,
) {
    if has_any(tokens, &["portrait", "closeup", "close", "headshot"]) {
        render_dog_portrait(img, rng, tokens, style);
        return;
    }

    let width = img.width() as i32;
    let height = img.height() as i32;
    let mut layer = transparent_layer_like(img);
    let mut fur = transparent_layer_like(img);

    let golden = has_any(tokens, &["golden", "retriever"]);
    let sitting = has_any(tokens, &["sit", "sitting", "portrait"]);
    let fur_base = if golden { [224, 188, 122, 255] } else { [176, 130, 84, 255] };
    let fur_mid = if golden { [202, 156, 95, 255] } else { [152, 108, 66, 255] };
    let fur_shadow = if golden { [132, 90, 50, 255] } else { [112, 76, 44, 255] };
    let fur_light = if golden { [246, 226, 188, 230] } else { [228, 198, 158, 230] };

    let cx = (width as f32 * 0.5) as i32;
    let ground = (height as f32 * 0.74) as i32;
    let body_w = (width as f32 * if sitting { 0.32 } else { 0.36 }) as i32;
    let body_h = (height as f32 * if sitting { 0.25 } else { 0.20 }) as i32;
    let torso_top = ground - body_h;
    let body_cx = cx + if sitting { 10 } else { 0 };

    draw_ellipse(
        &mut layer,
        body_cx - 4,
        ground + 3,
        body_w / 2 + 35,
        16,
        [25, 20, 16, 95],
    );
    draw_ellipse(&mut layer, body_cx, torso_top + body_h / 2 + 6, body_w / 2, body_h / 2, fur_base);
    draw_ellipse(
        &mut layer,
        body_cx - body_w / 4 - 12,
        torso_top + body_h / 2 + 16,
        body_w / 4 + 24,
        body_h / 2,
        fur_light,
    );
    draw_ellipse(
        &mut layer,
        body_cx + body_w / 4 + 8,
        torso_top + body_h / 2 + 8,
        body_w / 4,
        body_h / 2 - 6,
        fur_shadow,
    );

    draw_rect(&mut layer, body_cx - body_w / 2 - 10, torso_top + 22, body_cx - body_w / 6, torso_top + body_h / 2 + 10, fur_mid);

    let head_cx = body_cx - body_w / 2 - (width as f32 * 0.065) as i32;
    let head_cy = torso_top + (body_h as f32 * 0.4) as i32;
    let head_rx = (width as f32 * 0.085) as i32;
    let head_ry = (height as f32 * 0.08) as i32;
    draw_ellipse(&mut layer, head_cx, head_cy, head_rx, head_ry, fur_base);

    let muzzle_w = head_rx;
    let muzzle_h = (head_ry as f32 * 0.65) as i32;
    let muzzle_x0 = head_cx - head_rx - muzzle_w / 2;
    let muzzle_y0 = head_cy + head_ry / 4;
    draw_ellipse(
        &mut layer,
        muzzle_x0 + muzzle_w / 2,
        muzzle_y0 + muzzle_h / 2,
        muzzle_w / 2,
        muzzle_h / 2,
        [242, 226, 198, 238],
    );
    draw_ellipse(&mut layer, muzzle_x0 + 1, muzzle_y0 + muzzle_h / 3 + 6, 7, 6, [20, 18, 18, 255]);
    draw_line(
        &mut layer,
        muzzle_x0 + 2,
        muzzle_y0 + muzzle_h / 2 + 8,
        muzzle_x0 + 10,
        muzzle_y0 + muzzle_h / 2 + 12,
        [30, 24, 20, 220],
        2,
    );

    let eye_x = head_cx - 4;
    let eye_y = head_cy - 4;
    draw_ellipse(&mut layer, eye_x, eye_y, 4, 4, [24, 16, 12, 255]);
    draw_ellipse(&mut layer, eye_x - 2, eye_y - 1, 1, 1, [255, 255, 255, 190]);
    draw_arc_band(&mut layer, eye_x - 1, eye_y - 1, 10, 190.0, 355.0, fur_shadow, 2);

    draw_polygon(
        &mut layer,
        &[
            (head_cx + 2, head_cy - head_ry + 6),
            (head_cx + 22, head_cy - 8),
            (head_cx + 10, head_cy + head_ry + 12),
            (head_cx - 8, head_cy + 10),
        ],
        fur_shadow,
    );

    let leg_top = torso_top + body_h / 2 + 16;
    if sitting {
        for lx in [body_cx - body_w / 4, body_cx - body_w / 9] {
            draw_rect(&mut layer, lx, leg_top, lx + 18, ground + body_h / 2, fur_mid);
            draw_ellipse(&mut layer, lx + 10, ground + body_h / 2, 12, 6, [72, 52, 30, 255]);
        }
        let rear_x = body_cx + body_w / 7;
        draw_ellipse(&mut layer, rear_x + 13, ground + 12, 21, 14, fur_shadow);
        draw_ellipse(&mut layer, rear_x + 14, ground + 25, 16, 9, [70, 52, 34, 255]);
    } else {
        for lx in [
            body_cx - body_w / 3,
            body_cx - body_w / 8,
            body_cx + body_w / 8,
            body_cx + body_w / 3,
        ] {
            draw_rect(&mut layer, lx, leg_top, lx + 16, ground + body_h / 3, fur_mid);
            draw_ellipse(&mut layer, lx + 8, ground + body_h / 3, 11, 5, [72, 52, 32, 255]);
        }
    }

    let tail_base = (body_cx + body_w / 2 - 8, torso_top + 28);
    let tail_mid = ((tail_base.0 + body_cx + body_w / 2 + 42) / 2, torso_top - 4);
    let tail_tip = (body_cx + body_w / 2 + 42, torso_top - 12);
    draw_polyline(&mut layer, &[tail_base, tail_mid, tail_tip], fur_shadow, (width / 56).max(8));
    draw_ellipse(&mut layer, tail_tip.0, tail_tip.1 + 2, 10, 8, fur_light);

    let stroke_count = if matches!(style, "photo" | "oil_paint") { 420 } else { 210 };
    let x0 = body_cx - body_w / 2 - 24;
    let x1 = body_cx + body_w / 2 + 30;
    let y0 = torso_top - 10;
    let y1 = ground + body_h / 2;
    for _ in 0..stroke_count {
        let px = rng.gen_range(x0..x1.max(x0 + 1));
        let py = rng.gen_range(y0..y1.max(y0 + 1));
        let dx = rng.gen_range(-3..=3);
        let dy = rng.gen_range(3..=10);
        let shade = if rng.gen_bool(0.45) { fur_shadow } else { fur_light };
        draw_line(&mut fur, px, py, px + dx, py + dy, shade, 1);
    }

    for wx in [muzzle_x0 + 8, muzzle_x0 + 14, muzzle_x0 + 20] {
        draw_ellipse(&mut layer, wx + 1, muzzle_y0 + muzzle_h / 2 + 3, 1, 1, [86, 66, 46, 180]);
        draw_line(
            &mut layer,
            wx + 2,
            muzzle_y0 + muzzle_h / 2 + 3,
            wx + 14,
            muzzle_y0 + muzzle_h / 2 + 1,
            [130, 110, 90, 120],
            1,
        );
    }

    alpha_composite(img, &layer);
    alpha_composite(img, &fur);
}

fn render_dog_portrait(
    img: &mut RgbaImage,
    rng: &mut rand::rngs::StdRng,
    tokens: &BTreeSet<String>,
    style: &str,
) {
    let width = img.width() as i32;
    let height = img.height() as i32;
    let mut layer = transparent_layer_like(img);
    let mut fur = transparent_layer_like(img);
    let mut sss = transparent_layer_like(img);

    let golden = has_any(tokens, &["golden", "retriever"]);
    let fur_base = if golden { [224, 188, 122, 255] } else { [176, 130, 84, 255] };
    let fur_mid = if golden { [202, 156, 95, 255] } else { [152, 108, 66, 255] };
    let fur_shadow = if golden { [132, 90, 50, 255] } else { [112, 76, 44, 255] };
    let fur_light = if golden { [246, 226, 188, 230] } else { [228, 198, 158, 230] };

    let cx = width / 2;
    let cy = (height as f32 * 0.56) as i32;
    let head_r = ((width.min(height) as f32) * 0.16) as i32;

    draw_ellipse(&mut layer, cx, cy + head_r, head_r + 40, head_r / 2 + 60, fur_light);
    draw_ellipse(&mut layer, cx, cy + head_r + 30, head_r + 20, head_r / 2 + 54, fur_base);

    draw_polygon(
        &mut layer,
        &[
            (cx - head_r + 8, cy - head_r + 20),
            (cx - head_r - 70, cy - 10),
            (cx - head_r - 34, cy + head_r + 24),
            (cx - head_r + 14, cy + head_r / 3),
        ],
        fur_shadow,
    );
    draw_polygon(
        &mut layer,
        &[
            (cx + head_r - 8, cy - head_r + 20),
            (cx + head_r + 70, cy - 10),
            (cx + head_r + 34, cy + head_r + 24),
            (cx + head_r - 14, cy + head_r / 3),
        ],
        fur_shadow,
    );

    draw_circle(&mut layer, cx, cy, head_r, fur_base);
    draw_ellipse(&mut layer, cx, cy - head_r / 2 + 5, head_r - 12, head_r / 2, fur_mid);

    let muzzle_w = (head_r as f32 * 1.35) as i32;
    let muzzle_h = (head_r as f32 * 0.9) as i32;
    let muzzle_y0 = cy + head_r / 5;
    draw_ellipse(
        &mut layer,
        cx,
        muzzle_y0 + muzzle_h / 2,
        muzzle_w / 2,
        muzzle_h / 2,
        [243, 226, 198, 238],
    );

    let nose_x = cx;
    let nose_y = muzzle_y0 + muzzle_h / 3;
    draw_ellipse(&mut layer, nose_x, nose_y + 1, 14, 10, [22, 18, 18, 255]);
    draw_line(&mut layer, nose_x, nose_y + 6, nose_x, nose_y + 22, [32, 22, 20, 220], 2);
    draw_arc_band(&mut layer, nose_x - 10, nose_y + 24, 12, 300.0, 80.0 + 360.0, [34, 24, 20, 220], 2);
    draw_arc_band(&mut layer, nose_x + 10, nose_y + 24, 12, 100.0, 240.0, [34, 24, 20, 220], 2);

    let eye_y = cy - head_r / 4;
    let left_eye_x = cx - head_r / 3;
    let right_eye_x = cx + head_r / 3;
    let er = (head_r / 5).max(6);
    let iris = if golden { [118, 72, 28, 255] } else { [66, 48, 30, 255] };
    for ex in [left_eye_x, right_eye_x] {
        draw_ellipse(&mut layer, ex, eye_y, er + 2, er - 1, [238, 232, 218, 255]);
        draw_ellipse(&mut layer, ex, eye_y, er - 1, er - 2, iris);
        let pupil = (er / 2).max(3);
        draw_circle(&mut layer, ex, eye_y, pupil, [10, 8, 6, 255]);
        draw_ellipse(&mut layer, ex + pupil / 2 + 1, eye_y - pupil / 2 - 1, 3, 2, [255, 255, 255, 230]);
        draw_ellipse(&mut layer, ex - pupil / 2 - 1, eye_y + pupil / 3 + 2, 2, 2, [190, 210, 235, 130]);
        draw_arc_band(&mut layer, ex, eye_y - er / 2, er + 6, 195.0, 345.0, fur_shadow, 2);
    }

    let stroke_count = if matches!(style, "photo" | "oil_paint") { 560 } else { 280 };
    let x0 = cx - head_r - 70;
    let x1 = cx + head_r + 70;
    let y0 = cy - head_r - 10;
    let y1 = cy + head_r + 120;
    for i in 0..stroke_count {
        let px = rng.gen_range(x0..x1.max(x0 + 1));
        let py = rng.gen_range(y0..y1.max(y0 + 1));
        let dx_raw = (px - cx) as f32;
        let dy_raw = (py - (cy - head_r / 2)) as f32;
        let base_angle = dy_raw.atan2(dx_raw);
        let noise_angle = (i as f32 * 1.132).sin() * 0.38 + (i as f32 * 0.731).cos() * 0.18;
        let flow_angle = base_angle + noise_angle;
        let stroke_len = rng.gen_range(7..=14) as f32;
        let dx = (flow_angle.cos() * stroke_len) as i32;
        let dy = (flow_angle.sin() * stroke_len) as i32;
        let shade = if rng.gen_bool(0.5) { fur_light } else { fur_shadow };
        draw_line(&mut fur, px, py, px + dx, py + dy, shade, 1);
    }

    let sss_cx = cx;
    let sss_cy = muzzle_y0 + muzzle_h / 2;
    for ring in (1..=4).rev() {
        let r_sss = (muzzle_w as f32 * 0.55 * ring as f32 / 4.0) as i32;
        let alpha = (28 * (5 - ring)) as u8;
        draw_ellipse(&mut sss, sss_cx, sss_cy, r_sss, (r_sss / 2).max(1), [255, 190, 140, alpha]);
    }

    for wx in [cx - 16, cx - 6, cx + 4, cx + 14] {
        draw_ellipse(&mut layer, wx + 1, muzzle_y0 + muzzle_h / 2 + 5, 1, 1, [88, 68, 48, 180]);
    }

    alpha_composite(img, &layer);
    alpha_composite(img, &fur);
    alpha_composite(img, &sss);
}

fn render_cat(img: &mut RgbaImage, _tokens: &BTreeSet<String>) {
    let width = img.width() as i32;
    let height = img.height() as i32;
    let mut layer = transparent_layer_like(img);
    let base = [160, 158, 162, 255];
    let shadow = [96, 92, 100, 255];
    let cx = width / 2;
    let ground = (height as f32 * 0.74) as i32;

    draw_ellipse(&mut layer, cx - 20, ground - 45, 70, 55, base);
    draw_circle(&mut layer, cx - 83, ground - 102, 46, base);
    draw_polygon(
        &mut layer,
        &[(cx - 110, ground - 120), (cx - 126, ground - 168), (cx - 84, ground - 136)],
        shadow,
    );
    draw_polygon(
        &mut layer,
        &[(cx - 58, ground - 120), (cx - 40, ground - 168), (cx - 14, ground - 132)],
        shadow,
    );
    draw_polyline(&mut layer, &[(cx + 42, ground - 74), (cx + 84, ground - 130), (cx + 120, ground - 160)], shadow, 12);
    alpha_composite(img, &layer);
}

fn render_tree(img: &mut RgbaImage, rng: &mut rand::rngs::StdRng, palette: &[Rgb]) {
    let width = img.width() as i32;
    let height = img.height() as i32;
    let mut layer = transparent_layer_like(img);
    let x = width / 2;
    let trunk_w = (width / 22).max(10);
    draw_rect(
        &mut layer,
        x - trunk_w,
        height / 2,
        x + trunk_w,
        height - height / 8,
        [92, 56, 28, 255],
    );
    for _ in 0..18 {
        let r = rng.gen_range((width / 16).max(16)..(width / 8).max(32));
        let cx = x + rng.gen_range(-(width / 8)..=(width / 8));
        let cy = height / 2 + rng.gen_range(-(height / 5)..=(height / 12));
        let leaf = palette[rng.gen_range(0..palette.len())];
        draw_circle(&mut layer, cx, cy, r, [leaf[0], leaf[1], leaf[2], 180]);
    }
    alpha_composite(img, &layer);
}

fn render_sailboat(img: &mut RgbaImage, tokens: &BTreeSet<String>, style: &str) {
    let width = img.width() as i32;
    let height = img.height() as i32;
    let mut layer = transparent_layer_like(img);
    let mut glow = transparent_layer_like(img);
    let cx = width / 2;
    let waterline = if has_any(tokens, WATER_TOKENS) {
        (height as f32 * 0.76) as i32
    } else {
        (height as f32 * 0.80) as i32
    };
    let hull_w = (width as f32 * 0.22) as i32;
    let hull_h = (height as f32 * 0.05) as i32;
    let mast_h = (height as f32 * 0.30) as i32;
    let sail = if has_any(tokens, &["red", "crimson"]) {
        [224, 72, 68, 245]
    } else if has_any(tokens, &["black", "midnight"]) {
        [42, 44, 52, 245]
    } else if has_any(tokens, &["gold", "golden", "sunset"]) {
        [244, 218, 162, 245]
    } else {
        [244, 246, 248, 240]
    };

    draw_ellipse(&mut layer, cx, waterline + hull_h + 18, hull_w / 2, hull_h / 2, [18, 22, 30, 80]);
    draw_polygon(
        &mut layer,
        &[
            (cx - hull_w / 2, waterline),
            (cx + hull_w / 2, waterline),
            (cx + hull_w / 3, waterline + hull_h),
            (cx - hull_w / 3, waterline + hull_h),
        ],
        [88, 54, 34, 255],
    );
    draw_line(&mut layer, cx, waterline - mast_h, cx, waterline, [84, 66, 52, 255], (width / 320).max(2));
    draw_polygon(
        &mut layer,
        &[
            (cx - width / 60, waterline - mast_h + height / 80),
            (cx - width / 6, waterline - mast_h / 3),
            (cx - width / 60, waterline - mast_h / 6),
        ],
        sail,
    );
    draw_polygon(
        &mut layer,
        &[
            (cx + width / 70, waterline - mast_h + height / 28),
            (cx + width / 9, waterline - mast_h / 2),
            (cx + width / 70, waterline - mast_h / 7),
        ],
        [232, 236, 240, 224],
    );
    draw_line(&mut layer, cx - hull_w / 3, waterline + hull_h / 3, cx + hull_w / 3, waterline + hull_h / 3, [160, 116, 82, 210], 2);

    if matches!(style, "neon" | "photo") {
        for ring in (1..=3).rev() {
            draw_circle(&mut glow, cx, waterline - mast_h / 2, ring * (width / 90).max(6), [255, 234, 196, (20 * ring) as u8]);
        }
    }

    for idx in 0..4 {
        let y = waterline + hull_h + 6 + idx * (height / 70).max(4);
        draw_line(&mut layer, cx - hull_w / 2 - width / 14, y, cx + hull_w / 2 + width / 12, y, [224, 238, 250, (46 - idx * 8).max(12) as u8], 1);
    }

    alpha_composite(img, &layer);
    alpha_composite(img, &glow);
}

fn render_castle(img: &mut RgbaImage, tokens: &BTreeSet<String>, style: &str) {
    let width = img.width() as i32;
    let height = img.height() as i32;
    let mut layer = transparent_layer_like(img);
    let mut glow = transparent_layer_like(img);
    let cx = width / 2;
    let ground = if has_any(tokens, MOUNTAIN_TOKENS) {
        (height as f32 * 0.70) as i32
    } else {
        (height as f32 * 0.78) as i32
    };
    let body = [132, 132, 144, 245];
    let shadow = [90, 92, 106, 255];
    let window = if has_any(tokens, &["night", "moon", "midnight"]) || matches!(style, "neon") {
        [255, 214, 132, 220]
    } else {
        [182, 196, 214, 170]
    };
    let keep_w = (width as f32 * 0.18) as i32;
    let keep_h = (height as f32 * 0.24) as i32;
    let tower_w = (width as f32 * 0.10) as i32;
    let tower_h = (height as f32 * 0.32) as i32;

    draw_ellipse(&mut layer, cx, ground + 20, keep_w, height / 22, [20, 18, 22, 82]);
    draw_rect(&mut layer, cx - keep_w / 2, ground - keep_h, cx + keep_w / 2, ground, body);
    draw_rect(&mut layer, cx - keep_w / 2 - tower_w + 10, ground - tower_h, cx - keep_w / 2 + 12, ground, shadow);
    draw_rect(&mut layer, cx + keep_w / 2 - 12, ground - tower_h, cx + keep_w / 2 + tower_w - 10, ground, shadow);
    draw_rect(&mut layer, cx - keep_w, ground - keep_h / 2, cx + keep_w, ground, [108, 110, 124, 235]);

    for battlement in 0..5 {
        let x0 = cx - keep_w / 2 + battlement * (keep_w / 4);
        draw_rect(&mut layer, x0, ground - keep_h - height / 24, x0 + keep_w / 8, ground - keep_h, body);
    }
    for battlement in 0..3 {
        let left_x = cx - keep_w / 2 - tower_w + 10 + battlement * (tower_w / 3);
        let right_x = cx + keep_w / 2 - 12 + battlement * (tower_w / 3);
        draw_rect(&mut layer, left_x, ground - tower_h - height / 28, left_x + tower_w / 6, ground - tower_h, shadow);
        draw_rect(&mut layer, right_x, ground - tower_h - height / 28, right_x + tower_w / 6, ground - tower_h, shadow);
    }

    draw_rect(&mut layer, cx - width / 36, ground - height / 12, cx + width / 36, ground, [54, 38, 28, 255]);
    for (wx, wy) in [
        (cx - keep_w / 4, ground - keep_h / 2),
        (cx + keep_w / 4, ground - keep_h / 2),
        (cx - keep_w / 2 - tower_w / 3, ground - tower_h / 2),
        (cx + keep_w / 2 + tower_w / 3, ground - tower_h / 2),
    ] {
        draw_rect(&mut layer, wx - 6, wy - 10, wx + 6, wy + 10, window);
    }

    draw_line(&mut layer, cx - keep_w / 2 - tower_w / 2, ground - tower_h, cx - keep_w / 2 - tower_w / 2, ground - tower_h - height / 16, [118, 102, 90, 240], 2);
    draw_line(&mut layer, cx + keep_w / 2 + tower_w / 2, ground - tower_h, cx + keep_w / 2 + tower_w / 2, ground - tower_h - height / 18, [118, 102, 90, 240], 2);
    draw_polygon(&mut layer, &[(cx - keep_w / 2 - tower_w / 2, ground - tower_h - height / 16), (cx - keep_w / 2 - tower_w / 2 + width / 45, ground - tower_h - height / 20), (cx - keep_w / 2 - tower_w / 2, ground - tower_h - height / 26)], [164, 36, 48, 220]);
    draw_polygon(&mut layer, &[(cx + keep_w / 2 + tower_w / 2, ground - tower_h - height / 18), (cx + keep_w / 2 + tower_w / 2 + width / 48, ground - tower_h - height / 22), (cx + keep_w / 2 + tower_w / 2, ground - tower_h - height / 28)], [64, 108, 196, 220]);

    if matches!(style, "neon" | "photo") {
        for ring in (1..=2).rev() {
            draw_circle(&mut glow, cx, ground - keep_h / 2, ring * (width / 80).max(8), [255, 214, 120, (18 * ring) as u8]);
        }
    }

    alpha_composite(img, &layer);
    alpha_composite(img, &glow);
}

fn portrait_dof(img: &RgbaImage) -> RgbaImage {
    let width = img.width();
    let height = img.height();
    let cx = width as f32 * 0.5;
    let cy = height as f32 * 0.44;
    let subject_r = width.min(height) as f32 * 0.34;
    let blurred = blur(img, (width / 60).max(4) as f32);
    let mut result = img.clone();

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let t = ((dist - subject_r * 0.55) / (subject_r * 0.75)).clamp(0.0, 1.0);
            let mask = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
            let sharp = img.get_pixel(x, y).0;
            let soft = blurred.get_pixel(x, y).0;
            result.put_pixel(
                x,
                y,
                Rgba([
                    blend_channel(sharp[0], soft[0], mask),
                    blend_channel(sharp[1], soft[1], mask),
                    blend_channel(sharp[2], soft[2], mask),
                    255,
                ]),
            );
        }
    }

    result
}

fn portrait_lighting(img: &mut RgbaImage) {
    let width = img.width();
    let height = img.height();

    for y in 0..height {
        let yf = y as f32 / height as f32;
        for x in 0..width {
            let xf = x as f32 / width as f32;
            let key = (-(((xf - 0.22).powi(2) / 0.12) + ((yf - 0.20).powi(2) / 0.12))).exp() * 24.0;
            let fill = (-(((xf - 0.92).powi(2) / 0.18) + ((yf - 0.55).powi(2) / 0.28))).exp() * 14.0;
            let rim = ((xf * 0.55 + (1.0 - yf) * 0.45).powi(3)) * 18.0;

            let mut px = img.get_pixel(x, y).0;
            px[0] = add_light(px[0], key + rim * 0.85);
            px[1] = add_light(px[1], key * 0.72 + fill * 0.45 + rim * 0.78);
            px[2] = add_light(px[2], fill + rim * 0.55);
            img.put_pixel(x, y, Rgba(px));
        }
    }
}

fn add_depth_and_lighting(img: &mut RgbaImage, style: &str) {
    let width = img.width();
    let height = img.height();
    let mut overlay = RgbaImage::new(width, height);

    for y in 0..height {
        let row_color = if (y as f32) < height as f32 * 0.45 {
            [255, 190, 120, 255]
        } else {
            [24, 32, 50, 255]
        };
        for x in 0..width {
            overlay.put_pixel(x, y, Rgba(row_color));
        }
    }

    let overlay = blur(&overlay, (width / 40).max(8) as f32);
    let blend = if style == "photo" { 0.08 } else { 0.05 };
    for y in 0..height {
        for x in 0..width {
            let base = img.get_pixel(x, y).0;
            let over = overlay.get_pixel(x, y).0;
            img.put_pixel(
                x,
                y,
                Rgba([
                    blend_channel(base[0], over[0], blend),
                    blend_channel(base[1], over[1], blend),
                    blend_channel(base[2], over[2], blend),
                    255,
                ]),
            );
        }
    }
}

fn style_profile(style: &str) -> StyleProfile {
    match style {
        "pixel" => StyleProfile {
            blur: 0.0,
            contrast: 1.35,
            saturation: 1.55,
            sharpen: 1.2,
            pixelate: 8,
            vignette: 0.0,
            neon: false,
            greyscale: false,
            edge: false,
            paint: false,
            grain: 0.0,
        },
        "photo" => StyleProfile {
            blur: 0.25,
            contrast: 1.15,
            saturation: 1.05,
            sharpen: 1.5,
            pixelate: 0,
            vignette: 0.28,
            neon: false,
            greyscale: false,
            edge: false,
            paint: false,
            grain: 0.0,
        },
        "anime" => StyleProfile {
            blur: 0.15,
            contrast: 1.35,
            saturation: 1.85,
            sharpen: 1.3,
            pixelate: 0,
            vignette: 0.0,
            neon: false,
            greyscale: false,
            edge: false,
            paint: false,
            grain: 0.0,
        },
        "sketch" => StyleProfile {
            blur: 0.0,
            contrast: 1.9,
            saturation: 1.0,
            sharpen: 1.0,
            pixelate: 0,
            vignette: 0.0,
            neon: false,
            greyscale: true,
            edge: true,
            paint: false,
            grain: 0.0,
        },
        "oil_paint" => StyleProfile {
            blur: 1.1,
            contrast: 1.18,
            saturation: 1.3,
            sharpen: 1.0,
            pixelate: 0,
            vignette: 0.0,
            neon: false,
            greyscale: false,
            edge: false,
            paint: true,
            grain: 0.0,
        },
        "neon" => StyleProfile {
            blur: 0.35,
            contrast: 1.9,
            saturation: 2.4,
            sharpen: 1.0,
            pixelate: 0,
            vignette: 0.24,
            neon: true,
            greyscale: false,
            edge: false,
            paint: false,
            grain: 0.0,
        },
        "abstract" => StyleProfile {
            blur: 0.0,
            contrast: 1.45,
            saturation: 2.0,
            sharpen: 1.0,
            pixelate: 0,
            vignette: 0.0,
            neon: false,
            greyscale: false,
            edge: false,
            paint: false,
            grain: 0.05,
        },
        _ => style_profile("photo"),
    }
}

fn postprocess(mut img: RgbaImage, cfg: StyleProfile, seed: u64) -> RgbaImage {
    if cfg.greyscale {
        for pixel in img.pixels_mut() {
            let [r, g, b, a] = pixel.0;
            let l = ((r as f32 * 0.299) + (g as f32 * 0.587) + (b as f32 * 0.114)).round() as u8;
            *pixel = Rgba([l, l, l, a]);
        }
    }

    if cfg.blur > 0.0 {
        img = blur(&img, cfg.blur);
    }
    if cfg.paint {
        img = blur(&img, 1.1);
    }

    if cfg.edge {
        let mut edged = img.clone();
        apply_edge_mix(&mut edged);
        img = edged;
    }

    if (cfg.contrast - 1.0).abs() > f32::EPSILON {
        img = DynamicImage::ImageRgba8(img)
            .adjust_contrast((cfg.contrast - 1.0) * 100.0)
            .to_rgba8();
    }

    if (cfg.saturation - 1.0).abs() > f32::EPSILON {
        apply_saturation(&mut img, cfg.saturation);
    }

    if cfg.sharpen > 1.0 {
        img = DynamicImage::ImageRgba8(img).unsharpen(1.0, 1).to_rgba8();
    }

    if cfg.pixelate > 0 {
        let reduced_w = (img.width() / cfg.pixelate).max(1);
        let reduced_h = (img.height() / cfg.pixelate).max(1);
        let reduced = resize(&img, reduced_w, reduced_h, FilterType::Nearest);
        img = resize(&reduced, img.width(), img.height(), FilterType::Nearest);
    }

    if cfg.neon {
        let bloom = blur(&img, 6.0);
        alpha_blend_images(&mut img, &bloom, 0.45);
    }

    if cfg.vignette > 0.0 {
        apply_vignette(&mut img, cfg.vignette);
    }

    if cfg.grain > 0.0 {
        apply_grain(&mut img, cfg.grain, seed);
    }

    img
}

fn apply_edge_mix(img: &mut RgbaImage) {
    let width = img.width();
    let height = img.height();
    if width < 3 || height < 3 {
        return;
    }

    let source = img.clone();
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let l = luminance(source.get_pixel(x - 1, y).0);
            let r = luminance(source.get_pixel(x + 1, y).0);
            let u = luminance(source.get_pixel(x, y - 1).0);
            let d = luminance(source.get_pixel(x, y + 1).0);
            let edge = ((r - l).abs() + (d - u).abs()).clamp(0.0, 255.0);
            let boost = (edge * 0.6).clamp(0.0, 255.0);
            let px = source.get_pixel(x, y).0;
            img.put_pixel(
                x,
                y,
                Rgba([
                    add_light(px[0], boost),
                    add_light(px[1], boost),
                    add_light(px[2], boost),
                    px[3],
                ]),
            );
        }
    }
}

fn apply_saturation(img: &mut RgbaImage, saturation: f32) {
    for pixel in img.pixels_mut() {
        let [r, g, b, a] = pixel.0;
        let grey = (r as f32 + g as f32 + b as f32) / 3.0;
        let nr = (grey + (r as f32 - grey) * saturation).clamp(0.0, 255.0) as u8;
        let ng = (grey + (g as f32 - grey) * saturation).clamp(0.0, 255.0) as u8;
        let nb = (grey + (b as f32 - grey) * saturation).clamp(0.0, 255.0) as u8;
        *pixel = Rgba([nr, ng, nb, a]);
    }
}

fn apply_vignette(img: &mut RgbaImage, strength: f32) {
    let width = img.width();
    let height = img.height();
    for y in 0..height {
        let ny = (y as f32 / height as f32) * 2.0 - 1.0;
        for x in 0..width {
            let nx = (x as f32 / width as f32) * 2.0 - 1.0;
            let dist = (nx * nx + ny * ny).sqrt();
            let mask = (1.0 - ((dist - 0.2).max(0.0) * strength * 1.8)).clamp(0.55, 1.0);
            let mut px = img.get_pixel(x, y).0;
            px[0] = (px[0] as f32 * mask).round().clamp(0.0, 255.0) as u8;
            px[1] = (px[1] as f32 * mask).round().clamp(0.0, 255.0) as u8;
            px[2] = (px[2] as f32 * mask).round().clamp(0.0, 255.0) as u8;
            img.put_pixel(x, y, Rgba(px));
        }
    }
}

fn apply_grain(img: &mut RgbaImage, strength: f32, seed: u64) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
    let sigma = 255.0 * strength;
    for pixel in img.pixels_mut() {
        let [r, g, b, a] = pixel.0;
        let nr = (r as f32 + rng.gen_range(-sigma..=sigma)).clamp(0.0, 255.0) as u8;
        let ng = (g as f32 + rng.gen_range(-sigma..=sigma)).clamp(0.0, 255.0) as u8;
        let nb = (b as f32 + rng.gen_range(-sigma..=sigma)).clamp(0.0, 255.0) as u8;
        *pixel = Rgba([nr, ng, nb, a]);
    }
}

fn normalize_style(style: &str) -> &str {
    match style {
        "pixel" | "photo" | "anime" | "sketch" | "oil_paint" | "neon" | "abstract" => style,
        _ => "photo",
    }
}

fn normalize_quality(quality: &str) -> &str {
    match quality {
        "draft" | "balanced" | "ultra" => quality,
        _ => "balanced",
    }
}

fn render_scale_for(style: &str, quality: &str) -> u32 {
    let cinematic = matches!(style, "photo" | "oil_paint" | "neon" | "anime");
    match (quality, cinematic) {
        ("ultra", true) => 2,
        _ => 1,
    }
}

fn apply_scene_hint_tokens(tokens: &mut BTreeSet<String>, scene_hint: &str) {
    match scene_hint {
        "dog" => {
            tokens.insert("dog".into());
            tokens.insert("canine".into());
        }
        "dog_portrait" => {
            for token in ["dog", "canine", "portrait", "headshot"] {
                tokens.insert(token.into());
            }
        }
        "cat" => {
            tokens.insert("cat".into());
            tokens.insert("feline".into());
        }
        "tree" => {
            tokens.insert("tree".into());
            tokens.insert("forest".into());
        }
        "cityscape" => {
            for token in ["city", "skyline", "urban"] {
                tokens.insert(token.into());
            }
        }
        "sportscar" => {
            for token in ["sportscar", "car", "vehicle", "road", "coupe", "racing"] {
                tokens.insert(token.into());
            }
        }
        "sailboat" => {
            for token in ["boat", "ship", "sailboat", "ocean", "water"] {
                tokens.insert(token.into());
            }
        }
        "castle" => {
            for token in ["castle", "fortress", "tower"] {
                tokens.insert(token.into());
            }
        }
        "mountain_landscape" => {
            for token in ["mountain", "mountains", "peak", "ridge"] {
                tokens.insert(token.into());
            }
        }
        "flowers" => {
            for token in ["flower", "flowers", "garden", "blossom"] {
                tokens.insert(token.into());
            }
        }
        _ => {}
    }
}

fn is_portrait(tokens: &BTreeSet<String>, scene_hint: &str) -> bool {
    has_any(tokens, &["portrait", "closeup", "headshot"]) || scene_hint == "dog_portrait"
}

fn prompt_seed(prompt: &str) -> u32 {
    let digest = Sha256::digest(prompt.as_bytes());
    u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]])
}

fn slugify(text: &str) -> String {
    let mut out = String::new();
    let mut last_was_sep = true;
    for ch in text.chars().flat_map(|c| c.to_lowercase()) {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }
    out.trim_matches('_').chars().take(40).collect()
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn normalize_channel(value: f32, lo: f32, hi: f32) -> u8 {
    if hi > lo {
        (((value - lo) / (hi - lo)) * 255.0).clamp(0.0, 255.0) as u8
    } else {
        128
    }
}

fn transparent_layer_like(img: &RgbaImage) -> RgbaImage {
    RgbaImage::from_pixel(img.width(), img.height(), Rgba([0, 0, 0, 0]))
}

fn alpha_composite(base: &mut RgbaImage, overlay: &RgbaImage) {
    for (x, y, overlay_px) in overlay.enumerate_pixels() {
        let mut dst = base.get_pixel(x, y).0;
        blend_pixel_over(&mut dst, overlay_px.0);
        base.put_pixel(x, y, Rgba(dst));
    }
}

fn alpha_blend_images(base: &mut RgbaImage, overlay: &RgbaImage, amount: f32) {
    for (x, y, overlay_px) in overlay.enumerate_pixels() {
        let base_px = base.get_pixel(x, y).0;
        let src = overlay_px.0;
        base.put_pixel(
            x,
            y,
            Rgba([
                blend_channel(base_px[0], src[0], amount),
                blend_channel(base_px[1], src[1], amount),
                blend_channel(base_px[2], src[2], amount),
                255,
            ]),
        );
    }
}

fn blend_pixel_over(dst: &mut [u8; 4], src: [u8; 4]) {
    let alpha = src[3] as f32 / 255.0;
    if alpha <= 0.0 {
        return;
    }
    let inv = 1.0 - alpha;
    dst[0] = ((src[0] as f32 * alpha) + (dst[0] as f32 * inv)).round().clamp(0.0, 255.0) as u8;
    dst[1] = ((src[1] as f32 * alpha) + (dst[1] as f32 * inv)).round().clamp(0.0, 255.0) as u8;
    dst[2] = ((src[2] as f32 * alpha) + (dst[2] as f32 * inv)).round().clamp(0.0, 255.0) as u8;
    dst[3] = 255;
}

fn set_pixel_blend(img: &mut RgbaImage, x: i32, y: i32, color: RgbaColor) {
    if x < 0 || y < 0 || x >= img.width() as i32 || y >= img.height() as i32 {
        return;
    }
    let mut dst = img.get_pixel(x as u32, y as u32).0;
    blend_pixel_over(&mut dst, color);
    img.put_pixel(x as u32, y as u32, Rgba(dst));
}

fn draw_rect(img: &mut RgbaImage, x0: i32, y0: i32, x1: i32, y1: i32, color: RgbaColor) {
    let left = x0.min(x1).max(0);
    let right = x0.max(x1).min(img.width() as i32);
    let top = y0.min(y1).max(0);
    let bottom = y0.max(y1).min(img.height() as i32);
    for y in top..bottom {
        for x in left..right {
            set_pixel_blend(img, x, y, color);
        }
    }
}

fn draw_line(img: &mut RgbaImage, x0: i32, y0: i32, x1: i32, y1: i32, color: RgbaColor, thickness: i32) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let steps = dx.abs().max(dy.abs()).max(1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = x0 as f32 + dx as f32 * t;
        let y = y0 as f32 + dy as f32 * t;
        draw_circle(img, x.round() as i32, y.round() as i32, (thickness / 2).max(0), color);
    }
}

fn draw_polyline(img: &mut RgbaImage, points: &[(i32, i32)], color: RgbaColor, thickness: i32) {
    for segment in points.windows(2) {
        let a = segment[0];
        let b = segment[1];
        draw_line(img, a.0, a.1, b.0, b.1, color, thickness);
    }
}

fn draw_circle(img: &mut RgbaImage, cx: i32, cy: i32, radius: i32, color: RgbaColor) {
    let r = radius.max(0);
    let rr = r * r;
    for y in (cy - r)..=(cy + r) {
        for x in (cx - r)..=(cx + r) {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= rr {
                set_pixel_blend(img, x, y, color);
            }
        }
    }
}

fn draw_ellipse(img: &mut RgbaImage, cx: i32, cy: i32, rx: i32, ry: i32, color: RgbaColor) {
    let rx = rx.max(1);
    let ry = ry.max(1);
    let rx2 = (rx * rx) as f32;
    let ry2 = (ry * ry) as f32;
    for y in (cy - ry)..=(cy + ry) {
        for x in (cx - rx)..=(cx + rx) {
            let dx = x - cx;
            let dy = y - cy;
            let v = (dx * dx) as f32 / rx2 + (dy * dy) as f32 / ry2;
            if v <= 1.0 {
                set_pixel_blend(img, x, y, color);
            }
        }
    }
}

fn draw_polygon(img: &mut RgbaImage, points: &[(i32, i32)], color: RgbaColor) {
    if points.len() < 3 {
        return;
    }

    let min_x = points.iter().map(|(x, _)| *x).min().unwrap_or(0).max(0);
    let max_x = points.iter().map(|(x, _)| *x).max().unwrap_or(0).min(img.width() as i32 - 1);
    let min_y = points.iter().map(|(_, y)| *y).min().unwrap_or(0).max(0);
    let max_y = points.iter().map(|(_, y)| *y).max().unwrap_or(0).min(img.height() as i32 - 1);

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if point_in_polygon(x as f32 + 0.5, y as f32 + 0.5, points) {
                set_pixel_blend(img, x, y, color);
            }
        }
    }
}

fn point_in_polygon(x: f32, y: f32, points: &[(i32, i32)]) -> bool {
    let mut inside = false;
    let mut j = points.len() - 1;
    for i in 0..points.len() {
        let xi = points[i].0 as f32;
        let yi = points[i].1 as f32;
        let xj = points[j].0 as f32;
        let yj = points[j].1 as f32;
        let intersects = ((yi > y) != (yj > y))
            && (x < (xj - xi) * (y - yi) / ((yj - yi).abs().max(f32::EPSILON)) + xi);
        if intersects {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn draw_arc_band(
    img: &mut RgbaImage,
    cx: i32,
    cy: i32,
    radius: i32,
    start_deg: f32,
    end_deg: f32,
    color: RgbaColor,
    thickness: i32,
) {
    let start = start_deg;
    let mut end = end_deg;
    if end < start {
        end += 360.0;
    }
    let mut angle = start;
    while angle <= end {
        let rad = angle.to_radians();
        let x = cx + (rad.cos() * radius as f32) as i32;
        let y = cy + (rad.sin() * radius as f32) as i32;
        draw_circle(img, x, y, (thickness / 2).max(1), color);
        angle += 1.5;
    }
}

fn luminance(px: [u8; 4]) -> f32 {
    px[0] as f32 * 0.299 + px[1] as f32 * 0.587 + px[2] as f32 * 0.114
}

fn blend_channel(a: u8, b: u8, amount: f32) -> u8 {
    (a as f32 * (1.0 - amount) + b as f32 * amount).round().clamp(0.0, 255.0) as u8
}

fn add_light(channel: u8, amount: f32) -> u8 {
    (channel as f32 + amount).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::{generate_image, palette_from_hint, plan_scene, ResonartRenderRequest};

    #[test]
    fn sportscar_prompts_map_to_racing_scene_and_palette() {
        let plan = plan_scene(
            "a neon red sportscar drifting through a rain-soaked city street at night",
            "blurry, low resolution",
        );

        assert_eq!(plan.scene_hint, "sportscar");
        assert_eq!(plan.palette_hint, "racing");
        assert!(plan.tokens.iter().any(|token| token == "sportscar"));
    }

    #[test]
    fn blocked_sportscar_terms_fall_back_to_environment() {
        let plan = plan_scene(
            "a sleek car under neon lights",
            "car, vehicle, sportscar, supercar",
        );

        assert_eq!(plan.scene_hint, "environment");
    }

    #[test]
    fn racing_palette_has_expected_anchor_color_count() {
        let palette = palette_from_hint("racing");
        assert_eq!(palette.len(), 5);
        assert_eq!(palette[0], [110, 12, 12]);
    }

    #[test]
    fn native_renderer_smoke_succeeds() {
        let out_dir = std::env::temp_dir().join("quantoniumos_resonart_test");
        let req = ResonartRenderRequest {
            prompt: "neon red sportscar drifting through a city street".to_string(),
            negative_prompt: "blurry watermark".to_string(),
            steps: 12,
            style: "neon".to_string(),
            width: 256,
            height: 256,
            quality: "balanced".to_string(),
            scene_hint: "sportscar".to_string(),
            palette_hint: "racing".to_string(),
            spectral_complexity: 1.2,
            seed: Some(1234),
            output_dir: out_dir.to_string_lossy().to_string(),
        };

        let result = generate_image(req).expect("render should succeed");
        assert!(std::path::Path::new(&result.path).exists());
    }

    #[test]
    fn native_renderer_balanced_photo_1024_dog_succeeds() {
        let out_dir = std::env::temp_dir().join("quantoniumos_resonart_test");
        let req = ResonartRenderRequest {
            prompt: "a dog".to_string(),
            negative_prompt: "blurry, low resolution, watermark".to_string(),
            steps: 20,
            style: "photo".to_string(),
            width: 1024,
            height: 1024,
            quality: "balanced".to_string(),
            scene_hint: "dog".to_string(),
            palette_hint: "dog".to_string(),
            spectral_complexity: 0.1,
            seed: Some(1234),
            output_dir: out_dir.to_string_lossy().to_string(),
        };

        let result = generate_image(req).expect("1024 balanced photo render should succeed");
        assert!(std::path::Path::new(&result.path).exists());
    }
}
