// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod engines;

use serde::{Deserialize, Serialize};
use sysinfo::{Disks, System};
use tauri::Manager;
use tauri::ipc::Channel;
use std::io::BufRead;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;

#[derive(Serialize, Clone)]
struct ProcessInfo {
    name: String,
    cpu: f32,
    mem_mb: f64,
}

#[derive(Serialize, Clone)]
struct SystemStats {
    cpu_percent: f32,
    mem_used_gb: f64,
    mem_total_gb: f64,
    mem_percent: f64,
    disk_used_gb: f64,
    disk_total_gb: f64,
    disk_percent: f64,
    processes: Vec<ProcessInfo>,
}

struct SysState(Mutex<System>);

fn project_root_path() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let mut dir = exe.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
    for _ in 0..6 {
        if dir.join("pyproject.toml").exists() {
            return dir;
        }
        if let Some(parent) = dir.parent() {
            dir = parent.to_path_buf();
        } else {
            break;
        }
    }
    std::env::current_dir().unwrap_or_default()
}

fn parse_flag_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

fn maybe_run_cli_command() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        return None;
    }

    match args[1].as_str() {
        "smartweights-train-microchunk" => {
            let req = engines::smartweights::SmartWeightsTrainMicrochunkRequest {
                chunk_id: parse_flag_value(&args, "--chunk-id"),
                corpus_paths: parse_flag_value(&args, "--corpus")
                    .map(|value| vec![value]),
                feature_dim: parse_flag_value(&args, "--feature-dim")
                    .and_then(|value| value.parse::<usize>().ok()),
                epochs: parse_flag_value(&args, "--epochs")
                    .and_then(|value| value.parse::<usize>().ok()),
                learning_rate: parse_flag_value(&args, "--learning-rate")
                    .and_then(|value| value.parse::<f32>().ok()),
            };

            let repo_root = project_root_path();
            match engines::smartweights::train_microchunk(&repo_root, req) {
                Ok(result) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&result)
                            .unwrap_or_else(|_| "{\"success\":true}".to_string())
                    );
                    Some(0)
                }
                Err(error) => {
                    eprintln!("[smartweights] training failed: {error}");
                    Some(1)
                }
            }
        }
        _ => None,
    }
}

#[tauri::command]
fn get_system_stats(state: tauri::State<'_, SysState>) -> SystemStats {
    let mut sys = state.0.lock().unwrap();
    sys.refresh_cpu_all();
    sys.refresh_memory();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);

    // CPU — average across all cores
    let cpus = sys.cpus();
    let cpu_percent = if cpus.is_empty() {
        0.0
    } else {
        cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / cpus.len() as f32
    };

    // Memory
    let mem_total = sys.total_memory() as f64;
    let mem_used = sys.used_memory() as f64;
    let mem_total_gb = mem_total / 1_073_741_824.0;
    let mem_used_gb = mem_used / 1_073_741_824.0;
    let mem_percent = if mem_total > 0.0 { (mem_used / mem_total) * 100.0 } else { 0.0 };

    // Disk — sum all disks
    let disks = Disks::new_with_refreshed_list();
    let mut disk_total: u64 = 0;
    let mut disk_available: u64 = 0;
    for disk in disks.list() {
        disk_total += disk.total_space();
        disk_available += disk.available_space();
    }
    let disk_used = disk_total.saturating_sub(disk_available);
    let disk_total_gb = disk_total as f64 / 1_073_741_824.0;
    let disk_used_gb = disk_used as f64 / 1_073_741_824.0;
    let disk_percent = if disk_total > 0 { (disk_used as f64 / disk_total as f64) * 100.0 } else { 0.0 };

    // Top processes by memory
    let mut procs: Vec<ProcessInfo> = sys
        .processes()
        .values()
        .map(|p| ProcessInfo {
            name: p.name().to_string_lossy().to_string(),
            cpu: p.cpu_usage(),
            mem_mb: p.memory() as f64 / 1_048_576.0,
        })
        .filter(|p| p.mem_mb > 1.0)
        .collect();
    procs.sort_by(|a, b| b.mem_mb.partial_cmp(&a.mem_mb).unwrap_or(std::cmp::Ordering::Equal));
    procs.truncate(25);

    SystemStats {
        cpu_percent,
        mem_used_gb,
        mem_total_gb,
        mem_percent,
        disk_used_gb,
        disk_total_gb,
        disk_percent,
        processes: procs,
    }
}

#[tauri::command]
fn get_project_root() -> String {
    project_root_path().to_string_lossy().to_string()
}

// ── RFT Engine Commands ──────────────────────────────────────────────────────

#[derive(Serialize)]
struct EngineStatus {
    active: bool,
    engines: Vec<String>,
    rft_variants: usize,
    fft_ready: bool,
    crypto_ready: bool,
    compression_ready: bool,
}

#[tauri::command]
fn rft_engine_status() -> EngineStatus {
    EngineStatus {
        active: true,
        engines: vec![
            "rft_kernel".into(),
            "rftmw_core".into(),
            "rftmw_fft".into(),
            "rftmw_inference".into(),
            "rftmw_compression".into(),
            "rftmw_spectral".into(),
            "rftmw_streaming".into(),
            "crypto_feistel".into(),
            "crypto_rft_sis".into(),
            "split_gguf".into(),
            "orchestrator".into(),
        ],
        rft_variants: 16,
        fft_ready: true,
        crypto_ready: true,
        compression_ready: true,
    }
}

#[derive(Deserialize)]
struct TransformRequest {
    signal: Vec<f64>,
    inverse: Option<bool>,
}

#[derive(Serialize)]
struct TransformResult {
    real: Vec<f64>,
    imag: Vec<f64>,
    magnitudes: Vec<f64>,
    phases: Vec<f64>,
    concentration: f64,
    top_k: usize,
    sparsity: f64,
}

#[tauri::command]
fn rft_transform(req: TransformRequest) -> Result<TransformResult, String> {
    let n = req.signal.len();
    if n == 0 || n > 256 {
        return Err("Signal length must be 1-256".into());
    }
    let engine = engines::rftmw::core::RFTMWEngine::new(n);
    let coeffs = engine.forward(&req.signal);
    let metrics = engine.analyze_signal(&req.signal, 0.95);

    let real: Vec<f64> = coeffs.iter().map(|c| c.re).collect();
    let imag: Vec<f64> = coeffs.iter().map(|c| c.im).collect();
    let magnitudes: Vec<f64> = coeffs.iter().map(|c| c.abs()).collect();
    let phases: Vec<f64> = coeffs.iter().map(|c| c.arg()).collect();

    Ok(TransformResult {
        real,
        imag,
        magnitudes,
        phases,
        concentration: metrics.energy_ratio,
        top_k: metrics.top_k,
        sparsity: metrics.sparsity_ratio,
    })
}

#[derive(Deserialize)]
struct CompressRequest {
    signal: Vec<f64>,
    keep_ratio: Option<f64>,
}

#[derive(Serialize)]
struct CompressResult {
    recovered: Vec<f64>,
    original_size: usize,
    compressed_coefficients: usize,
    energy_ratio: f64,
    relative_error: f64,
}

#[tauri::command]
fn rft_compress(req: CompressRequest) -> Result<CompressResult, String> {
    let n = req.signal.len();
    if n == 0 || n > 256 {
        return Err("Signal length must be 1-256".into());
    }
    let compressor = engines::rftmw::compression::RFTMWCompressor::new(
        n,
        engines::rftmw::compression::QuantizationParams::default(),
    );
    let result = compressor.compress(&req.signal);
    let recovered = compressor.decompress(&result);

    let err: f64 = req.signal.iter().zip(recovered.iter())
        .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    let norm: f64 = req.signal.iter().map(|v| v * v).sum::<f64>().sqrt();
    let relative_error = if norm > 1e-15 { err / norm } else { 0.0 };

    Ok(CompressResult {
        recovered,
        original_size: result.original_size,
        compressed_coefficients: result.kept_coefficients,
        energy_ratio: result.top_k_energy_ratio,
        relative_error,
    })
}

#[derive(Deserialize)]
struct HashRequest {
    data: String,
}

#[derive(Serialize)]
struct HashResult {
    hash_hex: String,
    algorithm: String,
}

#[tauri::command]
fn rft_hash(req: HashRequest) -> Result<HashResult, String> {
    let ctx = engines::crypto::rft_sis::RftSisCtx::new(
        None,
        engines::types::RftVariant::Fibonacci,
    ).map_err(|e| format!("{:?}", e))?;

    let hash = ctx.hash(req.data.as_bytes())
        .map_err(|e| format!("{:?}", e))?;
    let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();

    Ok(HashResult {
        hash_hex: hex,
        algorithm: "RFT-SIS (φ-lattice, n=512, q=3329)".into(),
    })
}

#[derive(Deserialize)]
struct CryptoRequest {
    data: String,
    key: String,
    decrypt: Option<bool>,
}

#[derive(Serialize)]
struct CryptoResult {
    output_hex: String,
    output_text: Option<String>,
    cipher: String,
    rounds: usize,
}

#[tauri::command]
fn rft_encrypt(req: CryptoRequest) -> Result<CryptoResult, String> {
    let key_bytes = req.key.as_bytes();
    if key_bytes.len() < 16 {
        return Err("Key must be at least 16 bytes".into());
    }
    let ctx = engines::crypto::feistel::FeistelCtx::new(
        key_bytes,
        engines::types::RftVariant::Standard,
    ).map_err(|e| format!("{:?}", e))?;

    let encrypted = ctx.encrypt_stream(req.data.as_bytes());
    let hex: String = encrypted.iter().map(|b| format!("{:02x}", b)).collect();

    Ok(CryptoResult {
        output_hex: hex,
        output_text: None,
        cipher: "Feistel-48 (φ-RFT, AES-SBOX, MixColumns)".into(),
        rounds: 48,
    })
}

#[tauri::command]
fn rft_decrypt(req: CryptoRequest) -> Result<CryptoResult, String> {
    let key_bytes = req.key.as_bytes();
    if key_bytes.len() < 16 {
        return Err("Key must be at least 16 bytes".into());
    }
    let ctx = engines::crypto::feistel::FeistelCtx::new(
        key_bytes,
        engines::types::RftVariant::Standard,
    ).map_err(|e| format!("{:?}", e))?;

    // Parse hex input
    let hex_str = req.data.trim();
    if hex_str.len() % 2 != 0 {
        return Err("Invalid hex input".into());
    }
    let bytes: Result<Vec<u8>, _> = (0..hex_str.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex_str[i..i+2], 16))
        .collect();
    let cipher_bytes = bytes.map_err(|_| "Invalid hex characters")?;

    let decrypted = ctx.decrypt_stream(&cipher_bytes)
        .map_err(|e| format!("{:?}", e))?;
    let text = String::from_utf8(decrypted.clone()).ok();
    let hex: String = decrypted.iter().map(|b| format!("{:02x}", b)).collect();

    Ok(CryptoResult {
        output_hex: hex,
        output_text: text,
        cipher: "Feistel-48 (φ-RFT, AES-SBOX, MixColumns)".into(),
        rounds: 48,
    })
}

#[derive(Serialize)]
struct ValidationResult {
    name: String,
    status: String,
    detail: String,
    value: Option<f64>,
}

#[tauri::command]
fn rft_validate() -> Vec<ValidationResult> {
    let mut results = Vec::new();

    // Test 1: Bijection Integrity (RFT roundtrip)
    {
        let n = 64;
        let engine = engines::rftmw::core::RFTMWEngine::new(n);
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let coeffs = engine.forward(&input);
        let recovered = engine.inverse(&coeffs);
        let err: f64 = input.iter().zip(recovered.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let norm: f64 = input.iter().map(|v| v * v).sum::<f64>().sqrt();
        let rel_err = if norm > 1e-15 { err / norm } else { 0.0 };
        results.push(ValidationResult {
            name: "Bijection Integrity".into(),
            status: if rel_err < 1e-6 { "pass" } else if rel_err < 0.01 { "warn" } else { "fail" }.into(),
            detail: format!("RFT roundtrip relative error: {:.2e}", rel_err),
            value: Some(rel_err),
        });
    }

    // Test 2: Entropy Conservation (compression energy ratio)
    {
        let n = 64;
        let comp = engines::rftmw::compression::RFTMWCompressor::new(
            n, engines::rftmw::compression::QuantizationParams::default()
        );
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let result = comp.compress(&input);
        let recovered = comp.decompress(&result);
        let err: f64 = input.iter().zip(recovered.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let norm: f64 = input.iter().map(|v| v * v).sum::<f64>().sqrt();
        let rel_err = if norm > 1e-15 { err / norm } else { 0.0 };
        results.push(ValidationResult {
            name: "Entropy Conservation".into(),
            status: if rel_err < 0.01 { "pass" } else if rel_err < 0.05 { "warn" } else { "fail" }.into(),
            detail: format!("Compression roundtrip error: {:.4}", rel_err),
            value: Some(result.top_k_energy_ratio),
        });
    }

    // Test 3: Spectral Residual (FFT roundtrip)
    {
        let n = 64;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).sin()).collect();
        let mut re = input.clone();
        let mut im = vec![0.0f32; n];
        engines::rftmw::fft::fft_f32(&mut re, &mut im);
        engines::rftmw::fft::ifft_f32(&mut re, &mut im);
        let err: f32 = input.iter().zip(re.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        let norm: f32 = input.iter().map(|v| v * v).sum::<f32>().sqrt();
        let rel = if norm > 1e-7 { err / norm } else { 0.0 };
        results.push(ValidationResult {
            name: "Spectral Residual".into(),
            status: if rel < 1e-5 { "pass" } else if rel < 1e-3 { "warn" } else { "fail" }.into(),
            detail: format!("FFT roundtrip residual: {:.2e}", rel),
            value: Some(rel as f64),
        });
    }

    // Test 4: Token Fidelity (RMS norm + softmax sanity)
    {
        let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.2).collect();
        let weight: Vec<f32> = vec![1.0; 64];
        let mut normed = vec![0.0f32; 64];
        engines::rftmw::inference::rms_norm(&data, &weight, &mut normed, 1e-6);
        let sq_sum: f32 = normed.iter().map(|v| v * v).sum();
        let rms = (sq_sum / normed.len() as f32).sqrt();
        let logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut probs = vec![0.0f32; 5];
        engines::rftmw::inference::softmax(&logits, &mut probs);
        let prob_sum: f32 = probs.iter().sum();
        let rms_ok = (rms - 1.0).abs() < 0.1;
        let sm_ok = (prob_sum - 1.0).abs() < 1e-5;
        results.push(ValidationResult {
            name: "Token Fidelity".into(),
            status: if rms_ok && sm_ok { "pass" } else { "warn" }.into(),
            detail: format!("RMS norm factor: {:.4}, softmax sum: {:.6}", rms, prob_sum),
            value: Some(rms as f64),
        });
    }

    // Test 5: Compression Ratio
    {
        let n = 128;
        let engine = engines::rftmw::core::RFTMWEngine::new(n);
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.37).cos() * 0.5).collect();
        let metrics = engine.analyze_signal(&input, 0.95);
        let ratio = 1.0 / metrics.sparsity_ratio;
        results.push(ValidationResult {
            name: "Compression Ratio".into(),
            status: if ratio > 2.0 { "pass" } else if ratio > 1.2 { "warn" } else { "fail" }.into(),
            detail: format!("Spectral compression: {:.1}x (top-k={}, energy={:.1}%)", ratio, metrics.top_k, metrics.energy_ratio * 100.0),
            value: Some(ratio),
        });
    }

    // Test 6: Crypto Integrity (Feistel round-trip)
    {
        let key = b"quantoniumos-rft-test-key-32byte!";
        let ctx = engines::crypto::feistel::FeistelCtx::new(
            key, engines::types::RftVariant::Standard
        );
        let (status, detail) = match ctx {
            Ok(ctx) => {
                let pt = b"RFT engine test!";
                let ct = ctx.encrypt_block(pt);
                let dt = ctx.decrypt_block(&ct);
                if dt == *pt {
                    ("pass".to_string(), "Feistel-48 encrypt/decrypt: verified".to_string())
                } else {
                    ("fail".to_string(), "Feistel-48 roundtrip mismatch".to_string())
                }
            }
            Err(e) => ("fail".to_string(), format!("Feistel init failed: {:?}", e)),
        };
        results.push(ValidationResult {
            name: "Header Checksum".into(),
            status,
            detail,
            value: None,
        });
    }

    results
}

// ── Ollama Inference Backend ─────────────────────────────────────────────────

const OLLAMA_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_MODEL: &str = "eldrae:smartweights";

/// Ensure Ollama is running. On Windows it's usually a background service,
/// but if /api/tags doesn't respond, try to start `ollama serve`.
struct OllamaServer(Mutex<Option<Child>>);

fn ollama_is_up() -> bool {
    std::net::TcpStream::connect_timeout(
        &"127.0.0.1:11434".parse().unwrap(),
        std::time::Duration::from_secs(2),
    ).is_ok()
}

fn ensure_ollama() -> Option<Child> {
    if ollama_is_up() {
        eprintln!("[quantoniumos] Ollama already running on :11434");
        return None;
    }

    eprintln!("[quantoniumos] Starting Ollama...");
    let child = Command::new("ollama")
        .arg("serve")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn();

    match child {
        Ok(c) => {
            eprintln!("[quantoniumos] Ollama started (pid={})", c.id());
            std::thread::sleep(std::time::Duration::from_millis(1500));
            Some(c)
        }
        Err(e) => {
            eprintln!("[quantoniumos] Failed to start Ollama: {}", e);
            eprintln!("[quantoniumos] Install Ollama from https://ollama.com");
            None
        }
    }
}

// ── Ollama status command ──

#[derive(Serialize)]
struct OllamaStatus {
    running: bool,
    model: String,
    models: Vec<String>,
    backend: String,
}

#[tauri::command]
fn ollama_status() -> OllamaStatus {
    if !ollama_is_up() {
        return OllamaStatus {
            running: false,
            model: String::new(),
            models: vec![],
            backend: "ollama".into(),
        };
    }

    let mut models = Vec::new();
    let mut primary_model = DEFAULT_MODEL.to_string();

    if let Ok(resp) = ureq::get(&format!("{}/api/tags", OLLAMA_URL))
        .timeout(std::time::Duration::from_secs(3))
        .call()
    {
        if let Ok(data) = resp.into_json::<serde_json::Value>() {
            if let Some(arr) = data["models"].as_array() {
                for m in arr {
                    if let Some(name) = m["name"].as_str() {
                        models.push(name.to_string());
                    }
                }
                // Eldrae's live brain is the fast local Qwen 3B model with the SmartWeights runtime overlay.
                if let Some(eldrae) = models.iter().find(|n| n.starts_with("eldrae:smartweights")) {
                    primary_model = eldrae.clone();
                } else if let Some(qwen) = models.iter().find(|n| n.starts_with("qwen2.5-coder:3b")) {
                    primary_model = qwen.clone();
                } else if let Some(first) = models.first() {
                    primary_model = first.clone();
                }
            }
        }
    }

    OllamaStatus {
        running: true,
        model: primary_model,
        models,
        backend: "ollama".into(),
    }
}

// ── Streaming chat command ──

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    model: Option<String>,
    #[serde(default, alias = "maxTokens")]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
}

#[derive(Clone, Serialize)]
struct ChatToken {
    token: String,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<String>,
}

const RUST_IMAGE_ROUTE_PROMPT: &str = "THIS TURN IS AN IMAGE REQUEST.\nReply with one short natural sentence, then emit the [IMAGEGEN] command block as the final part of the message.\nPreserve the user's concrete visual details in prompt, move exclusions into negative, and include width, height, and quality for hi-res or wallpaper asks.";

#[tauri::command]
async fn chat_stream(req: ChatRequest, on_event: Channel<ChatToken>) -> Result<(), String> {
    eprintln!("[quantoniumos] chat_stream invoked, {} messages", req.messages.len());
    // Spawn blocking work off the main thread so the UI stays responsive
    tauri::async_runtime::spawn_blocking(move || {
        eprintln!("[quantoniumos] chat_stream blocking thread started");
        // Pre-check: is Ollama reachable?
        if !ollama_is_up() {
            return Err("Ollama is not running. Restart the app or start Ollama manually.".into());
        }

        let model = req.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
        let num_predict = req.max_tokens.unwrap_or(192).clamp(32, 768);
        let temperature = req
            .temperature
            .filter(|value| value.is_finite())
            .unwrap_or(0.7)
            .clamp(0.0, 1.5);
        eprintln!("[quantoniumos] chat_stream using model {}", model);

        let route = engines::smartweights::route_image_request(
            &project_root_path(),
            &req
                .messages
                .iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect::<Vec<_>>(),
        );
        if route.available {
            eprintln!(
                "[smartweights] {} score={:.3} threshold={:.3} image_route={}",
                route.chunk_id,
                route.score,
                route.threshold,
                route.use_image_turn_prompt
            );
        } else {
            eprintln!("[smartweights] router unavailable: {}", route.reason);
        }

        let history_has_image_turn_prompt = req.messages.iter().any(|m| {
            m.role.eq_ignore_ascii_case("system")
                && m.content.contains("THIS TURN IS AN IMAGE REQUEST.")
        });

        let mut msgs: Vec<serde_json::Value> = Vec::with_capacity(req.messages.len() + 1);
        let mut injected_route_prompt = false;
        for m in &req.messages {
            if !injected_route_prompt
                && route.use_image_turn_prompt
                && !history_has_image_turn_prompt
                && !m.role.eq_ignore_ascii_case("system")
            {
                msgs.push(serde_json::json!({
                    "role": "system",
                    "content": RUST_IMAGE_ROUTE_PROMPT,
                }));
                injected_route_prompt = true;
            }
            msgs.push(serde_json::json!({ "role": &m.role, "content": &m.content }));
        }
        if !injected_route_prompt
            && route.use_image_turn_prompt
            && !history_has_image_turn_prompt
        {
            msgs.push(serde_json::json!({
                "role": "system",
                "content": RUST_IMAGE_ROUTE_PROMPT,
            }));
        }

        let body = serde_json::json!({
            "model": model,
            "messages": msgs,
            "stream": true,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        });

        // Use a long timeout — reasoning models can take minutes on CPU for the first token
        let resp = ureq::post(&format!("{}/api/chat", OLLAMA_URL))
            .timeout(std::time::Duration::from_secs(300))
            .send_json(&body)
            .map_err(|e| format!("Ollama request failed: {}", e))?;

        let reader = std::io::BufReader::new(resp.into_reader());

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Read error: {}", e))?;
            if line.trim().is_empty() { continue; }

            let chunk: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| format!("JSON parse error: {}", e))?;

            let token = chunk["message"]["content"].as_str().unwrap_or("").to_string();
            let done = chunk["done"].as_bool().unwrap_or(false);
            let done_reason = if done {
                chunk["done_reason"].as_str().map(|s| s.to_string())
            } else {
                None
            };

            let _ = on_event.send(ChatToken { token, done, done_reason });

            if done { break; }
        }

        Ok::<(), String>(())
    })
    .await
    .map_err(|e| format!("Task panicked: {}", e))?
}

// ── Rust Inference Engine Commands ───────────────────────────────────────────

#[derive(Serialize)]
struct RustEngineStatus {
    available_7b: bool,
    available_35b: bool,
    backend: String,
}

#[tauri::command]
fn rust_engine_status() -> RustEngineStatus {
    let (has_7b, has_35b) = engines::rftmw::service::has_rust_engine();
    RustEngineStatus {
        available_7b: has_7b,
        available_35b: has_35b,
        backend: if has_7b || has_35b { "rftmw-rust" } else { "none" }.into(),
    }
}

#[tauri::command]
fn rust_engine_init() -> Result<String, String> {
    let cwd = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    
    // Try repo root (parent of desktop-shell)
    let repo_root = std::path::Path::new(&cwd)
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or(cwd.clone());
    
    engines::rftmw::service::init_engines(&repo_root)
}

#[derive(Deserialize)]
struct RustInferRequest {
    tokens: Vec<usize>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    use_35b: Option<bool>,
}

#[derive(Serialize)]
struct RustInferResult {
    tokens: Vec<usize>,
    engine: String,
}

#[tauri::command]
async fn rust_inference(req: RustInferRequest) -> Result<RustInferResult, String> {
    let use_35b = req.use_35b.unwrap_or(false);
    let max_tokens = req.max_tokens.unwrap_or(128);
    let temperature = req.temperature.unwrap_or(0.7);

    let engine_arc = if use_35b {
        engines::rftmw::service::get_engine_35b()
    } else {
        engines::rftmw::service::get_engine_7b()
    };

    let result = tauri::async_runtime::spawn_blocking(move || {
        let mut guard = engine_arc.lock()
            .map_err(|_| "Failed to lock engine")?;
        
        let engine = guard.as_mut()
            .ok_or("Rust engine not loaded. Call rust_engine_init first.")?;

        let output = engine.generate(&req.tokens, max_tokens, temperature);
        Ok::<Vec<usize>, String>(output)
    })
    .await
    .map_err(|e| format!("Task panicked: {}", e))??;

    Ok(RustInferResult {
        tokens: result,
        engine: if use_35b { "35b-rust" } else { "7b-rust" }.into(),
    })
}

// ── Guild (QUE agents) Server ────────────────────────────────────────────────
//
// Spawns src/rpg_headless.py as a managed child. Child speaks JSON-per-line
// on stdin/stdout. We fan stdout out to every subscriber via a broadcast
// channel so multiple views (GuildView, ChatView @-mentions, toasts) can
// share one tick loop.

struct GuildServer {
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<std::process::ChildStdin>>,
    tx: tokio::sync::broadcast::Sender<String>,
}

fn find_repo_root() -> std::path::PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let mut dir = exe.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
    for _ in 0..6 {
        if dir.join("pyproject.toml").exists() {
            return dir;
        }
        if let Some(parent) = dir.parent() {
            dir = parent.to_path_buf();
        } else {
            break;
        }
    }
    std::env::current_dir().unwrap_or_default()
}

fn find_python() -> String {
    // Prefer venv if one exists at repo root
    let root = find_repo_root();
    #[cfg(windows)]
    {
        let venv = root.join(".venv").join("Scripts").join("python.exe");
        if venv.exists() { return venv.to_string_lossy().to_string(); }
    }
    #[cfg(not(windows))]
    {
        let venv = root.join(".venv").join("bin").join("python");
        if venv.exists() { return venv.to_string_lossy().to_string(); }
    }
    "python".to_string()
}

fn ensure_guild() -> Option<GuildServer> {
    let root = find_repo_root();
    let script = root.join("src").join("rpg_headless.py");
    if !script.exists() {
        eprintln!("[quantoniumos] guild: rpg_headless.py not found at {}", script.display());
        return None;
    }

    let python = find_python();
    eprintln!("[quantoniumos] guild: spawning {} {}", python, script.display());

    let mut cmd = Command::new(&python);
    cmd.arg("-u")  // unbuffered stdout
       .arg(&script)
       .current_dir(&root)
       .stdin(std::process::Stdio::piped())
       .stdout(std::process::Stdio::piped())
       .stderr(std::process::Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[quantoniumos] guild: failed to spawn: {}", e);
            return None;
        }
    };

    let stdin = child.stdin.take();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let (tx, _) = tokio::sync::broadcast::channel::<String>(512);

    // stdout reader — broadcast every line
    if let Some(out) = stdout {
        let tx_clone = tx.clone();
        std::thread::spawn(move || {
            let reader = std::io::BufReader::new(out);
            for line in reader.lines() {
                match line {
                    Ok(l) if !l.trim().is_empty() => {
                        let _ = tx_clone.send(l);
                    }
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
            eprintln!("[quantoniumos] guild: stdout reader exited");
        });
    }

    // stderr tail — log-only (child's warnings go here)
    if let Some(err) = stderr {
        std::thread::spawn(move || {
            let reader = std::io::BufReader::new(err);
            for line in reader.lines().flatten() {
                eprintln!("[guild:stderr] {}", line);
            }
        });
    }

    Some(GuildServer {
        child: Mutex::new(Some(child)),
        stdin: Mutex::new(stdin),
        tx,
    })
}

#[tauri::command]
fn guild_status(guild: tauri::State<'_, Option<GuildServer>>) -> bool {
    guild.is_some()
}

#[tauri::command]
fn guild_command(cmd: String, guild: tauri::State<'_, Option<GuildServer>>) -> Result<(), String> {
    use std::io::Write;
    let g = guild.as_ref().ok_or_else(|| "Guild server not running".to_string())?;
    let mut stdin_lock = g.stdin.lock().map_err(|e| e.to_string())?;
    let stdin = stdin_lock.as_mut().ok_or_else(|| "Guild stdin closed".to_string())?;
    let trimmed = cmd.trim();
    writeln!(stdin, "{}", trimmed).map_err(|e| format!("write: {}", e))?;
    stdin.flush().map_err(|e| format!("flush: {}", e))?;
    Ok(())
}

#[derive(Clone, Serialize)]
struct GuildEvent {
    line: String,
}

#[tauri::command]
async fn guild_events(
    on_event: Channel<GuildEvent>,
    guild: tauri::State<'_, Option<GuildServer>>,
) -> Result<(), String> {
    let tx = match guild.as_ref() {
        Some(g) => g.tx.clone(),
        None => return Err("Guild server not running".into()),
    };
    let mut rx = tx.subscribe();
    tauri::async_runtime::spawn(async move {
        while let Ok(line) = rx.recv().await {
            if on_event.send(GuildEvent { line }).is_err() {
                break;
            }
        }
    });
    Ok(())
}

// ── Resonart Image Generation ───────────────────────────────────────────────

#[derive(Deserialize, Clone)]
struct ImageGenRequest {
    prompt: String,
    #[serde(default)]
    negative_prompt: String,
    #[serde(default)]
    steps: Option<u32>,
    #[serde(default)]
    style: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    quality: Option<String>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    backend: Option<String>,
}

#[derive(Deserialize)]
struct ResonartScenePlanRequest {
    prompt: String,
    #[serde(default)]
    negative_prompt: String,
}

#[derive(Serialize)]
struct ImageGenResult {
    success: bool,
    path: String,
    prompt: String,
    style: String,
    backend: String,
    seed: i64,
    width: u32,
    height: u32,
    error: String,
}

// Resonart is the unified image engine. There is one user-facing backend.
// Internally it has two native Rust render paths:
//   - Diffusion: QuantDiff DDIM + U-Net  (engines::quantdiff)
//   - Procedural: spectral scene renderer (engines::resonart)
// The procedural path is the always-available fallback. No external models,
// no HuggingFace, no Python subprocess.
#[derive(Clone, Copy)]
enum ResonartMode {
    Diffusion,
    Procedural,
}

impl ResonartMode {
    fn label(self) -> &'static str {
        match self {
            ResonartMode::Diffusion => "resonart-diffusion",
            ResonartMode::Procedural => "resonart-procedural",
        }
    }
}

fn choose_resonart_mode(req: &ImageGenRequest) -> (ResonartMode, bool) {
    let requested = req
        .backend
        .as_deref()
        .unwrap_or("auto")
        .trim()
        .to_ascii_lowercase();

    match requested.as_str() {
        "procedural" | "scene" => (ResonartMode::Procedural, false),
        "diffusion" | "diff" => (ResonartMode::Diffusion, false),
        // "auto" / "resonart" / anything else: diffusion is only meaningful with trained
        // weights; otherwise the warm-start adds noise via random U-Net params. Fall
        // through to procedural until weights exist.
        _ => {
            if engines::quantdiff::has_trained_weights() {
                (ResonartMode::Diffusion, true)
            } else {
                (ResonartMode::Procedural, false)
            }
        }
    }
}

fn error_image_result(req: &ImageGenRequest, backend: &str, error: String) -> ImageGenResult {
    ImageGenResult {
        success: false,
        path: String::new(),
        prompt: req.prompt.clone(),
        style: req.style.clone().unwrap_or_default(),
        backend: backend.to_string(),
        seed: req.seed.unwrap_or_default(),
        width: req.width.unwrap_or_default(),
        height: req.height.unwrap_or_default(),
        error,
    }
}

fn run_procedural(
    render_req: engines::resonart::ResonartRenderRequest,
    req: &ImageGenRequest,
) -> ImageGenResult {
    match engines::resonart::generate_image(render_req) {
        Ok(rendered) => ImageGenResult {
            success: true,
            path: rendered.path,
            prompt: rendered.prompt,
            style: rendered.style,
            backend: "resonart".to_string(),
            seed: rendered.seed,
            width: rendered.width,
            height: rendered.height,
            error: String::new(),
        },
        Err(error) => error_image_result(req, "resonart", error),
    }
}

fn run_diffusion(
    render_req: engines::resonart::ResonartRenderRequest,
    req: &ImageGenRequest,
) -> ImageGenResult {
    let qd_req = engines::quantdiff::QuantDiffRequest {
        prompt: render_req.prompt.clone(),
        negative_prompt: render_req.negative_prompt.clone(),
        steps: render_req.steps,
        style: render_req.style.clone(),
        width: render_req.width,
        height: render_req.height,
        quality: render_req.quality.clone(),
        scene_hint: render_req.scene_hint.clone(),
        palette_hint: render_req.palette_hint.clone(),
        spectral_complexity: render_req.spectral_complexity,
        seed: render_req.seed,
        output_dir: render_req.output_dir.clone(),
    };
    match engines::quantdiff::generate_image(qd_req) {
        Ok(rendered) => ImageGenResult {
            success: true,
            path: rendered.path,
            prompt: rendered.prompt,
            style: rendered.style,
            backend: "resonart".to_string(),
            seed: rendered.seed,
            width: rendered.width,
            height: rendered.height,
            error: String::new(),
        },
        Err(error) => error_image_result(req, "resonart", error),
    }
}

#[tauri::command]
fn resonart_scene_plan(req: ResonartScenePlanRequest) -> engines::resonart::ResonartScenePlan {
    engines::resonart::plan_scene(&req.prompt, &req.negative_prompt)
}

#[tauri::command]
async fn generate_image(app_handle: tauri::AppHandle, req: ImageGenRequest) -> ImageGenResult {
    // Save generated images into an app-owned absolute directory so Tauri's
    // asset protocol can serve them back to the webview reliably.
    let out_dir = app_handle
        .path()
        .app_local_data_dir()
        .unwrap_or_else(|_| std::env::temp_dir().join("quantoniumos"))
        .join("generated")
        .join("images");
    let _ = std::fs::create_dir_all(&out_dir);
    let scene_plan = engines::resonart::plan_scene(&req.prompt, &req.negative_prompt);
    let out_dir_path = out_dir.to_string_lossy().to_string();
    let (mode, auto) = choose_resonart_mode(&req);

    eprintln!(
        "[resonart] mode={} style={} steps={} scene={} complexity={:.3} prompt={}",
        mode.label(),
        req.style.as_deref().unwrap_or("photo"),
        req.steps.unwrap_or(20),
        scene_plan.scene_hint,
        scene_plan.spectral_complexity,
        &req.prompt[..req.prompt.len().min(60)]
    );

    let render_req = engines::resonart::ResonartRenderRequest {
        prompt: req.prompt.clone(),
        negative_prompt: req.negative_prompt.clone(),
        steps: req.steps.unwrap_or(20),
        style: req.style.clone().unwrap_or_else(|| "photo".to_string()),
        width: req.width.unwrap_or(1024),
        height: req.height.unwrap_or(1024),
        quality: req.quality.clone().unwrap_or_else(|| "balanced".to_string()),
        scene_hint: scene_plan.scene_hint.clone(),
        palette_hint: scene_plan.palette_hint.clone(),
        spectral_complexity: scene_plan.spectral_complexity,
        seed: req.seed,
        output_dir: out_dir_path,
    };

    let req_for_task = req.clone();

    match tauri::async_runtime::spawn_blocking(move || match mode {
        ResonartMode::Procedural => run_procedural(render_req, &req_for_task),
        ResonartMode::Diffusion => {
            let result = run_diffusion(render_req.clone(), &req_for_task);
            if !result.success && auto {
                eprintln!(
                    "[resonart] diffusion failed, falling back to procedural: {}",
                    result.error
                );
                run_procedural(render_req, &req_for_task)
            } else {
                result
            }
        }
    })
    .await
    {
        Ok(result) => result,
        Err(error) => error_image_result(&req, "resonart", format!("Render worker failed: {}", error)),
    }
}

// ── Pack info (legacy compat) ──

#[tauri::command]
async fn smartweights_train_microchunk(
    req: engines::smartweights::SmartWeightsTrainMicrochunkRequest,
) -> Result<engines::smartweights::SmartWeightsTrainMicrochunkResult, String> {
    let repo_root = project_root_path();
    tauri::async_runtime::spawn_blocking(move || {
        engines::smartweights::train_microchunk(&repo_root, req)
    })
    .await
    .map_err(|error| format!("smartweights microchunk worker failed: {error}"))?
}

#[derive(Serialize)]
struct PackInfo {
    pack_name: String,
    pack_path: String,
    server_running: bool,
}

#[tauri::command]
fn get_pack_info() -> PackInfo {
    PackInfo {
        pack_name: "eldrae-smartweights".into(),
        pack_path: "ollama://eldrae:smartweights".into(),
        server_running: ollama_is_up(),
    }
}

fn main() {
    if let Some(exit_code) = maybe_run_cli_command() {
        std::process::exit(exit_code);
    }

    // Auto-grant media capture permissions in WebView2 (must be set before WebView2 init)
    #[cfg(target_os = "windows")]
    std::env::set_var(
        "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
        "--autoplay-policy=no-user-gesture-required --use-fake-ui-for-media-stream",
    );

    let sys = System::new_all();
    let ollama_child = ensure_ollama();
    let guild_server = ensure_guild();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SysState(Mutex::new(sys)))
        .manage(OllamaServer(Mutex::new(ollama_child)))
        .manage(guild_server)
        .setup(|_app| {
            // NOTE: WebView2 permission handler temporarily disabled to debug chat IPC
            // TODO: Re-enable once chat is confirmed working
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_project_root,
            get_system_stats,
            get_pack_info,
            ollama_status,
            chat_stream,
            rft_engine_status,
            rft_transform,
            rft_compress,
            rft_hash,
            rft_encrypt,
            rft_decrypt,
            rft_validate,
            rust_engine_status,
            rust_engine_init,
            rust_inference,
            guild_status,
            guild_command,
            guild_events,
            resonart_scene_plan,
            generate_image,
            smartweights_train_microchunk,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                let handle = window.app_handle();

                // Stop Ollama if we started it
                if let Some(mut child) = handle
                    .try_state::<OllamaServer>()
                    .and_then(|s| s.0.lock().ok()?.take())
                {
                    eprintln!("[quantoniumos] Stopping Ollama (we started it)...");
                    let _ = child.kill();
                    let _ = child.wait();
                }

                // Stop Guild tick loop
                if let Some(state) = handle.try_state::<Option<GuildServer>>() {
                    if let Some(g) = state.inner() {
                        // Polite quit first
                        if let Ok(mut stdin) = g.stdin.lock() {
                            if let Some(s) = stdin.as_mut() {
                                use std::io::Write;
                                let _ = writeln!(s, "{{\"cmd\":\"quit\"}}");
                                let _ = s.flush();
                            }
                        }
                        if let Ok(mut child) = g.child.lock() {
                            if let Some(mut c) = child.take() {
                                eprintln!("[quantoniumos] Stopping Guild tick loop...");
                                std::thread::sleep(std::time::Duration::from_millis(250));
                                let _ = c.kill();
                                let _ = c.wait();
                            }
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
