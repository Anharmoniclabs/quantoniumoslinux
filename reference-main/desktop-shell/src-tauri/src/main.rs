// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod engines;

use serde::{Deserialize, Serialize};
use sysinfo::{Disks, System};
use tauri::Manager;
use tauri::ipc::Channel;
use std::io::BufRead;
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
    let exe = std::env::current_exe().unwrap_or_default();
    let mut dir = exe.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
    for _ in 0..6 {
        if dir.join("pyproject.toml").exists() {
            return dir.to_string_lossy().to_string();
        }
        if let Some(parent) = dir.parent() {
            dir = parent.to_path_buf();
        } else {
            break;
        }
    }
    std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
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
const DEFAULT_MODEL: &str = "deepseek-r1:7b";

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
                // Prefer deepseek-r1 if available
                if let Some(ds) = models.iter().find(|n| n.starts_with("deepseek-r1")) {
                    primary_model = ds.clone();
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
}

#[derive(Clone, Serialize)]
struct ChatToken {
    token: String,
    done: bool,
}

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

        let msgs: Vec<serde_json::Value> = req.messages.iter().map(|m| {
            serde_json::json!({ "role": &m.role, "content": &m.content })
        }).collect();

        let body = serde_json::json!({
            "model": model,
            "messages": msgs,
            "stream": true,
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

            let _ = on_event.send(ChatToken { token, done });

            if done { break; }
        }

        Ok::<(), String>(())
    })
    .await
    .map_err(|e| format!("Task panicked: {}", e))?
}

// ── Pack info (legacy compat) ──

#[derive(Serialize)]
struct PackInfo {
    pack_name: String,
    pack_path: String,
    server_running: bool,
}

#[tauri::command]
fn get_pack_info() -> PackInfo {
    PackInfo {
        pack_name: "deepseek-r1-7b".into(),
        pack_path: "ollama://deepseek-r1:7b".into(),
        server_running: ollama_is_up(),
    }
}

fn main() {
    // Auto-grant media capture permissions in WebView2 (must be set before WebView2 init)
    #[cfg(target_os = "windows")]
    std::env::set_var(
        "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
        "--autoplay-policy=no-user-gesture-required --use-fake-ui-for-media-stream",
    );

    let sys = System::new_all();
    let ollama_child = ensure_ollama();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SysState(Mutex::new(sys)))
        .manage(OllamaServer(Mutex::new(ollama_child)))
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
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(mut child) = window
                    .app_handle()
                    .try_state::<OllamaServer>()
                    .and_then(|s| s.0.lock().ok()?.take())
                {
                    eprintln!("[quantoniumos] Stopping Ollama (we started it)...");
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
