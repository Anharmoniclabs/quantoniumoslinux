// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// service.rs — Unified inference service for Guild agents
//
// Provides a high-level API that:
// - Loads RFTMWPK2 model files
// - Runs transformer inference with quantized weights
// - Falls back to Ollama if Rust engine unavailable

use std::sync::{Arc, Mutex, OnceLock};
use std::path::PathBuf;
use crate::engines::rftmw::mmap::{MmapModel, QuantType};
use crate::engines::rftmw::inference::*;

// ── Model Configuration ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
}

impl From<&MmapModel> for ModelConfig {
    fn from(model: &MmapModel) -> Self {
        Self {
            hidden_size: model.header.hidden_size as usize,
            num_layers: model.header.num_layers as usize,
            num_heads: model.header.num_heads as usize,
            num_kv_heads: model.header.num_kv_heads as usize,
            head_dim: model.header.head_dim as usize,
            ffn_dim: model.header.ffn_dim as usize,
            vocab_size: model.header.vocab_size as usize,
            rope_theta: model.header.rope_theta,
        }
    }
}

// ── KV Cache ────────────────────────────────────────────────────────────────

pub struct KVCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub max_seq: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl KVCache {
    pub fn new(max_seq: usize, num_kv_heads: usize, head_dim: usize, num_layers: usize) -> Vec<Self> {
        (0..num_layers)
            .map(|_| {
                let cap = max_seq * num_kv_heads * head_dim;
                Self {
                    k: vec![0.0; cap],
                    v: vec![0.0; cap],
                    max_seq,
                    num_kv_heads,
                    head_dim,
                }
            })
            .collect()
    }

    pub fn write(&mut self, pos: usize, k: &[f32], v: &[f32]) {
        let offset = pos * self.num_kv_heads * self.head_dim;
        let len = self.num_kv_heads * self.head_dim;
        if offset + len <= self.k.len() {
            self.k[offset..offset + len].copy_from_slice(&k[..len]);
            self.v[offset..offset + len].copy_from_slice(&v[..len]);
        }
    }
}

// ── Inference Engine ────────────────────────────────────────────────────────

pub struct RustInferenceEngine {
    model: MmapModel,
    config: ModelConfig,
    kv_cache: Vec<KVCache>,
    embed_tokens: Vec<f32>,
    final_norm: Vec<f32>,
    lm_head: Vec<f32>,
}

impl RustInferenceEngine {
    /// Load model from RFTMWPK2 file
    pub fn load(path: &str) -> Result<Self, String> {
        let model = MmapModel::load(path)
            .map_err(|e| format!("Failed to load model: {:?}", e))?;
        
        let config = ModelConfig::from(&model);
        let kv_cache = KVCache::new(2048, config.num_kv_heads, config.head_dim, config.num_layers);

        // Extract embeddings
        let embed_tokens = Self::extract_f32_tensor(&model, "token_embd.weight")?;
        let final_norm = Self::extract_f32_tensor(&model, "output_norm.weight")
            .unwrap_or_else(|_| vec![1.0; config.hidden_size]);
        let lm_head = Self::extract_f32_tensor(&model, "output.weight")
            .unwrap_or_else(|_| vec![0.0; config.vocab_size * config.hidden_size]);

        Ok(Self {
            model,
            config,
            kv_cache,
            embed_tokens,
            final_norm,
            lm_head,
        })
    }

    fn extract_f32_tensor(model: &MmapModel, name: &str) -> Result<Vec<f32>, String> {
        let tensor = model.find_tensor(name)
            .ok_or_else(|| format!("Tensor {} not found", name))?;
        let data = model.tensor_data(tensor);
        
        match tensor.quant_type {
            QuantType::Fp32 => {
                Ok(data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect())
            }
            QuantType::Fp16 => {
                Ok(data.chunks_exact(2)
                    .map(|b| fp16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect())
            }
            _ => Err(format!("Unsupported quant type for {}", name)),
        }
    }

    /// Get embedding for a token
    fn embed(&self, token_id: usize) -> Result<Vec<f32>, String> {
        let h = self.config.hidden_size;
        let start = token_id * h;
        let end = start + h;
        if end > self.embed_tokens.len() {
            return Err(format!("Token {} out of vocab range", token_id));
        }
        Ok(self.embed_tokens[start..end].to_vec())
    }

    /// Matmul with automatic dequantization
    fn matmul(&self, tensor_name: &str, layer: usize, x: &[f32], y: &mut [f32]) -> Result<(), String> {
        let full_name = format!("blk.{}.{}", layer, tensor_name);
        let tensor = self.model.find_tensor(&full_name)
            .ok_or_else(|| format!("Tensor {} not found", full_name))?;
        let data = self.model.tensor_data(tensor);
        let rows = tensor.rows as usize;
        let cols = tensor.cols as usize;

        match tensor.quant_type {
            QuantType::Fp32 => {
                let weights: Vec<f32> = data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                dense_sgemv_f32(&weights, x, y, rows, cols);
            }
            QuantType::Fp16 => {
                let weights: Vec<f32> = data.chunks_exact(2)
                    .map(|b| fp16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect();
                dense_sgemv_f32(&weights, x, y, rows, cols);
            }
            QuantType::Q4K => {
                let bpr = data.len() / rows;
                q4k_matmul_vec(data, x, y, rows, bpr);
            }
            QuantType::Q6K => {
                let bpr = data.len() / rows;
                q6k_matmul_vec(data, x, y, rows, bpr);
            }
        }
        Ok(())
    }

    /// Get norm weights for a layer
    fn get_norm(&self, tensor_name: &str, layer: usize) -> Vec<f32> {
        let full_name = format!("blk.{}.{}", layer, tensor_name);
        if let Some(tensor) = self.model.find_tensor(&full_name) {
            let data = self.model.tensor_data(tensor);
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        } else {
            vec![1.0; self.config.hidden_size]
        }
    }

    /// Forward pass for a single token
    pub fn forward_token(&mut self, token_id: usize, pos: usize) -> Result<Vec<f32>, String> {
        let h = self.config.hidden_size;
        let nh = self.config.num_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let ff = self.config.ffn_dim;

        // Get embedding
        let mut x = self.embed(token_id)?;

        // Precompute RoPE
        let half_dim = hd / 2;
        let mut rope_cos = vec![0.0_f32; half_dim];
        let mut rope_sin = vec![0.0_f32; half_dim];
        for i in 0..half_dim {
            let freq = 1.0 / self.config.rope_theta.powf(2.0 * i as f32 / hd as f32);
            let angle = pos as f32 * freq;
            rope_cos[i] = angle.cos();
            rope_sin[i] = angle.sin();
        }

        // Per-layer forward
        for layer in 0..self.config.num_layers {
            let residual = x.clone();

            // Attention norm
            let attn_norm_w = self.get_norm("attn_norm.weight", layer);
            let mut normed = vec![0.0; h];
            rms_norm(&x, &attn_norm_w, &mut normed, 1e-5);

            // Q/K/V projections
            let mut q = vec![0.0_f32; nh * hd];
            let mut k = vec![0.0_f32; nkv * hd];
            let mut v = vec![0.0_f32; nkv * hd];

            self.matmul("attn_q.weight", layer, &normed, &mut q)?;
            self.matmul("attn_k.weight", layer, &normed, &mut k)?;
            self.matmul("attn_v.weight", layer, &normed, &mut v)?;

            // RoPE
            let mut q_roped = vec![0.0_f32; q.len()];
            let mut k_roped = vec![0.0_f32; k.len()];
            apply_rope(&q, &rope_cos, &rope_sin, &mut q_roped, nh, hd);
            apply_rope(&k, &rope_cos, &rope_sin, &mut k_roped, nkv, hd);

            // Write KV cache
            self.kv_cache[layer].write(pos, &k_roped, &v);

            // GQA attention
            let seq_len = pos + 1;
            let mut attn_out = vec![0.0_f32; nh * hd];
            gqa_attention(
                &q_roped,
                &self.kv_cache[layer].k,
                &self.kv_cache[layer].v,
                &mut attn_out,
                nh, nkv, hd, seq_len,
            );

            // O projection
            let mut o_out = vec![0.0_f32; h];
            self.matmul("attn_output.weight", layer, &attn_out, &mut o_out)?;

            // Residual
            x = residual.iter().zip(o_out.iter()).map(|(r, o)| r + o).collect();

            // FFN
            let ffn_residual = x.clone();
            let ffn_norm_w = self.get_norm("ffn_norm.weight", layer);
            let mut ffn_normed = vec![0.0; h];
            rms_norm(&x, &ffn_norm_w, &mut ffn_normed, 1e-5);

            let mut gate = vec![0.0_f32; ff];
            let mut up = vec![0.0_f32; ff];
            let mut down_in = vec![0.0_f32; ff];
            let mut down_out = vec![0.0_f32; h];

            self.matmul("ffn_gate.weight", layer, &ffn_normed, &mut gate)?;
            self.matmul("ffn_up.weight", layer, &ffn_normed, &mut up)?;
            silu_mul(&gate, &up, &mut down_in);
            self.matmul("ffn_down.weight", layer, &down_in, &mut down_out)?;

            x = ffn_residual.iter().zip(down_out.iter()).map(|(r, d)| r + d).collect();
        }

        // Final norm
        let mut normed = vec![0.0; h];
        rms_norm(&x, &self.final_norm, &mut normed, 1e-5);

        // LM head → logits
        let mut logits = vec![0.0_f32; self.config.vocab_size];
        dense_sgemv_f32(&self.lm_head, &normed, &mut logits, self.config.vocab_size, h);

        Ok(logits)
    }

    /// Sample next token (temperature sampling)
    pub fn sample(&self, logits: &[f32], temperature: f32) -> usize {
        if temperature <= 0.0 || temperature < 0.01 {
            // Greedy
            logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        } else {
            // Temperature sampling
            let scaled: Vec<f32> = logits.iter().map(|l| l / temperature).collect();
            let mut probs = vec![0.0_f32; scaled.len()];
            softmax(&scaled, &mut probs);
            
            // Simple top-p sampling (p=0.9)
            let mut sorted: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            let mut cumsum = 0.0_f32;
            let mut cutoff = sorted.len();
            for (i, (_, p)) in sorted.iter().enumerate() {
                cumsum += p;
                if cumsum > 0.9 {
                    cutoff = i + 1;
                    break;
                }
            }
            
            // Re-normalize
            let top_p: Vec<(usize, f32)> = sorted[..cutoff].to_vec();
            let sum: f32 = top_p.iter().map(|(_, p)| p).sum();
            
            // Random selection
            let r: f32 = rand::random::<f32>() * sum;
            let mut acc = 0.0_f32;
            for (idx, p) in &top_p {
                acc += p;
                if acc >= r {
                    return *idx;
                }
            }
            top_p.last().map(|(idx, _)| *idx).unwrap_or(0)
        }
    }

    /// Generate text from a prompt (simplified - expects token IDs)
    pub fn generate(&mut self, prompt_tokens: &[usize], max_tokens: usize, temperature: f32) -> Vec<usize> {
        let mut output = Vec::new();
        
        // Process prompt
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            let _ = self.forward_token(tok, pos);
        }

        // Generate
        let mut pos = prompt_tokens.len();
        let mut last_tok = *prompt_tokens.last().unwrap_or(&0);
        
        for _ in 0..max_tokens {
            match self.forward_token(last_tok, pos) {
                Ok(logits) => {
                    let next_tok = self.sample(&logits, temperature);
                    output.push(next_tok);
                    
                    // Stop on EOS (assuming token 2 is EOS for most models)
                    if next_tok == 2 || next_tok == 151643 { // Qwen EOS
                        break;
                    }
                    
                    last_tok = next_tok;
                    pos += 1;
                }
                Err(_) => break,
            }
        }

        output
    }

    /// Reset KV cache for new conversation
    pub fn reset(&mut self) {
        for cache in &mut self.kv_cache {
            cache.k.fill(0.0);
            cache.v.fill(0.0);
        }
    }
}

// ── Global Engine Instance ──────────────────────────────────────────────────

static ENGINE_7B: OnceLock<Arc<Mutex<Option<RustInferenceEngine>>>> = OnceLock::new();
static ENGINE_35B: OnceLock<Arc<Mutex<Option<RustInferenceEngine>>>> = OnceLock::new();

pub fn get_engine_7b() -> Arc<Mutex<Option<RustInferenceEngine>>> {
    ENGINE_7B.get_or_init(|| Arc::new(Mutex::new(None))).clone()
}

pub fn get_engine_35b() -> Arc<Mutex<Option<RustInferenceEngine>>> {
    ENGINE_35B.get_or_init(|| Arc::new(Mutex::new(None))).clone()
}

/// Initialize engines from pack files
pub fn init_engines(repo_root: &str) -> Result<String, String> {
    let root = PathBuf::from(repo_root);
    let mut loaded = Vec::new();

    // Try to load 7B engine (deepseek-r1-7b or qwen2.5-coder_3b)
    let packs_7b = [
        "deepseek-r1-7b.rftmwpk",
        "qwen2.5-coder_3b.rftmwpk",
    ];
    
    for pack in &packs_7b {
        let path = root.join(pack);
        if path.exists() {
            match RustInferenceEngine::load(path.to_str().unwrap_or_default()) {
                Ok(engine) => {
                    if let Ok(mut guard) = get_engine_7b().lock() {
                        *guard = Some(engine);
                        loaded.push(format!("7B: {}", pack));
                        break;
                    }
                }
                Err(e) => eprintln!("[rust-engine] Failed to load {}: {}", pack, e),
            }
        }
    }

    // Try to load 35B engine
    let packs_35b = [
        "qwen35-opus-distil.rftmwpk",
        "qwen35-opus-distil_finetuned.rftmwpk",
    ];
    
    for pack in &packs_35b {
        let path = root.join(pack);
        if path.exists() {
            match RustInferenceEngine::load(path.to_str().unwrap_or_default()) {
                Ok(engine) => {
                    if let Ok(mut guard) = get_engine_35b().lock() {
                        *guard = Some(engine);
                        loaded.push(format!("35B: {}", pack));
                        break;
                    }
                }
                Err(e) => eprintln!("[rust-engine] Failed to load {}: {}", pack, e),
            }
        }
    }

    if loaded.is_empty() {
        Err("No RFTMWPK2 model packs found".into())
    } else {
        Ok(format!("Loaded: {}", loaded.join(", ")))
    }
}

/// Check if Rust engines are available
pub fn has_rust_engine() -> (bool, bool) {
    let has_7b = get_engine_7b().lock().map(|g| g.is_some()).unwrap_or(false);
    let has_35b = get_engine_35b().lock().map(|g| g.is_some()).unwrap_or(false);
    (has_7b, has_35b)
}

// ── Streaming Inference ─────────────────────────────────────────────────────

use std::sync::mpsc;

/// Token stream for real-time generation
pub struct TokenStream {
    rx: mpsc::Receiver<StreamEvent>,
}

pub enum StreamEvent {
    Token(usize),
    Done,
    Error(String),
}

impl Iterator for TokenStream {
    type Item = StreamEvent;
    fn next(&mut self) -> Option<Self::Item> {
        self.rx.recv().ok()
    }
}

/// Generate tokens with streaming - returns immediately, tokens arrive via channel
pub fn generate_streaming(
    engine_arc: Arc<Mutex<Option<RustInferenceEngine>>>,
    prompt_tokens: Vec<usize>,
    max_tokens: usize,
    temperature: f32,
) -> TokenStream {
    let (tx, rx) = mpsc::channel();
    
    std::thread::spawn(move || {
        let mut guard = match engine_arc.lock() {
            Ok(g) => g,
            Err(_) => {
                let _ = tx.send(StreamEvent::Error("Failed to lock engine".into()));
                return;
            }
        };
        
        let engine = match guard.as_mut() {
            Some(e) => e,
            None => {
                let _ = tx.send(StreamEvent::Error("Engine not loaded".into()));
                return;
            }
        };
        
        // Process prompt
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            if engine.forward_token(tok, pos).is_err() {
                let _ = tx.send(StreamEvent::Error("Forward failed".into()));
                return;
            }
        }
        
        // Generate tokens one at a time
        let mut pos = prompt_tokens.len();
        let mut last_tok = *prompt_tokens.last().unwrap_or(&0);
        
        for _ in 0..max_tokens {
            match engine.forward_token(last_tok, pos) {
                Ok(logits) => {
                    let next_tok = engine.sample(&logits, temperature);
                    
                    if tx.send(StreamEvent::Token(next_tok)).is_err() {
                        return; // Receiver dropped
                    }
                    
                    // Stop on EOS
                    if next_tok == 2 || next_tok == 151643 {
                        break;
                    }
                    
                    last_tok = next_tok;
                    pos += 1;
                }
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(e));
                    return;
                }
            }
        }
        
        let _ = tx.send(StreamEvent::Done);
    });
    
    TokenStream { rx }
}
