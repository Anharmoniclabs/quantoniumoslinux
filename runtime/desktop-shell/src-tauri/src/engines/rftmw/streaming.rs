// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// streaming.rs — Port of rftmw_streaming_engine.hpp
//
// Layer-streaming inference: one-layer-at-a-time loading for 7B models in ~630 MB.

use std::path::Path;
use crate::engines::types::{RftResult, RftError};
use super::mmap::{MmapModel, TensorDesc, QuantType};
use super::inference::*;
use super::kv_compress::{KVCompressor, KVCompressConfig};

// ── RFTL Chunk Reader ────────────────────────────────────────────────────────

const RFTL_MAGIC: &[u8; 4] = b"RFTL";

pub struct ChunkTensor {
    pub name: String,
    pub gguf_type: u32,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

pub struct ChunkReader {
    pub tensors: Vec<ChunkTensor>,
}

impl ChunkReader {
    pub fn load(raw: &[u8]) -> RftResult<Self> {
        if raw.len() < 12 || &raw[0..4] != RFTL_MAGIC {
            return Err(RftError::InvalidParam);
        }
        let _version = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
        let num_tensors = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
        let mut cursor = 12;
        let mut tensors = Vec::with_capacity(num_tensors);

        for _ in 0..num_tensors {
            if cursor + 4 > raw.len() { break; }
            let name_len = u32::from_le_bytes([raw[cursor], raw[cursor+1], raw[cursor+2], raw[cursor+3]]) as usize;
            cursor += 4;
            if cursor + name_len > raw.len() { break; }
            let name = String::from_utf8_lossy(&raw[cursor..cursor + name_len]).to_string();
            cursor += name_len;

            if cursor + 4 > raw.len() { break; }
            let gguf_type = u32::from_le_bytes([raw[cursor], raw[cursor+1], raw[cursor+2], raw[cursor+3]]);
            cursor += 4;

            if cursor + 4 > raw.len() { break; }
            let ndims = u32::from_le_bytes([raw[cursor], raw[cursor+1], raw[cursor+2], raw[cursor+3]]) as usize;
            cursor += 4;

            let mut shape = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                if cursor + 8 > raw.len() { break; }
                let dim = u64::from_le_bytes([
                    raw[cursor], raw[cursor+1], raw[cursor+2], raw[cursor+3],
                    raw[cursor+4], raw[cursor+5], raw[cursor+6], raw[cursor+7],
                ]);
                cursor += 8;
                shape.push(dim);
            }

            if cursor + 8 > raw.len() { break; }
            let data_size = u64::from_le_bytes([
                raw[cursor], raw[cursor+1], raw[cursor+2], raw[cursor+3],
                raw[cursor+4], raw[cursor+5], raw[cursor+6], raw[cursor+7],
            ]) as usize;
            cursor += 8;

            if cursor + data_size > raw.len() { break; }
            let data = raw[cursor..cursor + data_size].to_vec();
            cursor += data_size;

            tensors.push(ChunkTensor { name, gguf_type, shape, data });
        }

        Ok(Self { tensors })
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> RftResult<Self> {
        let data = std::fs::read(path).map_err(|_| RftError::Memory)?;
        Self::load(&data)
    }

    pub fn find(&self, name: &str) -> Option<&ChunkTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

// ── Streaming Config ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub kv_compress_enabled: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            num_layers: 36,
            hidden_size: 2048,
            num_heads: 16,
            num_kv_heads: 2,
            head_dim: 128,
            ffn_dim: 5632,
            vocab_size: 151936,
            rope_theta: 1_000_000.0,
            kv_compress_enabled: true,
        }
    }
}

// ── StreamWeight2D ───────────────────────────────────────────────────────────

pub struct StreamWeight2D {
    pub data: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
    pub quant_type: u32, // 0=F32, 1=F16, 12=Q4_K, 14=Q6_K
}

impl StreamWeight2D {
    pub fn bind(tensor: &ChunkTensor) -> Self {
        let rows = tensor.shape.first().copied().unwrap_or(0) as usize;
        let cols = tensor.shape.get(1).copied().unwrap_or(0) as usize;
        Self {
            data: tensor.data.clone(),
            rows,
            cols,
            quant_type: tensor.gguf_type,
        }
    }

    pub fn matmul(&self, x: &[f32], y: &mut [f32]) {
        match self.quant_type {
            0 => { // F32
                let weights: Vec<f32> = self.data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                dense_sgemv_f32(&weights, x, y, self.rows, self.cols);
            }
            1 => { // F16
                let weights: Vec<f32> = self.data
                    .chunks_exact(2)
                    .map(|b| fp16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect();
                dense_sgemv_f32(&weights, x, y, self.rows, self.cols);
            }
            12 => { // Q4_K
                let bpr = if self.rows > 0 { self.data.len() / self.rows } else { 0 };
                q4k_matmul_vec(&self.data, x, y, self.rows, bpr);
            }
            14 => { // Q6_K
                let bpr = if self.rows > 0 { self.data.len() / self.rows } else { 0 };
                q6k_matmul_vec(&self.data, x, y, self.rows, bpr);
            }
            _ => {}
        }
    }
}

// ── Layer KV Cache ───────────────────────────────────────────────────────────

pub struct LayerKVCache {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub max_seq: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl LayerKVCache {
    pub fn new(max_seq: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let cap = max_seq * num_kv_heads * head_dim;
        Self {
            k_cache: vec![0.0; cap],
            v_cache: vec![0.0; cap],
            max_seq,
            num_kv_heads,
            head_dim,
        }
    }

    pub fn write_kv(&mut self, pos: usize, k: &[f32], v: &[f32]) {
        let offset = pos * self.num_kv_heads * self.head_dim;
        let len = self.num_kv_heads * self.head_dim;
        self.k_cache[offset..offset + len].copy_from_slice(&k[..len]);
        self.v_cache[offset..offset + len].copy_from_slice(&v[..len]);
    }
}

// ── Streaming Model ──────────────────────────────────────────────────────────

pub struct StreamingModel {
    pub config: StreamingConfig,
    pub embed_tokens: Vec<f32>,
    pub lm_head_weight: Vec<f32>,
    pub final_norm_weight: Vec<f32>,
    pub kv_caches: Vec<LayerKVCache>,
    pub kv_compressor: Option<KVCompressor>,
}

impl StreamingModel {
    pub fn new(config: StreamingConfig) -> Self {
        let mut kv_caches = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            kv_caches.push(LayerKVCache::new(2048, config.num_kv_heads, config.head_dim));
        }

        let kv_compressor = if config.kv_compress_enabled {
            Some(KVCompressor::new(KVCompressConfig {
                head_dim: config.head_dim,
                num_kv_heads: config.num_kv_heads,
                base_keep_ratio: 0.60,
                total_layers: config.num_layers,
                min_positions_before_compress: 4,
            }))
        } else {
            None
        };

        Self {
            config,
            embed_tokens: Vec::new(),
            lm_head_weight: Vec::new(),
            final_norm_weight: Vec::new(),
            kv_caches,
            kv_compressor,
        }
    }

    /// Load embeddings and LM head from RFTL chunk file.
    pub fn load_embeddings<P: AsRef<Path>>(&mut self, path: P) -> RftResult<()> {
        let chunk = ChunkReader::load_from_file(path)?;

        if let Some(t) = chunk.find("token_embd.weight") {
            self.embed_tokens = t.data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
        }
        if let Some(t) = chunk.find("output.weight") {
            self.lm_head_weight = t.data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
        }
        if let Some(t) = chunk.find("output_norm.weight") {
            self.final_norm_weight = t.data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
        }

        Ok(())
    }

    /// Single-token forward pass (embedding → per-layer → norm → LM head → logits).
    pub fn forward_token(
        &mut self,
        token_id: usize,
        pos: usize,
        cache_len: usize,
        layer_paths: &[String],
    ) -> RftResult<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;

        // Embedding lookup
        if token_id * hidden + hidden > self.embed_tokens.len() {
            return Err(RftError::InvalidParam);
        }
        let mut x: Vec<f32> = self.embed_tokens[token_id * hidden..(token_id + 1) * hidden].to_vec();

        // Precompute RoPE cos/sin for this position
        let half_dim = head_dim / 2;
        let mut rope_cos = vec![0.0_f32; half_dim];
        let mut rope_sin = vec![0.0_f32; half_dim];
        for i in 0..half_dim {
            let freq = 1.0 / self.config.rope_theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            rope_cos[i] = angle.cos();
            rope_sin[i] = angle.sin();
        }

        // Per-layer forward
        for (layer_idx, layer_path) in layer_paths.iter().enumerate() {
            let chunk = ChunkReader::load_from_file(layer_path)?;
            let residual = x.clone();

            // Attention norm
            if let Some(t) = chunk.find("attn_norm.weight") {
                let norm_w: Vec<f32> = t.data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                let mut normed = vec![0.0; hidden];
                rms_norm(&x, &norm_w, &mut normed, 1e-5);
                x = normed;
            }

            // Q/K/V projections
            let mut q = vec![0.0_f32; num_heads * head_dim];
            let mut k = vec![0.0_f32; num_kv_heads * head_dim];
            let mut v = vec![0.0_f32; num_kv_heads * head_dim];

            if let Some(t) = chunk.find("attn_q.weight") {
                StreamWeight2D::bind(t).matmul(&x, &mut q);
            }
            if let Some(t) = chunk.find("attn_k.weight") {
                StreamWeight2D::bind(t).matmul(&x, &mut k);
            }
            if let Some(t) = chunk.find("attn_v.weight") {
                StreamWeight2D::bind(t).matmul(&x, &mut v);
            }

            // RoPE
            let mut q_roped = vec![0.0_f32; q.len()];
            let mut k_roped = vec![0.0_f32; k.len()];
            apply_rope(&q, &rope_cos, &rope_sin, &mut q_roped, num_heads, head_dim);
            apply_rope(&k, &rope_cos, &rope_sin, &mut k_roped, num_kv_heads, head_dim);

            // Write to KV cache
            self.kv_caches[layer_idx].write_kv(pos, &k_roped, &v);

            // GQA attention
            let seq_len = cache_len + 1;
            let mut attn_out = vec![0.0_f32; num_heads * head_dim];
            gqa_attention(
                &q_roped,
                &self.kv_caches[layer_idx].k_cache,
                &self.kv_caches[layer_idx].v_cache,
                &mut attn_out,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
            );

            // O projection
            let mut o_out = vec![0.0_f32; hidden];
            if let Some(t) = chunk.find("attn_output.weight") {
                StreamWeight2D::bind(t).matmul(&attn_out, &mut o_out);
            }

            // Residual add
            for (xi, ri) in x.iter_mut().zip(residual.iter()) {
                *xi = o_out[xi as *const _ as usize % o_out.len()] + ri;
            }
            x = residual.iter().zip(o_out.iter()).map(|(r, o)| r + o).collect();

            // FFN norm
            let ffn_residual = x.clone();
            if let Some(t) = chunk.find("ffn_norm.weight") {
                let norm_w: Vec<f32> = t.data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                let mut normed = vec![0.0; hidden];
                rms_norm(&x, &norm_w, &mut normed, 1e-5);
                x = normed;
            }

            // SwiGLU FFN
            let ffn_dim = self.config.ffn_dim;
            let mut gate = vec![0.0_f32; ffn_dim];
            let mut up = vec![0.0_f32; ffn_dim];
            let mut down_in = vec![0.0_f32; ffn_dim];
            let mut down_out = vec![0.0_f32; hidden];

            if let Some(t) = chunk.find("ffn_gate.weight") {
                StreamWeight2D::bind(t).matmul(&x, &mut gate);
            }
            if let Some(t) = chunk.find("ffn_up.weight") {
                StreamWeight2D::bind(t).matmul(&x, &mut up);
            }
            silu_mul(&gate, &up, &mut down_in);
            if let Some(t) = chunk.find("ffn_down.weight") {
                StreamWeight2D::bind(t).matmul(&down_in, &mut down_out);
            }

            // Residual add
            x = ffn_residual.iter().zip(down_out.iter()).map(|(r, d)| r + d).collect();

            // KV compression for past positions
            if let Some(ref compressor) = self.kv_compressor {
                if cache_len >= 4 {
                    let kv = &mut self.kv_caches[layer_idx];
                    compressor.compress_layer(
                        layer_idx,
                        cache_len,
                        &mut kv.k_cache,
                        &mut kv.v_cache,
                    );
                }
            }
        }

        // Final norm
        if !self.final_norm_weight.is_empty() {
            let mut normed = vec![0.0; hidden];
            rms_norm(&x, &self.final_norm_weight, &mut normed, 1e-5);
            x = normed;
        }

        // LM head → logits
        let vocab = self.config.vocab_size;
        let mut logits = vec![0.0_f32; vocab];
        if !self.lm_head_weight.is_empty() {
            dense_sgemv_f32(&self.lm_head_weight, &x, &mut logits, vocab, hidden);
        }

        Ok(logits)
    }

    /// Greedy argmax sampling.
    pub fn sample_greedy(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}
