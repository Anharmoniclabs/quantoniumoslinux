// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// mmap.rs — Port of rftmw_mmap.hpp
//
// Cross-platform memory-mapped model file support (zero-copy tensor access).

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use crate::engines::types::RftResult;
use crate::engines::types::RftError;
use super::inference::{fp16_to_f32, dequant_q4k_block, dequant_q6k_block, dense_sgemv_f32};

// ── Pack Header ──────────────────────────────────────────────────────────────

const MAGIC: &[u8; 8] = b"RFTMWPK2";

#[derive(Debug, Clone)]
pub struct PackHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub num_tensors: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub ffn_dim: u32,
    pub vocab_size: u32,
    pub rope_theta: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum QuantType {
    Fp32 = 0,
    Q4K = 1,
    Q6K = 2,
    Fp16 = 3,
}

impl QuantType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => QuantType::Fp32,
            1 => QuantType::Q4K,
            2 => QuantType::Q6K,
            3 => QuantType::Fp16,
            _ => QuantType::Fp32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub name: String,
    pub offset: u64,
    pub size_bytes: u64,
    pub rows: u32,
    pub cols: u32,
    pub quant_type: QuantType,
    pub flags: u32,
}

// ── MmapFile ─────────────────────────────────────────────────────────────────

pub struct MmapFile {
    mmap: Mmap,
}

impl MmapFile {
    pub fn open<P: AsRef<Path>>(path: P) -> RftResult<Self> {
        let file = File::open(path).map_err(|_| RftError::Memory)?;
        let mmap = unsafe { Mmap::map(&file).map_err(|_| RftError::Memory)? };
        Ok(Self { mmap })
    }

    pub fn data(&self) -> &[u8] {
        &self.mmap
    }

    pub fn len(&self) -> usize {
        self.mmap.len()
    }
}

// ── MmapModel ────────────────────────────────────────────────────────────────

pub struct MmapModel {
    file: MmapFile,
    pub header: PackHeader,
    pub tensors: Vec<TensorDesc>,
}

impl MmapModel {
    pub fn load<P: AsRef<Path>>(path: P) -> RftResult<Self> {
        let file = MmapFile::open(path)?;
        let data = file.data();
        if data.len() < 64 {
            return Err(RftError::InvalidParam);
        }

        // Parse header
        if &data[0..8] != MAGIC {
            return Err(RftError::InvalidParam);
        }

        let read_u32 = |off: usize| -> u32 {
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        };
        let read_f32 = |off: usize| -> f32 {
            f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        };

        let header = PackHeader {
            magic: *MAGIC,
            version: read_u32(8),
            num_tensors: read_u32(12),
            hidden_size: read_u32(16),
            num_layers: read_u32(20),
            num_heads: read_u32(24),
            num_kv_heads: read_u32(28),
            head_dim: read_u32(32),
            ffn_dim: read_u32(36),
            vocab_size: read_u32(40),
            rope_theta: read_f32(44),
        };

        // Parse tensor descriptors
        let num_t = header.num_tensors as usize;
        let mut tensors = Vec::with_capacity(num_t);
        let mut cursor = 64usize; // After header

        for _ in 0..num_t {
            if cursor + 4 > data.len() { break; }
            let name_len = read_u32(cursor) as usize;
            cursor += 4;
            if cursor + name_len > data.len() { break; }
            let name = String::from_utf8_lossy(&data[cursor..cursor + name_len]).to_string();
            cursor += name_len;

            if cursor + 28 > data.len() { break; }
            let offset = u64::from_le_bytes([
                data[cursor], data[cursor+1], data[cursor+2], data[cursor+3],
                data[cursor+4], data[cursor+5], data[cursor+6], data[cursor+7],
            ]);
            cursor += 8;
            let size_bytes = u64::from_le_bytes([
                data[cursor], data[cursor+1], data[cursor+2], data[cursor+3],
                data[cursor+4], data[cursor+5], data[cursor+6], data[cursor+7],
            ]);
            cursor += 8;
            let rows = read_u32(cursor); cursor += 4;
            let cols = read_u32(cursor); cursor += 4;
            let qt = read_u32(cursor); cursor += 4;
            let flags = read_u32(cursor); cursor += 4;

            tensors.push(TensorDesc {
                name,
                offset,
                size_bytes,
                rows,
                cols,
                quant_type: QuantType::from_u32(qt),
                flags,
            });
        }

        Ok(Self { file, header, tensors })
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorDesc> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn tensor_data(&self, tensor: &TensorDesc) -> &[u8] {
        let start = tensor.offset as usize;
        let end = start + tensor.size_bytes as usize;
        &self.file.data()[start..end.min(self.file.len())]
    }

    /// Dispatch matmul based on quantization type.
    pub fn matmul_vec(&self, tensor: &TensorDesc, x: &[f32], y: &mut [f32]) {
        let data = self.tensor_data(tensor);
        let rows = tensor.rows as usize;
        let cols = tensor.cols as usize;

        match tensor.quant_type {
            QuantType::Fp32 => {
                let weights: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                dense_sgemv_f32(&weights, x, y, rows, cols);
            }
            QuantType::Fp16 => {
                let weights: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|b| fp16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect();
                dense_sgemv_f32(&weights, x, y, rows, cols);
            }
            QuantType::Q4K => {
                let bytes_per_row = data.len() / rows;
                super::inference::q4k_matmul_vec(data, x, y, rows, bytes_per_row);
            }
            QuantType::Q6K => {
                let bytes_per_row = data.len() / rows;
                super::inference::q6k_matmul_vec(data, x, y, rows, bytes_per_row);
            }
        }
    }
}
