// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// split_gguf.rs — Rust port of src/apps/split_gguf_layers.cpp
//
// Splits a GGUF model into per-layer RFTL chunks for streaming inference.
// Chunk format:
//   [4] magic "RFTL"
//   [4] version (1)
//   [4] num_tensors
//   [4] reserved
//   For each tensor:
//     [4] name_len
//     [name_len] name (UTF-8)
//     [4] type (GGUF type code)
//     [4] n_dims
//     [n_dims*8] shape (u64 each)
//     [8] data_size
//     [data_size] raw tensor bytes
//     [pad to 32-byte alignment]

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write, BufWriter};
use std::path::{Path, PathBuf};
use regex::Regex;

use super::types::{GgufType, RftError, RftResult};

// ── GGUF binary cursor ──────────────────────────────────────────────────────

struct GgufCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> GgufCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_u32(&mut self) -> RftResult<u32> {
        if self.remaining() < 4 {
            return Err(RftError::BufferTooSmall);
        }
        let v = u32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> RftResult<u64> {
        if self.remaining() < 8 {
            return Err(RftError::BufferTooSmall);
        }
        let v = u64::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
            self.data[self.pos + 4],
            self.data[self.pos + 5],
            self.data[self.pos + 6],
            self.data[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(v)
    }

    fn read_string(&mut self) -> RftResult<String> {
        let n = self.read_u64()? as usize;
        if self.remaining() < n {
            return Err(RftError::BufferTooSmall);
        }
        let s = String::from_utf8_lossy(&self.data[self.pos..self.pos + n]).to_string();
        self.pos += n;
        Ok(s)
    }

    fn read_bytes(&self, offset: usize, len: usize) -> RftResult<&'a [u8]> {
        if offset + len > self.data.len() {
            return Err(RftError::BufferTooSmall);
        }
        Ok(&self.data[offset..offset + len])
    }

    fn skip_gguf_scalar(&mut self, type_code: u32) -> RftResult<()> {
        match type_code {
            0 | 1 | 7 => { self.pos += 1; }
            2 | 3 => { self.pos += 2; }
            4 | 5 | 6 => { self.pos += 4; }
            8 => {
                let n = self.read_u64()? as usize;
                self.pos += n;
            }
            10 | 11 | 12 => { self.pos += 8; }
            _ => return Err(RftError::InvalidParam),
        }
        Ok(())
    }

    fn skip_gguf_value(&mut self, type_code: u32) -> RftResult<()> {
        if type_code == 9 {
            let elem_type = self.read_u32()?;
            let n = self.read_u64()?;
            for _ in 0..n {
                self.skip_gguf_value(elem_type)?;
            }
        } else {
            self.skip_gguf_scalar(type_code)?;
        }
        Ok(())
    }

    fn read_string_array(&mut self) -> RftResult<Vec<String>> {
        let elem_type = self.read_u32()?;
        let n = self.read_u64()?;
        if elem_type != 8 {
            return Err(RftError::InvalidParam);
        }
        let mut out = Vec::with_capacity(n as usize);
        for _ in 0..n {
            out.push(self.read_string()?);
        }
        Ok(out)
    }

    fn offset(&self) -> usize {
        self.pos
    }
}

// ── Tensor entry ─────────────────────────────────────────────────────────────

struct TensorEntry<'a> {
    name: String,
    ne: Vec<u64>,
    gguf_type: u32,
    offset: u64,
    bytes: u64,
    data: &'a [u8],
}

// ── Chunk writer ─────────────────────────────────────────────────────────────

fn write_chunk(path: &Path, tensors: &[TensorEntry<'_>]) -> io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    // Header
    w.write_all(b"RFTL")?;
    w.write_all(&1u32.to_le_bytes())?;                          // version
    w.write_all(&(tensors.len() as u32).to_le_bytes())?;        // num_tensors
    w.write_all(&0u32.to_le_bytes())?;                          // reserved

    let mut written: u64 = 16;
    for t in tensors {
        // Name
        w.write_all(&(t.name.len() as u32).to_le_bytes())?;
        w.write_all(t.name.as_bytes())?;
        written += 4 + t.name.len() as u64;

        // Type and shape
        w.write_all(&t.gguf_type.to_le_bytes())?;
        w.write_all(&(t.ne.len() as u32).to_le_bytes())?;
        for &d in &t.ne {
            w.write_all(&d.to_le_bytes())?;
        }
        written += 4 + 4 + (t.ne.len() as u64 * 8);

        // Data
        w.write_all(&t.bytes.to_le_bytes())?;
        w.write_all(t.data)?;
        written += 8 + t.bytes;

        // Pad to 32-byte alignment
        let aligned = (written + 31) & !31;
        if aligned > written {
            let pad = vec![0u8; (aligned - written) as usize];
            w.write_all(&pad)?;
            written = aligned;
        }
    }

    w.flush()?;
    eprintln!(
        "  wrote {} ({} tensors, {} MB)",
        path.file_name().unwrap_or_default().to_string_lossy(),
        tensors.len(),
        written / (1024 * 1024)
    );
    Ok(())
}

// ── Layer classification ─────────────────────────────────────────────────────

fn extract_layer_index(name: &str) -> Option<i32> {
    let re = Regex::new(r"^blk\.(\d+)\.").unwrap();
    re.captures(name)
        .and_then(|cap| cap.get(1))
        .and_then(|m| m.as_str().parse::<i32>().ok())
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Split a GGUF model file into per-layer RFTL chunks.
/// Returns the number of layer chunks written.
pub fn split_gguf_layers(gguf_path: &Path, out_dir: &Path) -> RftResult<usize> {
    // Read entire file into memory
    let file_data = fs::read(gguf_path).map_err(|_| RftError::InvalidParam)?;
    let mut cursor = GgufCursor::new(&file_data);

    // Parse GGUF header
    let magic = cursor.read_u32()?;
    if magic != u32::from_le_bytes(*b"GGUF") {
        return Err(RftError::InvalidParam);
    }
    let _version = cursor.read_u32()?;
    let n_tensors = cursor.read_u64()?;
    let n_kv = cursor.read_u64()?;

    eprintln!("[split] {} tensors, {} metadata keys", n_tensors, n_kv);

    // Parse metadata (skip, but capture tokenizer if present)
    for _ in 0..n_kv {
        let key = cursor.read_string()?;
        let type_code = cursor.read_u32()?;
        if (key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges") && type_code == 9 {
            let _strings = cursor.read_string_array()?;
        } else {
            cursor.skip_gguf_value(type_code)?;
        }
    }

    // Parse tensor headers
    let mut all_tensors: Vec<(String, Vec<u64>, u32, u64, u64)> = Vec::new();
    for _ in 0..n_tensors {
        let name = cursor.read_string()?;
        let n_dims = cursor.read_u32()?;
        let mut ne = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            ne.push(cursor.read_u64()?);
        }
        let tensor_type = cursor.read_u32()?;
        let offset = cursor.read_u64()?;

        let numel: u64 = ne.iter().product();
        let gguf_type = GgufType::from_u32(tensor_type).ok_or(RftError::InvalidParam)?;
        let bytes = gguf_type.tensor_nbytes(numel);

        all_tensors.push((name, ne, tensor_type, offset, bytes));
    }

    // Compute data start (after all headers, aligned to 32)
    let header_end = cursor.offset() as u64;
    let data_start = (header_end + 31) & !31;

    // Build TensorEntry with data slices
    let entries: Vec<TensorEntry<'_>> = all_tensors
        .iter()
        .map(|(name, ne, gguf_type, offset, bytes)| {
            let start = (data_start + offset) as usize;
            let end = start + *bytes as usize;
            TensorEntry {
                name: name.clone(),
                ne: ne.clone(),
                gguf_type: *gguf_type,
                offset: *offset,
                bytes: *bytes,
                data: &file_data[start..end],
            }
        })
        .collect();

    // Classify tensors into layers vs embeddings
    let mut layer_tensors: HashMap<i32, Vec<usize>> = HashMap::new();
    let mut embedding_indices: Vec<usize> = Vec::new();
    let mut max_layer: i32 = -1;

    for (i, entry) in entries.iter().enumerate() {
        if let Some(layer_idx) = extract_layer_index(&entry.name) {
            layer_tensors.entry(layer_idx).or_default().push(i);
            max_layer = max_layer.max(layer_idx);
        } else {
            embedding_indices.push(i);
        }
    }

    let embedding_entries: Vec<&TensorEntry<'_>> =
        embedding_indices.iter().map(|&i| &entries[i]).collect();

    eprintln!(
        "[split] {} embedding/output tensors, {} layers",
        embedding_entries.len(),
        max_layer + 1
    );

    // Create output directory
    fs::create_dir_all(out_dir).map_err(|_| RftError::InvalidParam)?;

    // Write embedding chunk
    // We need to convert references for write_chunk
    let emb_owned: Vec<TensorEntry<'_>> = embedding_indices
        .iter()
        .map(|&i| TensorEntry {
            name: entries[i].name.clone(),
            ne: entries[i].ne.clone(),
            gguf_type: entries[i].gguf_type,
            offset: entries[i].offset,
            bytes: entries[i].bytes,
            data: entries[i].data,
        })
        .collect();
    write_chunk(&out_dir.join("embeddings.bin"), &emb_owned)
        .map_err(|_| RftError::Computation)?;

    // Write per-layer chunks
    for i in 0..=max_layer {
        if let Some(indices) = layer_tensors.get(&i) {
            let layer_entries: Vec<TensorEntry<'_>> = indices
                .iter()
                .map(|&idx| TensorEntry {
                    name: entries[idx].name.clone(),
                    ne: entries[idx].ne.clone(),
                    gguf_type: entries[idx].gguf_type,
                    offset: entries[idx].offset,
                    bytes: entries[idx].bytes,
                    data: entries[idx].data,
                })
                .collect();
            let fname = format!("layer_{:02}.bin", i);
            write_chunk(&out_dir.join(&fname), &layer_entries)
                .map_err(|_| RftError::Computation)?;
        } else {
            eprintln!("WARNING: no tensors for layer {}", i);
        }
    }

    let total_bytes: u64 = entries.iter().map(|t| t.bytes).sum();
    eprintln!(
        "[split] done: {} layer chunks + embeddings",
        max_layer + 1
    );
    eprintln!(
        "[split] total tensor data: {} MB",
        total_bytes / (1024 * 1024)
    );

    Ok((max_layer + 1) as usize)
}
