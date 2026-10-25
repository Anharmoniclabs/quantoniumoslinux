// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// rft_sis.rs — Rust port of rft_sis.c
//
// Experimental lattice-style hash combining φ-RFT diffusion with SIS compression.
// Parameters: n=512, m=1024, q=3329, β=100

use sha2::{Sha256, Digest};
use crate::engines::types::*;

const SIS_N: usize = 512;
const SIS_M: usize = 1024;
const SIS_Q: i64 = 3329;
const SIS_BETA: i32 = 100;
pub const SIS_HASH_SIZE: usize = 32;

// ── PRNG (SHA-256 counter mode) ──────────────────────────────────────────────

struct Prng {
    seed_block: [u8; 32],
    counter: u32,
    block: [u8; 32],
    offset: usize,
}

impl Prng {
    fn new(seed: &[u8]) -> Self {
        let seed_block = sha256(seed);
        Self {
            seed_block,
            counter: 0,
            block: [0u8; 32],
            offset: 32,
        }
    }

    fn refill(&mut self) {
        let mut input = [0u8; 36];
        input[..32].copy_from_slice(&self.seed_block);
        input[32..36].copy_from_slice(&self.counter.to_le_bytes());
        self.block = sha256(&input);
        self.offset = 0;
        self.counter = self.counter.wrapping_add(1);
    }

    fn next_u32(&mut self) -> u32 {
        if self.offset + 4 > 32 {
            self.refill();
        }
        let v = u32::from_le_bytes([
            self.block[self.offset],
            self.block[self.offset + 1],
            self.block[self.offset + 2],
            self.block[self.offset + 3],
        ]);
        self.offset += 4;
        v
    }

    fn sample_zq(&mut self) -> i32 {
        let limit = (u32::MAX as u64) + 1;
        let bound = limit - (limit % SIS_Q as u64);
        loop {
            let val = self.next_u32();
            if (val as u64) < bound {
                return (val as i64 % SIS_Q) as i32;
            }
        }
    }
}

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

// ── RFT-SIS Context ─────────────────────────────────────────────────────────

pub struct RftSisCtx {
    a: Vec<Vec<i32>>,        // M × N lattice matrix
    rft_phases: Vec<f64>,    // N phases
    variant: RftVariant,
    initialized: bool,
}

impl RftSisCtx {
    pub fn new(seed: Option<&[u8]>, variant: RftVariant) -> RftResult<Self> {
        let default_seed = b"RFT-SIS-v1.0-quantoniumos-2025";
        let seed = seed.unwrap_or(default_seed.as_slice());

        let mut prng = Prng::new(seed);

        // Generate lattice matrix A
        let mut a = vec![vec![0i32; SIS_N]; SIS_M];
        for row in a.iter_mut() {
            for val in row.iter_mut() {
                *val = prng.sample_zq();
            }
        }

        // Pre-compute RFT phases: θ_k = 2π·frac((k+1)·φ)
        let mut rft_phases = vec![0.0; SIS_N];
        for k in 0..SIS_N {
            let freq = (k as f64 + 1.0) * PHI;
            let frac = freq - freq.floor();
            rft_phases[k] = TWO_PI * frac;
        }

        Ok(Self {
            a,
            rft_phases,
            variant,
            initialized: true,
        })
    }

    pub fn hash(&self, input: &[u8]) -> RftResult<[u8; SIS_HASH_SIZE]> {
        if !self.initialized {
            return Err(RftError::NotInitialized);
        }

        // Step 1: Expand input to N-dimensional vector
        let mut expanded = vec![0.0_f64; SIS_N];
        let mut hash_chain = if input.is_empty() {
            sha256(b"")
        } else {
            sha256(input)
        };

        let mut chain_input = [0u8; 64];
        chain_input[..32].copy_from_slice(&hash_chain);
        if !input.is_empty() && input.len() <= 32 {
            chain_input[32..32 + input.len()].copy_from_slice(input);
        }

        let mut dim_idx = 0;
        while dim_idx < SIS_N {
            let mut i = 0;
            while i < 32 && dim_idx < SIS_N {
                let val = u32::from_le_bytes([
                    hash_chain[i],
                    hash_chain[i + 1],
                    hash_chain[i + 2],
                    hash_chain[i + 3],
                ]);
                expanded[dim_idx] = (val as f64 / 4_294_967_296.0) * 2.0 - 1.0;
                i += 4;
                dim_idx += 1;
            }
            hash_chain = sha256(&chain_input);
            chain_input[..32].copy_from_slice(&hash_chain);
        }

        // Step 2: Apply φ-RFT transformation (2 rounds)
        let rft1 = self.apply_rft_phases(&expanded);
        let rft2 = self.apply_rft_phases(&rft1);

        // Step 3: Quantize to short integer vector [-β, β]
        let s = quantize_to_sis(&rft2);

        // Step 4: Compute SIS: y = A·s mod q
        let y = self.compute_sis(&s);

        // Step 5: Final hash
        let y_bytes: Vec<u8> = y.iter()
            .flat_map(|&v| (v as i32).to_le_bytes().to_vec())
            .collect();
        let mut final_input = Vec::with_capacity(y_bytes.len() + 12);
        final_input.extend_from_slice(&y_bytes);
        final_input.extend_from_slice(b"RFT-SIS-v1.0");

        Ok(sha256(&final_input))
    }

    pub fn kdf(
        &self,
        master_key: &[u8],
        info: &[u8],
        output: &mut [u8],
    ) -> RftResult<()> {
        if !self.initialized {
            return Err(RftError::NotInitialized);
        }

        let mut counter = [0u8; 4];
        let mut offset = 0;

        while offset < output.len() {
            let mut block_input = Vec::with_capacity(master_key.len() + info.len() + 4);
            block_input.extend_from_slice(master_key);
            block_input.extend_from_slice(info);
            block_input.extend_from_slice(&counter);

            let block = self.hash(&block_input)?;
            let copy_len = (output.len() - offset).min(SIS_HASH_SIZE);
            output[offset..offset + copy_len].copy_from_slice(&block[..copy_len]);
            offset += copy_len;

            // Increment counter
            for b in counter.iter_mut() {
                *b = b.wrapping_add(1);
                if *b != 0 { break; }
            }
        }
        Ok(())
    }

    pub fn mac(&self, key: &[u8], data: &[u8]) -> RftResult<[u8; SIS_HASH_SIZE]> {
        if !self.initialized {
            return Err(RftError::NotInitialized);
        }

        // Inner hash: H(key || data)
        let mut inner_input = Vec::with_capacity(key.len() + data.len());
        inner_input.extend_from_slice(key);
        inner_input.extend_from_slice(data);
        let inner_hash = self.hash(&inner_input)?;

        // Outer hash: H(key || inner_hash)
        let mut outer_input = Vec::with_capacity(key.len() + SIS_HASH_SIZE);
        outer_input.extend_from_slice(key);
        outer_input.extend_from_slice(&inner_hash);
        self.hash(&outer_input)
    }

    pub fn mac_verify(&self, key: &[u8], data: &[u8], tag: &[u8; SIS_HASH_SIZE]) -> RftResult<()> {
        let computed = self.mac(key, data)?;
        // Constant-time comparison
        let diff = computed.iter().zip(tag.iter()).fold(0u8, |acc, (&a, &b)| acc | (a ^ b));
        if diff == 0 {
            Ok(())
        } else {
            Err(RftError::HashFailed)
        }
    }

    // ── Internal ─────────────────────────────────────────────────────────

    fn apply_rft_phases(&self, input: &[f64]) -> Vec<f64> {
        let n = input.len().min(SIS_N);
        let mut output = vec![0.0; n];
        for i in 0..n {
            let phase = self.rft_phases[i];
            let c = phase.cos();
            let s = phase.sin();
            let prev = if i == 0 { n - 1 } else { i - 1 };
            let next = if i == n - 1 { 0 } else { i + 1 };
            output[i] = input[i] * c + input[prev] * s * PHI + input[next] * s * PHI_INV;
        }
        output
    }

    fn compute_sis(&self, s: &[i32]) -> Vec<i32> {
        let mut y = vec![0i32; SIS_M];
        for (i, row) in self.a.iter().enumerate() {
            let mut sum: i64 = 0;
            for (j, &a_ij) in row.iter().enumerate() {
                sum += a_ij as i64 * s[j] as i64;
            }
            let mut val = (sum % SIS_Q) as i32;
            if val > (SIS_Q / 2) as i32 { val -= SIS_Q as i32; }
            if val < -(SIS_Q / 2) as i32 { val += SIS_Q as i32; }
            y[i] = val;
        }
        y
    }
}

fn quantize_to_sis(input: &[f64]) -> Vec<i32> {
    let max_val = input.iter().map(|v| v.abs()).fold(1e-15_f64, f64::max);
    let scale = (SIS_BETA as f64 * 0.95) / max_val;
    input
        .iter()
        .map(|&v| {
            let val = (v * scale).round() as i32;
            val.clamp(-SIS_BETA, SIS_BETA)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rft_sis_hash() {
        let ctx = RftSisCtx::new(None, RftVariant::Fibonacci).unwrap();
        let h1 = ctx.hash(b"hello world").unwrap();
        let h2 = ctx.hash(b"hello world").unwrap();
        assert_eq!(h1, h2, "deterministic hashes should match");

        let h3 = ctx.hash(b"hello world!").unwrap();
        assert_ne!(h1, h3, "different inputs should produce different hashes");
    }

    #[test]
    fn test_rft_sis_mac_verify() {
        let ctx = RftSisCtx::new(None, RftVariant::Fibonacci).unwrap();
        let key = b"secret-key-1234";
        let data = b"authenticate this data";
        let tag = ctx.mac(key, data).unwrap();
        ctx.mac_verify(key, data, &tag).unwrap();

        let mut bad_tag = tag;
        bad_tag[0] ^= 1;
        assert!(ctx.mac_verify(key, data, &bad_tag).is_err());
    }
}
