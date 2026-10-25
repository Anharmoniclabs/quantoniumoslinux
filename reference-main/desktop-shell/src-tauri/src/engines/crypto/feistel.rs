// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// feistel.rs — Rust port of feistel_round48.c
//
// 48-round Feistel cipher with 128-bit blocks, φ-RFT integration:
//   - AES S-box substitution
//   - φ-RFT permutation (golden-ratio byte reordering)
//   - MixColumns GF(2^8) diffusion
//   - ARX operations
//   - Key derivation via RFT-SIS lattice KDF or classical HKDF

use sha2::{Sha256, Digest};
use crate::engines::types::*;
use super::rft_sis::RftSisCtx;

const ROUNDS: usize = 48;
const BLOCK_SIZE: usize = 16;
const KEY_SIZE: usize = 32;
const ROUND_KEY_SIZE: usize = 16;

// AES S-box
static SBOX: [u8; 256] = [
    0x63,0x7C,0x77,0x7B,0xF2,0x6B,0x6F,0xC5,0x30,0x01,0x67,0x2B,0xFE,0xD7,0xAB,0x76,
    0xCA,0x82,0xC9,0x7D,0xFA,0x59,0x47,0xF0,0xAD,0xD4,0xA2,0xAF,0x9C,0xA4,0x72,0xC0,
    0xB7,0xFD,0x93,0x26,0x36,0x3F,0xF7,0xCC,0x34,0xA5,0xE5,0xF1,0x71,0xD8,0x31,0x15,
    0x04,0xC7,0x23,0xC3,0x18,0x96,0x05,0x9A,0x07,0x12,0x80,0xE2,0xEB,0x27,0xB2,0x75,
    0x09,0x83,0x2C,0x1A,0x1B,0x6E,0x5A,0xA0,0x52,0x3B,0xD6,0xB3,0x29,0xE3,0x2F,0x84,
    0x53,0xD1,0x00,0xED,0x20,0xFC,0xB1,0x5B,0x6A,0xCB,0xBE,0x39,0x4A,0x4C,0x58,0xCF,
    0xD0,0xEF,0xAA,0xFB,0x43,0x4D,0x33,0x85,0x45,0xF9,0x02,0x7F,0x50,0x3C,0x9F,0xA8,
    0x51,0xA3,0x40,0x8F,0x92,0x9D,0x38,0xF5,0xBC,0xB6,0xDA,0x21,0x10,0xFF,0xF3,0xD2,
    0xCD,0x0C,0x13,0xEC,0x5F,0x97,0x44,0x17,0xC4,0xA7,0x7E,0x3D,0x64,0x5D,0x19,0x73,
    0x60,0x81,0x4F,0xDC,0x22,0x2A,0x90,0x88,0x46,0xEE,0xB8,0x14,0xDE,0x5E,0x0B,0xDB,
    0xE0,0x32,0x3A,0x0A,0x49,0x06,0x24,0x5C,0xC2,0xD3,0xAC,0x62,0x91,0x95,0xE4,0x79,
    0xE7,0xC8,0x37,0x6D,0x8D,0xD5,0x4E,0xA9,0x6C,0x56,0xF4,0xEA,0x65,0x7A,0xAE,0x08,
    0xBA,0x78,0x25,0x2E,0x1C,0xA6,0xB4,0xC6,0xE8,0xDD,0x74,0x1F,0x4B,0xBD,0x8B,0x8A,
    0x70,0x3E,0xB5,0x66,0x48,0x03,0xF6,0x0E,0x61,0x35,0x57,0xB9,0x86,0xC1,0x1D,0x9E,
    0xE1,0xF8,0x98,0x11,0x69,0xD9,0x8E,0x94,0x9B,0x1E,0x87,0xE9,0xCE,0x55,0x28,0xDF,
    0x8C,0xA1,0x89,0x0D,0xBF,0xE6,0x42,0x68,0x41,0x99,0x2D,0x0F,0xB0,0x54,0xBB,0x16,
];

// MixColumns matrix
static MIXCOL: [[u8; 4]; 4] = [
    [2, 3, 1, 1],
    [1, 2, 3, 1],
    [1, 1, 2, 3],
    [3, 1, 1, 2],
];

// Fibonacci mod 256 for φ-RFT phase rotation
static PHI_POWERS: [u8; 16] = [1,1,2,3,5,8,13,21,34,55,89,144,233,121,98,219];

// ── Feistel Cipher Context ───────────────────────────────────────────────────

pub struct FeistelCtx {
    round_keys: [[u8; ROUND_KEY_SIZE]; ROUNDS],
    pre_whiten: [u8; ROUND_KEY_SIZE],
    post_whiten: [u8; ROUND_KEY_SIZE],
    auth_key: [u8; KEY_SIZE],
    variant: RftVariant,
}

impl FeistelCtx {
    /// Initialize with experimental RFT-SIS lattice KDF, falling back to HKDF.
    pub fn new(master_key: &[u8], variant: RftVariant) -> RftResult<Self> {
        if master_key.len() < 16 {
            return Err(RftError::InvalidParam);
        }

        let mut ctx = Self {
            round_keys: [[0u8; ROUND_KEY_SIZE]; ROUNDS],
            pre_whiten: [0u8; ROUND_KEY_SIZE],
            post_whiten: [0u8; ROUND_KEY_SIZE],
            auth_key: [0u8; KEY_SIZE],
            variant,
        };

        // Try RFT-SIS KDF first, fall back to classical HKDF
        let sis = RftSisCtx::new(None, RftVariant::Fibonacci);
        let use_sis = sis.is_ok();
        let sis_ctx = sis.ok();

        for round in 0..ROUNDS {
            let info = format!("RFT_SIS_ROUND_{:02}", round);
            if let Some(ref s) = sis_ctx {
                if s.kdf(master_key, info.as_bytes(), &mut ctx.round_keys[round]).is_err() {
                    hkdf(master_key, info.as_bytes(), &mut ctx.round_keys[round]);
                }
            } else {
                hkdf(master_key, info.as_bytes(), &mut ctx.round_keys[round]);
            }
            // Apply golden ratio parameterization
            for i in 0..ROUND_KEY_SIZE {
                let phi_factor = ((round as f64 * PHI + i as f64 / PHI) % 256.0) as u8;
                ctx.round_keys[round][i] ^= phi_factor;
            }
        }

        // Derive whitening keys
        let derive = |info: &[u8], out: &mut [u8]| {
            if let Some(ref s) = sis_ctx {
                if s.kdf(master_key, info, out).is_ok() { return; }
            }
            hkdf(master_key, info, out);
        };

        derive(b"RFT_SIS_PRE_WHITEN_2025", &mut ctx.pre_whiten);
        derive(b"RFT_SIS_POST_WHITEN_2025", &mut ctx.post_whiten);
        derive(b"RFT_SIS_AUTH_KEY_2025", &mut ctx.auth_key);

        Ok(ctx)
    }

    /// Encrypt a single 128-bit block.
    pub fn encrypt_block(&self, plaintext: &[u8; BLOCK_SIZE]) -> [u8; BLOCK_SIZE] {
        let mut left = [0u8; 8];
        let mut right = [0u8; 8];
        left.copy_from_slice(&plaintext[..8]);
        right.copy_from_slice(&plaintext[8..]);

        xor_blocks(&mut left, &self.pre_whiten[..8]);
        xor_blocks(&mut right, &self.pre_whiten[8..]);

        for round in 0..ROUNDS {
            let mut ext = [0u8; BLOCK_SIZE];
            ext[..8].copy_from_slice(&right);
            ext[8..].copy_from_slice(&right);
            let temp = round_function(&ext, &self.round_keys[round]);
            // Fold 16-byte output to 8-byte to ensure all input bytes
            // influence the result (phi_permute maps some positions to bytes 8-15)
            let mut folded = [0u8; 8];
            for i in 0..8 { folded[i] = temp[i] ^ temp[i + 8]; }
            xor_blocks(&mut left, &folded);

            if round < ROUNDS - 1 {
                std::mem::swap(&mut left, &mut right);
            }
        }

        xor_blocks(&mut left, &self.post_whiten[..8]);
        xor_blocks(&mut right, &self.post_whiten[8..]);

        let mut ciphertext = [0u8; BLOCK_SIZE];
        ciphertext[..8].copy_from_slice(&left);
        ciphertext[8..].copy_from_slice(&right);
        ciphertext
    }

    /// Decrypt a single 128-bit block (reversed round key order).
    pub fn decrypt_block(&self, ciphertext: &[u8; BLOCK_SIZE]) -> [u8; BLOCK_SIZE] {
        let mut left = [0u8; 8];
        let mut right = [0u8; 8];
        left.copy_from_slice(&ciphertext[..8]);
        right.copy_from_slice(&ciphertext[8..]);

        xor_blocks(&mut left, &self.post_whiten[..8]);
        xor_blocks(&mut right, &self.post_whiten[8..]);

        for round in 0..ROUNDS {
            let key_idx = ROUNDS - 1 - round;
            let mut ext = [0u8; BLOCK_SIZE];
            ext[..8].copy_from_slice(&right);
            ext[8..].copy_from_slice(&right);
            let temp = round_function(&ext, &self.round_keys[key_idx]);
            let mut folded = [0u8; 8];
            for i in 0..8 { folded[i] = temp[i] ^ temp[i + 8]; }
            xor_blocks(&mut left, &folded);

            if round < ROUNDS - 1 {
                std::mem::swap(&mut left, &mut right);
            }
        }

        xor_blocks(&mut left, &self.pre_whiten[..8]);
        xor_blocks(&mut right, &self.pre_whiten[8..]);

        let mut plaintext = [0u8; BLOCK_SIZE];
        plaintext[..8].copy_from_slice(&left);
        plaintext[8..].copy_from_slice(&right);
        plaintext
    }

    /// Encrypt arbitrary-length data (ECB mode for individual blocks).
    pub fn encrypt_stream(&self, plaintext: &[u8]) -> Vec<u8> {
        let pad_len = (BLOCK_SIZE - (plaintext.len() % BLOCK_SIZE)) % BLOCK_SIZE;
        let mut padded = plaintext.to_vec();
        padded.resize(padded.len() + pad_len, pad_len as u8);
        if pad_len == 0 {
            padded.resize(padded.len() + BLOCK_SIZE, BLOCK_SIZE as u8);
        }

        let mut out = Vec::with_capacity(padded.len());
        for chunk in padded.chunks_exact(BLOCK_SIZE) {
            let block: [u8; BLOCK_SIZE] = chunk.try_into().unwrap();
            out.extend_from_slice(&self.encrypt_block(&block));
        }
        out
    }

    /// Decrypt stream and remove padding.
    pub fn decrypt_stream(&self, ciphertext: &[u8]) -> RftResult<Vec<u8>> {
        if ciphertext.len() % BLOCK_SIZE != 0 || ciphertext.is_empty() {
            return Err(RftError::InvalidParam);
        }

        let mut out = Vec::with_capacity(ciphertext.len());
        for chunk in ciphertext.chunks_exact(BLOCK_SIZE) {
            let block: [u8; BLOCK_SIZE] = chunk.try_into().unwrap();
            out.extend_from_slice(&self.decrypt_block(&block));
        }

        // Remove PKCS7 padding
        let pad = *out.last().ok_or(RftError::InvalidParam)? as usize;
        if pad == 0 || pad > BLOCK_SIZE || out.len() < pad {
            return Err(RftError::InvalidParam);
        }
        out.truncate(out.len() - pad);
        Ok(out)
    }
}

impl Drop for FeistelCtx {
    fn drop(&mut self) {
        // Securely zero sensitive key material
        for rk in self.round_keys.iter_mut() {
            rk.fill(0);
        }
        self.pre_whiten.fill(0);
        self.post_whiten.fill(0);
        self.auth_key.fill(0);
    }
}

// ── Round function ───────────────────────────────────────────────────────────

fn round_function(input: &[u8; BLOCK_SIZE], round_key: &[u8; ROUND_KEY_SIZE]) -> [u8; BLOCK_SIZE] {
    let mut temp1 = [0u8; BLOCK_SIZE];
    let mut temp2 = [0u8; BLOCK_SIZE];
    let mut temp3 = [0u8; BLOCK_SIZE];
    let mut temp4 = [0u8; BLOCK_SIZE];

    // 1. XOR with round key
    for i in 0..BLOCK_SIZE {
        temp1[i] = input[i] ^ round_key[i];
    }

    // 2. S-box substitution
    for i in 0..BLOCK_SIZE {
        temp2[i] = SBOX[temp1[i] as usize];
    }

    // 3. φ-RFT permutation (golden-ratio byte reordering)
    phi_permute(&temp2, &mut temp3);

    // 4. MixColumns GF(2^8) diffusion
    mixcolumns(&temp3, &mut temp4);

    // 5. φ-RFT rotation (golden-ratio phase mixing)
    phi_rotate(&mut temp4);

    // 6. ARX operation
    arx_operation(&temp4, round_key)
}

fn phi_permute(input: &[u8; BLOCK_SIZE], output: &mut [u8; BLOCK_SIZE]) {
    output.fill(0);
    for i in 0..BLOCK_SIZE {
        let freq = (i as f64 + 1.0) * PHI;
        let frac = freq - freq.floor();
        let idx = (frac * BLOCK_SIZE as f64) as usize % BLOCK_SIZE;
        output[idx] = input[i];
    }
}

fn phi_rotate(data: &mut [u8; BLOCK_SIZE]) {
    for i in 0..BLOCK_SIZE {
        let rotation = PHI_POWERS[i];
        data[i] = data[i].wrapping_add(rotation) ^ (rotation >> 1);
    }
}

fn mixcolumns(input: &[u8; BLOCK_SIZE], output: &mut [u8; BLOCK_SIZE]) {
    let mut state = *input;
    for col in 0..4 {
        let c = col * 4;
        let column = [state[c], state[c + 1], state[c + 2], state[c + 3]];
        for row in 0..4 {
            let mut val = 0u8;
            for j in 0..4 {
                val ^= gf_mul(MIXCOL[row][j], column[j]);
            }
            state[c + row] = val;
        }
    }
    *output = state;
}

fn arx_operation(a: &[u8; BLOCK_SIZE], b: &[u8; ROUND_KEY_SIZE]) -> [u8; BLOCK_SIZE] {
    let mut output = [0u8; BLOCK_SIZE];
    for i in 0..4 {
        let a32 = u32::from_le_bytes([a[i*4], a[i*4+1], a[i*4+2], a[i*4+3]]);
        let b32 = u32::from_le_bytes([b[i*4], b[i*4+1], b[i*4+2], b[i*4+3]]);
        let b_next = u32::from_le_bytes([
            b[((i+1)&3)*4], b[((i+1)&3)*4+1], b[((i+1)&3)*4+2], b[((i+1)&3)*4+3]
        ]);
        let sum = a32.wrapping_add(b32);
        let rot = if (i & 1) != 0 { sum.rotate_left(13) } else { sum.rotate_left(7) };
        let result = rot ^ b_next;
        output[i*4..i*4+4].copy_from_slice(&result.to_le_bytes());
    }
    output
}

fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut result = 0u8;
    while b != 0 {
        if b & 1 != 0 {
            result ^= a;
        }
        a = if a & 0x80 != 0 { (a << 1) ^ 0x1B } else { a << 1 };
        b >>= 1;
    }
    result
}

fn xor_blocks(dst: &mut [u8], src: &[u8]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d ^= s;
    }
}

fn hkdf(key: &[u8], info: &[u8], output: &mut [u8]) {
    let mut hasher = Sha256::new();
    hasher.update(key);
    hasher.update(info);
    let result = hasher.finalize();
    let copy_len = output.len().min(32);
    output[..copy_len].copy_from_slice(&result[..copy_len]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_block() {
        let key = b"0123456789abcdef0123456789abcdef";
        let ctx = FeistelCtx::new(key, RftVariant::Standard).unwrap();
        let plaintext: [u8; 16] = *b"Hello, RFT Cryp!";
        let ciphertext = ctx.encrypt_block(&plaintext);
        let decrypted = ctx.decrypt_block(&ciphertext);
        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_encrypt_decrypt_stream() {
        let key = b"0123456789abcdef0123456789abcdef";
        let ctx = FeistelCtx::new(key, RftVariant::Standard).unwrap();
        let data = b"The quick brown fox jumps over the lazy dog";
        let encrypted = ctx.encrypt_stream(data);
        let decrypted = ctx.decrypt_stream(&encrypted).unwrap();
        assert_eq!(data.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_avalanche() {
        let key = b"0123456789abcdef0123456789abcdef";
        let ctx = FeistelCtx::new(key, RftVariant::Standard).unwrap();
        let mut pt1 = [0u8; 16];
        let mut pt2 = [0u8; 16];
        pt2[0] = 1; // single bit difference
        let ct1 = ctx.encrypt_block(&pt1);
        let ct2 = ctx.encrypt_block(&pt2);
        let diff_bits: u32 = ct1.iter().zip(ct2.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        // Expect roughly 50% bit change (64 of 128 bits)
        assert!(diff_bits > 30, "avalanche effect too weak: {} bits changed", diff_bits);
    }
}
