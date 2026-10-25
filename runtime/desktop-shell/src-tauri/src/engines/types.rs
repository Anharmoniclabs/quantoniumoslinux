// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// Shared types for all RFT/RFTMW engines — ported from rft_kernel.h

use std::fmt;
use std::ops::{Add, Mul, Sub, Neg};

// ── Mathematical constants ───────────────────────────────────────────────────

pub const PHI: f64 = 1.618_033_988_749_894_8;
pub const PHI_INV: f64 = 0.618_033_988_749_894_8;
pub const PI: f64 = std::f64::consts::PI;
pub const TWO_PI: f64 = 2.0 * PI;

// ── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RftError {
    InvalidParam,
    Memory,
    Computation,
    NotInitialized,
    BufferTooSmall,
    AuthFailed,
    HashFailed,
}

impl fmt::Display for RftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParam => write!(f, "invalid parameter"),
            Self::Memory => write!(f, "memory allocation failed"),
            Self::Computation => write!(f, "computation error"),
            Self::NotInitialized => write!(f, "not initialized"),
            Self::BufferTooSmall => write!(f, "buffer too small"),
            Self::AuthFailed => write!(f, "authentication failed"),
            Self::HashFailed => write!(f, "hash failed"),
        }
    }
}

impl std::error::Error for RftError {}

pub type RftResult<T> = Result<T, RftError>;

// ── Complex number ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    #[inline]
    pub fn abs(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn conj(self) -> Self {
        Self { re: self.re, im: -self.im }
    }

    #[inline]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    #[inline]
    pub fn exp_i(theta: f64) -> Self {
        Self { re: theta.cos(), im: theta.sin() }
    }

    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self { re: self.re * s, im: self.im * s }
    }
}

impl Add for Complex {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl Sub for Complex {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

impl Mul for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Neg for Complex {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { re: -self.re, im: -self.im }
    }
}

impl Mul<f64> for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self { re: self.re * rhs, im: self.im * rhs }
    }
}

impl std::iter::Sum for Complex {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, c| acc + c)
    }
}

// ── RFT Variant ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum RftVariant {
    Standard = 0,
    Harmonic = 1,
    Fibonacci = 2,
    Chaotic = 3,
    Geometric = 4,
    PhiChaotic = 5,
    Hyperbolic = 6,
    Dct = 7,
    HybridDct = 8,
    Cascade = 9,
    AdaptiveSplit = 10,
    EntropyGuided = 11,
    Dictionary = 12,
    Canonical = 13,
    BinaryWave = 14,
    CryptoSis = 15,
}

impl Default for RftVariant {
    fn default() -> Self {
        Self::Standard
    }
}

// ── Configuration flags ──────────────────────────────────────────────────────

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct RftFlags: u32 {
        const DEFAULT         = 0x0000;
        const OPTIMIZE_SIMD   = 0x0001;
        const HIGH_PRECISION  = 0x0002;
        const QUANTUM_SAFE    = 0x0004;
        const DEBUG           = 0x0008;
        const TOPOLOGICAL     = 0x0010;
    }
}

// ── GGUF type codes ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufType {
    F32 = 0,
    F16 = 1,
    Q4K = 12,
    Q6K = 14,
}

impl GgufType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            12 => Some(Self::Q4K),
            14 => Some(Self::Q6K),
            _ => None,
        }
    }

    /// Bytes per 256-element block for quantized types, or per-element for float.
    pub fn tensor_nbytes(self, numel: u64) -> u64 {
        match self {
            Self::F32 => numel * 4,
            Self::F16 => numel * 2,
            Self::Q4K => (numel / 256) * 144,
            Self::Q6K => (numel / 256) * 210,
        }
    }
}
