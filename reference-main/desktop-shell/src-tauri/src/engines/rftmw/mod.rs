// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// rftmw/ — Rust port of the RFTMW inference engine
// ~8,746 lines C/C++ → organized Rust submodules

pub mod core;
pub mod fft;
pub mod inference;
pub mod compression;
pub mod kv_compress;
pub mod streaming;
pub mod mmap;
pub mod spectral;
