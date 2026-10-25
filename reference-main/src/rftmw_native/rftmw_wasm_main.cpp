/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
 *
 * rftmw_wasm_main.cpp — WASM entry point for streaming inference
 *
 * This file exists solely to force the compiler to instantiate the
 * C API from rftmw_streaming_engine.hpp.  All logic lives in the header.
 */

#define RFTMW_WASM_BUILD 1
#include "rftmw_streaming_engine.hpp"

// The extern "C" functions are defined in the header when RFTMW_WASM_BUILD is set.
// Emscripten will export them via EXPORTED_FUNCTIONS in the CMake config.
