/**
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
 *
 * rftmw_webgpu_runtime.js — WebGPU Compute Runtime for Spectral GEMV
 * ====================================================================
 *
 * Manages GPU device, pipelines, and buffer lifecycle for:
 *   - FFT + φ-phase forward transform (input → spectral domain)
 *   - Quantized sparse dot product (spectral GEMV)
 *   - Q4K dequant GEMV (fallback/baseline)
 *
 * Usage:
 *   const gpu = new RFTMWWebGPURuntime();
 *   await gpu.init();
 *   gpu.uploadPhaseLUT(lut);              // once
 *   gpu.uploadSpectralWeights(row, data); // per layer
 *   gpu.spectralGemv(x, y, outFeatures);  // per matmul
 */

const PHI = 0.6180339887498949;  // (√5 - 1) / 2

export class RFTMWWebGPURuntime {
  constructor() {
    this.device = null;
    this.adapter = null;
    this.fftPipeline = null;
    this.sparseDotPipeline = null;
    this.q4kPipeline = null;
    this.shaderModule = null;

    // Persistent buffers
    this.uniformBuffer = null;
    this.phaseLUTBuffer = null;
    this.xInBuffer = null;
    this.xSpecReBuffer = null;
    this.xSpecImBuffer = null;
    this.yOutBuffer = null;
    this.yReadBuffer = null;  // staging for GPU→CPU readback

    // Sparse weight buffers (per-layer, swapped on layer load)
    this.rowOffsetsBuffer = null;
    this.sparseDataBuffer = null;

    // Current dimensions
    this.outFeatures = 0;
    this.inFeatures = 0;
    this.blockSize = 128;
    this.nBlocks = 0;

    // Bind groups
    this.fftBindGroup = null;
    this.sparseDotBindGroup0 = null;
    this.sparseDotBindGroup1 = null;
  }

  /**
   * Initialize WebGPU adapter, device, and compile shaders.
   * Returns false if WebGPU is not available.
   */
  async init() {
    if (!navigator.gpu) {
      console.warn("[rftmw-gpu] WebGPU not available — falling back to WASM SIMD");
      return false;
    }

    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!this.adapter) {
      console.warn("[rftmw-gpu] No GPU adapter found");
      return false;
    }

    // Request device with sufficient limits for 7B model GEMV
    this.device = await this.adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: 256 * 1024 * 1024,  // 256 MB per buffer
        maxBufferSize: 256 * 1024 * 1024,
        maxComputeWorkgroupSizeX: 256,
        maxComputeInvocationsPerWorkgroup: 256,
        maxStorageBuffersPerShaderStage: 8,
      },
    });

    this.device.lost.then((info) => {
      console.error(`[rftmw-gpu] Device lost: ${info.message}`);
    });

    // Fetch and compile the WGSL shader
    const shaderCode = await this._loadShader();
    this.shaderModule = this.device.createShaderModule({
      label: "rftmw_spectral_gemv",
      code: shaderCode,
    });

    // Create compute pipelines
    this.fftPipeline = this.device.createComputePipeline({
      label: "fft_phi_forward",
      layout: "auto",
      compute: { module: this.shaderModule, entryPoint: "fft_phi_forward" },
    });

    this.sparseDotPipeline = this.device.createComputePipeline({
      label: "sparse_dot_rows",
      layout: "auto",
      compute: { module: this.shaderModule, entryPoint: "sparse_dot_rows" },
    });

    this.q4kPipeline = this.device.createComputePipeline({
      label: "q4k_gemv",
      layout: "auto",
      compute: { module: this.shaderModule, entryPoint: "q4k_gemv" },
    });

    console.log(
      `[rftmw-gpu] WebGPU initialized: ${this.adapter.info?.device || "unknown"}, ` +
      `max buffer: ${(this.device.limits.maxBufferSize / 1024 / 1024).toFixed(0)} MB`
    );

    return true;
  }

  /**
   * Load WGSL shader source. In production this would be bundled;
   * for development, fetch from the server.
   */
  async _loadShader() {
    // Try fetching from relative path (works in web worker context)
    try {
      const resp = await fetch("rftmw_spectral_gemv.wgsl");
      if (resp.ok) return await resp.text();
    } catch { /* fall through */ }

    // Fallback: inline the shader (populated at build time by bundler)
    if (typeof __RFTMW_WGSL_INLINE__ !== "undefined") {
      return __RFTMW_WGSL_INLINE__;
    }

    throw new Error("[rftmw-gpu] Could not load WGSL shader");
  }

  /**
   * Upload the phase LUT (512 floats: 256 cos + 256 sin).
   * Called once at model init time.
   */
  uploadPhaseLUT(cosLut, sinLut) {
    const data = new Float32Array(512);
    data.set(cosLut, 0);
    data.set(sinLut, 256);

    this.phaseLUTBuffer = this.device.createBuffer({
      label: "phase_lut",
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.phaseLUTBuffer, 0, data);
  }

  /**
   * Allocate GPU buffers for a specific GEMV dimension.
   * Called when layer dimensions change (typically once per layer shape).
   */
  allocateBuffers(outFeatures, inFeatures, blockSize = 128) {
    if (this.outFeatures === outFeatures &&
        this.inFeatures === inFeatures &&
        this.blockSize === blockSize) {
      return;  // already allocated
    }

    this.outFeatures = outFeatures;
    this.inFeatures = inFeatures;
    this.blockSize = blockSize;
    this.nBlocks = Math.ceil(inFeatures / blockSize);

    const log2Block = Math.log2(blockSize) | 0;

    // Free old buffers
    this._freeBuffers();

    // Uniform buffer (GemvUniforms struct)
    const uniformData = new ArrayBuffer(32);
    const uView = new DataView(uniformData);
    uView.setUint32(0, outFeatures, true);
    uView.setUint32(4, inFeatures, true);
    uView.setUint32(8, blockSize, true);
    uView.setUint32(12, this.nBlocks, true);
    uView.setUint32(16, log2Block, true);
    uView.setFloat32(20, PHI, true);
    uView.setUint32(24, 0, true);  // pad
    uView.setUint32(28, 0, true);  // pad

    this.uniformBuffer = this.device.createBuffer({
      label: "gemv_uniforms",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    // Input vector
    this.xInBuffer = this.device.createBuffer({
      label: "x_in",
      size: inFeatures * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Spectral cache
    const specSize = this.nBlocks * blockSize * 4;
    this.xSpecReBuffer = this.device.createBuffer({
      label: "x_spec_re",
      size: specSize,
      usage: GPUBufferUsage.STORAGE,
    });
    this.xSpecImBuffer = this.device.createBuffer({
      label: "x_spec_im",
      size: specSize,
      usage: GPUBufferUsage.STORAGE,
    });

    // Output vector
    this.yOutBuffer = this.device.createBuffer({
      label: "y_out",
      size: outFeatures * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.yReadBuffer = this.device.createBuffer({
      label: "y_read",
      size: outFeatures * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  _freeBuffers() {
    for (const name of [
      "uniformBuffer", "xInBuffer", "xSpecReBuffer", "xSpecImBuffer",
      "yOutBuffer", "yReadBuffer", "rowOffsetsBuffer", "sparseDataBuffer"
    ]) {
      if (this[name]) {
        this[name].destroy();
        this[name] = null;
      }
    }
    this.fftBindGroup = null;
    this.sparseDotBindGroup0 = null;
    this.sparseDotBindGroup1 = null;
  }

  /**
   * Upload spectral weight data for one projection (e.g., q_proj).
   *
   * @param {Uint32Array} rowOffsets - byte offsets for each row into sparseData
   * @param {Uint32Array} sparseData - packed sparse coefficient data
   */
  uploadSpectralWeights(rowOffsets, sparseData) {
    if (this.rowOffsetsBuffer) this.rowOffsetsBuffer.destroy();
    if (this.sparseDataBuffer) this.sparseDataBuffer.destroy();

    this.rowOffsetsBuffer = this.device.createBuffer({
      label: "row_offsets",
      size: rowOffsets.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.rowOffsetsBuffer, 0, rowOffsets);

    this.sparseDataBuffer = this.device.createBuffer({
      label: "sparse_data",
      size: sparseData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.sparseDataBuffer, 0, sparseData);

    // Rebuild bind groups
    this._buildBindGroups();
  }

  /**
   * Upload raw Q4K quantized weight data for one projection.
   * Used for the non-spectral GPU GEMV path.
   *
   * @param {Uint8Array} q4kData  - raw Q4K weight bytes (row-major)
   * @param {number} outFeatures  - number of output rows
   * @param {number} inFeatures   - input dimension
   * @param {number} bytesPerRow  - Q4K bytes per row
   */
  uploadQ4KWeights(q4kData, outFeatures, inFeatures, bytesPerRow) {
    if (this.sparseDataBuffer) this.sparseDataBuffer.destroy();

    // Reuse sparseDataBuffer slot for Q4K data (same binding slot 7)
    // Also need a separate Q4K uniforms buffer
    // For simplicity, pack both into group(0) bindings 6 and 7

    const q4kUniform = new ArrayBuffer(16);
    const v = new DataView(q4kUniform);
    v.setUint32(0, outFeatures, true);
    v.setUint32(4, inFeatures, true);
    v.setUint32(8, bytesPerRow, true);
    v.setUint32(12, 0, true);

    // TODO: create q4k uniform + data buffers and bind group for q4k_gemv
    // This is a placeholder — the spectral path is the primary target.
  }

  _buildBindGroups() {
    if (!this.uniformBuffer || !this.phaseLUTBuffer || !this.xInBuffer) return;

    // Bind group 0 for FFT pipeline (shared with sparse dot pass 2)
    this.fftBindGroup = this.device.createBindGroup({
      label: "fft_bind_group_0",
      layout: this.fftPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.phaseLUTBuffer } },
        { binding: 2, resource: { buffer: this.xInBuffer } },
        { binding: 3, resource: { buffer: this.xSpecReBuffer } },
        { binding: 4, resource: { buffer: this.xSpecImBuffer } },
        { binding: 5, resource: { buffer: this.yOutBuffer } },
      ],
    });

    // Bind group 0 for sparse dot pipeline (same layout as FFT)
    this.sparseDotBindGroup0 = this.device.createBindGroup({
      label: "sparse_dot_bind_group_0",
      layout: this.sparseDotPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.phaseLUTBuffer } },
        { binding: 2, resource: { buffer: this.xInBuffer } },
        { binding: 3, resource: { buffer: this.xSpecReBuffer } },
        { binding: 4, resource: { buffer: this.xSpecImBuffer } },
        { binding: 5, resource: { buffer: this.yOutBuffer } },
      ],
    });

    // Bind group 1 for sparse dot: row offsets + sparse data
    if (this.rowOffsetsBuffer && this.sparseDataBuffer) {
      this.sparseDotBindGroup1 = this.device.createBindGroup({
        label: "sparse_dot_bind_group_1",
        layout: this.sparseDotPipeline.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: { buffer: this.rowOffsetsBuffer } },
          { binding: 1, resource: { buffer: this.sparseDataBuffer } },
        ],
      });
    }
  }

  /**
   * Execute spectral GEMV: y = SpectralW @ x
   *
   * Two-pass dispatch:
   *   Pass 1 (FFT): workgroups = nBlocks, transforms input blocks
   *   Pass 2 (Dot): workgroups = outFeatures, sparse dot per row
   *
   * @param {Float32Array} x - input vector (inFeatures)
   * @returns {Float32Array} y - output vector (outFeatures)
   */
  async spectralGemv(x) {
    if (!this.fftBindGroup || !this.sparseDotBindGroup1) {
      throw new Error("[rftmw-gpu] Buffers not initialized — call allocateBuffers + uploadSpectralWeights first");
    }

    // Upload input
    this.device.queue.writeBuffer(this.xInBuffer, 0, x);

    // Encode both passes
    const encoder = this.device.createCommandEncoder({ label: "spectral_gemv" });

    // Pass 1: FFT + φ-phase forward
    const fftPass = encoder.beginComputePass({ label: "fft_phi_forward" });
    fftPass.setPipeline(this.fftPipeline);
    fftPass.setBindGroup(0, this.fftBindGroup);
    fftPass.dispatchWorkgroups(this.nBlocks);
    fftPass.end();

    // Pass 2: Sparse dot product per row
    const dotPass = encoder.beginComputePass({ label: "sparse_dot_rows" });
    dotPass.setPipeline(this.sparseDotPipeline);
    dotPass.setBindGroup(0, this.sparseDotBindGroup0);
    dotPass.setBindGroup(1, this.sparseDotBindGroup1);
    dotPass.dispatchWorkgroups(this.outFeatures);
    dotPass.end();

    // Copy output to staging buffer
    encoder.copyBufferToBuffer(
      this.yOutBuffer, 0,
      this.yReadBuffer, 0,
      this.outFeatures * 4
    );

    this.device.queue.submit([encoder.finish()]);

    // Map and read back
    await this.yReadBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(this.yReadBuffer.getMappedRange().slice(0));
    this.yReadBuffer.unmap();

    return result;
  }

  /**
   * Execute standard Q4K GEMV on GPU (non-spectral path).
   * Still faster than WASM for large matrices.
   */
  async q4kGemv(x) {
    // TODO: Implement Q4K bind group and dispatch
    // For now, this is a placeholder — the main speedup is spectral.
    throw new Error("[rftmw-gpu] Q4K GPU GEMV not yet implemented — use WASM fallback");
  }

  /**
   * Clean up all GPU resources.
   */
  destroy() {
    this._freeBuffers();
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }
  }
}
