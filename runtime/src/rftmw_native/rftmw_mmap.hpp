/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
 *
 * rftmw_mmap.hpp - Memory-Mapped Pack File Support
 * =================================================
 *
 * Zero-copy mmap interface for .rftmwpk files.
 * Enables instant startup via on-demand page faults.
 *
 * Design goals:
 *   - No Python/NumPy overhead at inference time
 *   - Weights stay in kernel page cache
 *   - Contiguous C arrays, no heap allocation
 *   - Compatible with GGUF block layouts
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace rftmw {
namespace mmap {

// ===================================================================
// Portable FP16 to FP32 conversion (no F16C required)
// ===================================================================
inline float fp16_to_fp32(uint16_t h) {
    // IEEE 754 half precision breakdown:
    // Sign: 1 bit, Exponent: 5 bits, Mantissa: 10 bits
    uint32_t sign = (h >> 15) & 0x0001;
    uint32_t exp = (h >> 10) & 0x001F;
    uint32_t mant = h & 0x03FF;
    
    uint32_t result;
    
    if (exp == 0) {
        if (mant == 0) {
            // Zero
            result = sign << 31;
        } else {
            // Subnormal: normalize it
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= ~0x0400;
            result = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        // Inf or NaN
        result = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        // Normal number
        result = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    
    float f;
    std::memcpy(&f, &result, 4);
    return f;
}

// ===================================================================
// Pack file header (on-disk format, mmap-friendly)
// Must be 64-byte aligned for optimal page mapping
// ===================================================================
#pragma pack(push, 1)
struct PackHeader {
    char magic[8];           // "RFTMWPK2" (v2 = mmap-native)
    uint64_t version;        // Format version
    uint64_t header_size;    // Bytes until tensor data
    uint64_t num_tensors;    // Total tensor count
    uint64_t model_type;     // 0=qwen25, 1=qwen35moe, 2=llama, etc.
    uint64_t hidden_size;
    uint64_t num_layers;
    uint64_t num_heads;
    uint64_t num_kv_heads;
    uint64_t intermediate_size;
    uint64_t vocab_size;
    uint64_t rope_theta;     // Fixed-point * 1e6
    uint64_t max_position;
    uint64_t reserved[4];    // Future use
    char padding[64 - (8 + 11*8 + 4*8) % 64];  // Align to 64 bytes
};

// Tensor descriptor (on-disk, immediately follows header)
struct TensorDesc {
    char name[64];           // e.g. "layer.0.attn.q_proj"
    uint64_t offset;         // Byte offset from file start
    uint64_t size_bytes;     // Total bytes (packed quant data)
    uint64_t rows;
    uint64_t cols;
    uint32_t quant_type;     // 0=FP32, 1=Q4_K, 2=Q6_K, 3=FP16
    uint32_t flags;          // Reserved
};
#pragma pack(pop)

// ===================================================================
// Quant type enum (matches GGUF)
// ===================================================================
enum QuantType : uint32_t {
    QUANT_FP32 = 0,
    QUANT_Q4_K = 1,
    QUANT_Q6_K = 2,
    QUANT_FP16 = 3,
    QUANT_Q8_0 = 4,
};

// Block sizes
constexpr size_t Q4_K_BLOCK_SIZE = 144;  // 256 values
constexpr size_t Q6_K_BLOCK_SIZE = 210;  // 256 values
constexpr size_t BLOCK_VALUES = 256;

// ===================================================================
// MmapFile - Cross-platform memory-mapped file
// ===================================================================
class MmapFile {
public:
    MmapFile() = default;
    ~MmapFile() { close(); }

    // Non-copyable, movable
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;
    MmapFile(MmapFile&& other) noexcept { *this = std::move(other); }
    MmapFile& operator=(MmapFile&& other) noexcept {
        if (this != &other) {
            close();
            data_ = other.data_;
            size_ = other.size_;
#ifdef _WIN32
            file_ = other.file_;
            mapping_ = other.mapping_;
            other.file_ = INVALID_HANDLE_VALUE;
            other.mapping_ = nullptr;
#else
            fd_ = other.fd_;
            other.fd_ = -1;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    bool open(const std::string& path) {
        close();

#ifdef _WIN32
        file_ = CreateFileA(
            path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        if (file_ == INVALID_HANDLE_VALUE) return false;

        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(file_, &file_size)) {
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            return false;
        }
        size_ = static_cast<size_t>(file_size.QuadPart);

        mapping_ = CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) {
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            return false;
        }

        data_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
        if (!data_) {
            CloseHandle(mapping_);
            CloseHandle(file_);
            mapping_ = nullptr;
            file_ = INVALID_HANDLE_VALUE;
            return false;
        }
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) return false;

        struct stat st;
        if (fstat(fd_, &st) < 0) {
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        size_ = static_cast<size_t>(st.st_size);

        data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            ::close(fd_);
            fd_ = -1;
            data_ = nullptr;
            return false;
        }

        // Start readahead of the whole pack. madvise values are not flags: the
        // former MADV_SEQUENTIAL | MADV_WILLNEED evaluated to MADV_WILLNEED anyway,
        // and SEQUENTIAL would be wrong here (every token re-reads all weights, so
        // pages must not be dropped behind the reader).
        madvise(data_, size_, MADV_WILLNEED);
#endif
        return true;
    }

    void close() {
        if (!data_) return;
#ifdef _WIN32
        if (data_) UnmapViewOfFile(data_);
        if (mapping_) CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        file_ = INVALID_HANDLE_VALUE;
        mapping_ = nullptr;
#else
        if (data_) ::munmap(data_, size_);
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
#endif
        data_ = nullptr;
        size_ = 0;
    }

    const void* data() const { return data_; }
    size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr; }

    template<typename T>
    const T* as(size_t offset = 0) const {
        return reinterpret_cast<const T*>(static_cast<const char*>(data_) + offset);
    }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// ===================================================================
// MmapModel - Memory-mapped model with tensor lookup
// ===================================================================
class MmapModel {
public:
    bool load(const std::string& path) {
        if (!file_.open(path)) return false;
        
        // Validate magic
        const auto* header = file_.as<PackHeader>(0);
        if (std::memcmp(header->magic, "RFTMWPK2", 8) != 0) {
            // Try v1 format fallback
            if (std::memcmp(header->magic, "RFTMWPK1", 8) == 0) {
                // V1 not mmap-native, need conversion
                file_.close();
                return false;
            }
            file_.close();
            return false;
        }

        header_ = header;
        num_tensors_ = header->num_tensors;
        
        // Tensor descriptors follow header
        tensors_ = file_.as<TensorDesc>(sizeof(PackHeader));
        
        return true;
    }

    // Get raw pointer to tensor data (zero-copy)
    const void* tensor_data(size_t idx) const {
        if (idx >= num_tensors_) return nullptr;
        return file_.as<char>(tensors_[idx].offset);
    }

    const TensorDesc* tensor_desc(size_t idx) const {
        if (idx >= num_tensors_) return nullptr;
        return &tensors_[idx];
    }

    const TensorDesc* find_tensor(const char* name) const {
        for (size_t i = 0; i < num_tensors_; ++i) {
            if (std::strncmp(tensors_[i].name, name, 64) == 0) {
                return &tensors_[i];
            }
        }
        return nullptr;
    }

    // Model config accessors
    uint64_t hidden_size() const { return header_->hidden_size; }
    uint64_t num_layers() const { return header_->num_layers; }
    uint64_t num_heads() const { return header_->num_heads; }
    uint64_t num_kv_heads() const { return header_->num_kv_heads; }
    uint64_t intermediate_size() const { return header_->intermediate_size; }
    uint64_t vocab_size() const { return header_->vocab_size; }
    double rope_theta() const { return header_->rope_theta / 1e6; }
    uint64_t max_position() const { return header_->max_position; }
    size_t num_tensors() const { return num_tensors_; }

    bool is_loaded() const { return file_.is_open(); }

private:
    MmapFile file_;
    const PackHeader* header_ = nullptr;
    const TensorDesc* tensors_ = nullptr;
    size_t num_tensors_ = 0;
};

// ===================================================================
// Direct matmul on mmap'd quantized data (no intermediate copies)
// ===================================================================

// Q4_K fused matmul: reads directly from mmap'd pack
inline void q4k_matmul_mmap(
    const void* packed_data,  // Raw Q4_K blocks from mmap
    const float* x,           // Input vector
    float* y,                 // Output vector
    int rows,
    int cols
) {
    const uint8_t* data = static_cast<const uint8_t*>(packed_data);
    const int blocks_per_row = (cols + 255) / 256;
    
    #pragma omp parallel for if(rows > 16)
    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        const uint8_t* row_data = data + row * blocks_per_row * Q4_K_BLOCK_SIZE;
        
        for (int blk = 0; blk < blocks_per_row; ++blk) {
            const uint8_t* block = row_data + blk * Q4_K_BLOCK_SIZE;
            
            // Parse Q4_K block header
            uint16_t d_bits, m_bits;
            std::memcpy(&d_bits, block, 2);
            std::memcpy(&m_bits, block + 2, 2);
            
            // FP16 -> FP32 (portable, no F16C required)
            float d = fp16_to_fp32(d_bits);
            float m = fp16_to_fp32(m_bits);
            
            // Sub-block scales (6 bits each, packed in 12 bytes)
            const uint8_t* scales = block + 4;
            
            // Quantized values (4 bits each, 128 bytes for 256 values)
            const uint8_t* qs = block + 16;
            
            int col_base = blk * 256;
            
            // Process 256 values in this block
            for (int sb = 0; sb < 8; ++sb) {
                // Extract 6-bit scale
                int scale_idx = (sb * 6) / 8;
                int scale_shift = (sb * 6) % 8;
                uint8_t s = (scales[scale_idx] >> scale_shift) & 0x3F;
                if (scale_shift > 2 && scale_idx + 1 < 12) {
                    s |= (scales[scale_idx + 1] << (8 - scale_shift)) & 0x3F;
                }
                float sub_scale = d * s;
                
                // 32 values per sub-block
                for (int i = 0; i < 32; ++i) {
                    int val_idx = sb * 32 + i;
                    int byte_idx = val_idx / 2;
                    int nibble = (val_idx % 2 == 0) 
                        ? (qs[byte_idx] & 0x0F) 
                        : (qs[byte_idx] >> 4);
                    
                    float w = sub_scale * nibble + m;
                    int col = col_base + val_idx;
                    if (col < cols) {
                        sum += w * x[col];
                    }
                }
            }
        }
        y[row] = sum;
    }
}

// Q6_K fused matmul: reads directly from mmap'd pack
inline void q6k_matmul_mmap(
    const void* packed_data,
    const float* x,
    float* y,
    int rows,
    int cols
) {
    const uint8_t* data = static_cast<const uint8_t*>(packed_data);
    const int blocks_per_row = (cols + 255) / 256;
    
    #pragma omp parallel for if(rows > 16)
    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        const uint8_t* row_data = data + row * blocks_per_row * Q6_K_BLOCK_SIZE;
        
        for (int blk = 0; blk < blocks_per_row; ++blk) {
            const uint8_t* block = row_data + blk * Q6_K_BLOCK_SIZE;
            
            // Q6_K layout: 
            // - 128 bytes: low 4 bits of each value
            // - 64 bytes: high 2 bits of each value
            // - 16 bytes: sub-block scales
            // - 2 bytes: FP16 super-scale
            
            uint16_t d_bits;
            std::memcpy(&d_bits, block + 128 + 64 + 16, 2);
            float d = fp16_to_fp32(d_bits);
            
            const uint8_t* ql = block;           // Low 4 bits
            const uint8_t* qh = block + 128;     // High 2 bits
            const int8_t* scales = reinterpret_cast<const int8_t*>(block + 128 + 64);
            
            int col_base = blk * 256;
            
            for (int sb = 0; sb < 16; ++sb) {
                float sub_scale = d * scales[sb];
                
                for (int i = 0; i < 16; ++i) {
                    int val_idx = sb * 16 + i;
                    
                    // Assemble 6-bit value from low 4 + high 2
                    int byte_idx_l = val_idx / 2;
                    int nibble_l = (val_idx % 2 == 0)
                        ? (ql[byte_idx_l] & 0x0F)
                        : (ql[byte_idx_l] >> 4);
                    
                    int byte_idx_h = val_idx / 4;
                    int shift_h = (val_idx % 4) * 2;
                    int bits_h = (qh[byte_idx_h] >> shift_h) & 0x03;
                    
                    int q6 = nibble_l | (bits_h << 4);
                    float w = sub_scale * (q6 - 32);  // Centered at 32
                    
                    int col = col_base + val_idx;
                    if (col < cols) {
                        sum += w * x[col];
                    }
                }
            }
        }
        y[row] = sum;
    }
}

#ifdef __AVX2__
// AVX2 optimized Q4_K matmul with prefetch
inline void q4k_matmul_mmap_avx2(
    const void* packed_data,
    const float* x,
    float* y,
    int rows,
    int cols
) {
    const uint8_t* data = static_cast<const uint8_t*>(packed_data);
    const int blocks_per_row = (cols + 255) / 256;
    const size_t row_stride = blocks_per_row * Q4_K_BLOCK_SIZE;
    
    #pragma omp parallel for if(rows > 8)
    for (int row = 0; row < rows; ++row) {
        __m256 acc = _mm256_setzero_ps();
        const uint8_t* row_data = data + row * row_stride;
        
        // Prefetch next row
        if (row + 1 < rows) {
            _mm_prefetch(reinterpret_cast<const char*>(row_data + row_stride), _MM_HINT_T0);
        }
        
        for (int blk = 0; blk < blocks_per_row; ++blk) {
            const uint8_t* block = row_data + blk * Q4_K_BLOCK_SIZE;
            
            uint16_t d_bits, m_bits;
            std::memcpy(&d_bits, block, 2);
            std::memcpy(&m_bits, block + 2, 2);
            float d = fp16_to_fp32(d_bits);
            float m = fp16_to_fp32(m_bits);
            
            __m256 vd = _mm256_set1_ps(d);
            __m256 vm = _mm256_set1_ps(m);
            
            const uint8_t* scales = block + 4;
            const uint8_t* qs = block + 16;
            int col_base = blk * 256;
            
            // Process 8 values at a time with AVX2
            for (int sb = 0; sb < 8; ++sb) {
                int scale_idx = (sb * 6) / 8;
                int scale_shift = (sb * 6) % 8;
                uint8_t s = (scales[scale_idx] >> scale_shift) & 0x3F;
                if (scale_shift > 2 && scale_idx + 1 < 12) {
                    s |= (scales[scale_idx + 1] << (8 - scale_shift)) & 0x3F;
                }
                __m256 vs = _mm256_set1_ps(static_cast<float>(s));
                
                for (int i = 0; i < 32; i += 8) {
                    int val_idx = sb * 32 + i;
                    int col = col_base + val_idx;
                    if (col + 8 > cols) break;
                    
                    // Extract 8 nibbles
                    alignas(32) float weights[8];
                    for (int k = 0; k < 8; ++k) {
                        int vi = val_idx + k;
                        int byte_idx = vi / 2;
                        int nibble = (vi % 2 == 0)
                            ? (qs[byte_idx] & 0x0F)
                            : (qs[byte_idx] >> 4);
                        weights[k] = static_cast<float>(nibble);
                    }
                    
                    __m256 vq = _mm256_load_ps(weights);
                    __m256 vw = _mm256_fmadd_ps(_mm256_mul_ps(vd, vs), vq, vm);
                    __m256 vx = _mm256_loadu_ps(&x[col]);
                    acc = _mm256_fmadd_ps(vw, vx, acc);
                }
            }
        }
        
        // Horizontal sum
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        y[row] = _mm_cvtss_f32(lo);
    }
}
#endif

} // namespace mmap
} // namespace rftmw
