/*
 * Native Qwen3.5-MoE token runner for RFTMW/GGUF packs.
 *
 * This executable intentionally does not embed Python.  It reads the
 * .rftmwpk sidecar, mmaps the source GGUF blob, and runs the token-level
 * inference loop in C++ using rftmw_inference.hpp kernels.
 */

#include "rftmw_inference.hpp"
#include "rftmw_mmap.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

constexpr int GGUF_F32 = 0;
constexpr int GGUF_F16 = 1;
constexpr int GGUF_Q4_K = 12;
constexpr int GGUF_Q6_K = 14;

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> ne;
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    const uint8_t* data = nullptr;
};

struct NativeConfig {
    int hidden = 2048;
    int layers = 40;
    int vocab = 248320;
    int n_experts = 256;
    int experts_per_tok = 8;
    int expert_dim = 512;
    int full_attn_interval = 4;
    int attn_heads = 16;
    int attn_kv_heads = 2;
    int attn_head_dim = 256;
    int rope_dim = 64;
    double rope_theta = 10000000.0;
    float rms_eps = 1e-6f;
    int linear_k_heads = 16;
    int linear_v_heads = 32;
    int linear_k_dim = 128;
    int linear_v_dim = 128;
    int ssm_conv_k = 4;
};

uint64_t read_u64(const uint8_t*& p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    p += sizeof(v);
    return v;
}

uint32_t read_u32(const uint8_t*& p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    p += sizeof(v);
    return v;
}

float read_f32(const uint8_t*& p) {
    float v;
    std::memcpy(&v, p, sizeof(v));
    p += sizeof(v);
    return v;
}

double read_f64(const uint8_t*& p) {
    double v;
    std::memcpy(&v, p, sizeof(v));
    p += sizeof(v);
    return v;
}

std::string read_string(const uint8_t*& p) {
    uint64_t n = read_u64(p);
    std::string s(reinterpret_cast<const char*>(p), reinterpret_cast<const char*>(p + n));
    p += n;
    return s;
}

void skip_gguf_value(const uint8_t*& p, uint32_t type);

void skip_gguf_scalar(const uint8_t*& p, uint32_t type) {
    switch (type) {
    case 0:  p += 1; break;  // uint8
    case 1:  p += 1; break;  // int8
    case 2:  p += 2; break;  // uint16
    case 3:  p += 2; break;  // int16
    case 4:  p += 4; break;  // uint32
    case 5:  p += 4; break;  // int32
    case 6:  p += 4; break;  // float32
    case 7:  p += 1; break;  // bool
    case 8:  (void)read_string(p); break;
    case 10: p += 8; break;  // uint64
    case 11: p += 8; break;  // int64
    case 12: p += 8; break;  // float64
    default:
        throw std::runtime_error("unsupported GGUF metadata scalar type");
    }
}

void skip_gguf_value(const uint8_t*& p, uint32_t type) {
    if (type == 9) {
        uint32_t elem_type = read_u32(p);
        uint64_t n = read_u64(p);
        for (uint64_t i = 0; i < n; ++i) skip_gguf_value(p, elem_type);
        return;
    }
    skip_gguf_scalar(p, type);
}

uint64_t gguf_tensor_nbytes(uint32_t type, const std::vector<uint64_t>& ne) {
    uint64_t n = 1;
    for (uint64_t d : ne) n *= d;
    if (type == GGUF_F32) return n * 4;
    if (type == GGUF_F16) return n * 2;
    if (type == GGUF_Q4_K) return (n / 256) * 144;
    if (type == GGUF_Q6_K) return (n / 256) * 210;
    return 0;
}

std::vector<std::string> read_string_array_value(const uint8_t*& p) {
    uint32_t elem_type = read_u32(p);
    uint64_t n = read_u64(p);
    if (elem_type != 8) throw std::runtime_error("expected GGUF string array");
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; ++i) out.push_back(read_string(p));
    return out;
}

std::vector<int> read_int_array_value(const uint8_t*& p) {
    uint32_t elem_type = read_u32(p);
    uint64_t n = read_u64(p);
    std::vector<int> out;
    out.reserve(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; ++i) {
        if (elem_type == 4) {
            out.push_back(static_cast<int>(read_u32(p)));
        } else if (elem_type == 5) {
            int32_t v;
            std::memcpy(&v, p, sizeof(v));
            p += sizeof(v);
            out.push_back(static_cast<int>(v));
        } else if (elem_type == 10) {
            out.push_back(static_cast<int>(read_u64(p)));
        } else if (elem_type == 11) {
            int64_t v;
            std::memcpy(&v, p, sizeof(v));
            p += sizeof(v);
            out.push_back(static_cast<int>(v));
        } else {
            throw std::runtime_error("expected GGUF integer array");
        }
    }
    return out;
}

struct GGUFModel {
    rftmw::mmap::MmapFile file;
    std::unordered_map<std::string, TensorInfo> tensors;
    std::vector<std::string> tokenizer_tokens;
    std::vector<std::string> tokenizer_merges;
    std::vector<int> tokenizer_token_types;
    uint64_t data_start = 0;

    void load(const fs::path& path) {
        if (!file.open(path.string())) {
            throw std::runtime_error("failed to mmap GGUF: " + path.string());
        }
        const uint8_t* base = static_cast<const uint8_t*>(file.data());
        const uint8_t* p = base;
        if (std::memcmp(p, "GGUF", 4) != 0) {
            throw std::runtime_error("bad GGUF magic");
        }
        p += 4;
        uint32_t version = read_u32(p);
        (void)version;
        uint64_t n_tensors = read_u64(p);
        uint64_t n_kv = read_u64(p);

        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = read_string(p);
            uint32_t type = read_u32(p);
            if (key == "tokenizer.ggml.tokens" && type == 9) {
                tokenizer_tokens = read_string_array_value(p);
            } else if (key == "tokenizer.ggml.merges" && type == 9) {
                tokenizer_merges = read_string_array_value(p);
            } else if (key == "tokenizer.ggml.token_type" && type == 9) {
                tokenizer_token_types = read_int_array_value(p);
            } else {
                skip_gguf_value(p, type);
            }
        }

        std::vector<TensorInfo> tmp;
        tmp.reserve(static_cast<size_t>(n_tensors));
        for (uint64_t i = 0; i < n_tensors; ++i) {
            TensorInfo t;
            t.name = read_string(p);
            uint32_t n_dims = read_u32(p);
            t.ne.reserve(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) t.ne.push_back(read_u64(p));
            t.type = read_u32(p);
            t.offset = read_u64(p);
            t.bytes = gguf_tensor_nbytes(t.type, t.ne);
            tmp.push_back(std::move(t));
        }

        uint64_t pos = static_cast<uint64_t>(p - base);
        data_start = (pos + 31u) & ~31u;
        for (auto& t : tmp) {
            t.data = base + data_start + t.offset;
            tensors.emplace(t.name, std::move(t));
        }
    }

    const TensorInfo& tensor(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("missing tensor: " + name);
        return it->second;
    }
};

std::string utf8_from_codepoint(uint32_t cp) {
    std::string s;
    if (cp <= 0x7F) {
        s.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return s;
}

std::vector<uint32_t> utf8_to_codepoints(const std::string& s) {
    std::vector<uint32_t> cps;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            cps.push_back(c);
            ++i;
        } else if ((c >> 5) == 0x6 && i + 1 < s.size()) {
            cps.push_back(((c & 0x1F) << 6)
                          | (static_cast<unsigned char>(s[i + 1]) & 0x3F));
            i += 2;
        } else if ((c >> 4) == 0xE && i + 2 < s.size()) {
            cps.push_back(((c & 0x0F) << 12)
                          | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6)
                          | (static_cast<unsigned char>(s[i + 2]) & 0x3F));
            i += 3;
        } else if ((c >> 3) == 0x1E && i + 3 < s.size()) {
            cps.push_back(((c & 0x07) << 18)
                          | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12)
                          | ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6)
                          | (static_cast<unsigned char>(s[i + 3]) & 0x3F));
            i += 4;
        } else {
            cps.push_back(c);
            ++i;
        }
    }
    return cps;
}

struct NativeTokenizer {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<std::string, int> merge_rank;
    std::string byte_encoder[256];
    std::unordered_map<uint32_t, uint8_t> byte_decoder;
    std::vector<std::string> special_tokens;

    void build_byte_maps() {
        std::vector<int> bs;
        for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        std::unordered_set<int> used(bs.begin(), bs.end());
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!used.count(b)) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) {
            byte_encoder[bs[i]] = utf8_from_codepoint(static_cast<uint32_t>(cs[i]));
            byte_decoder[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
        }
    }

    void init(const GGUFModel& m) {
        id_to_token = m.tokenizer_tokens;
        if (id_to_token.empty()) throw std::runtime_error("GGUF has no tokenizer tokens");
        token_to_id.reserve(id_to_token.size());
        for (size_t i = 0; i < id_to_token.size(); ++i) {
            token_to_id[id_to_token[i]] = static_cast<int>(i);
            int ttype = (i < m.tokenizer_token_types.size()) ? m.tokenizer_token_types[i] : 1;
            if (ttype != 1) special_tokens.push_back(id_to_token[i]);
        }
        std::sort(special_tokens.begin(), special_tokens.end(),
                  [](const std::string& a, const std::string& b) {
                      return a.size() > b.size();
                  });
        build_byte_maps();
        merge_rank.reserve(m.tokenizer_merges.size());
        for (size_t i = 0; i < m.tokenizer_merges.size(); ++i) {
            const std::string& merge = m.tokenizer_merges[i];
            size_t sp = merge.find(' ');
            if (sp != std::string::npos) {
                std::string key = merge.substr(0, sp) + "\t" + merge.substr(sp + 1);
                merge_rank[key] = static_cast<int>(i);
            }
        }
    }

    static bool is_alpha(unsigned char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    }
    static bool is_digit(unsigned char c) {
        return c >= '0' && c <= '9';
    }
    static bool is_space_no_newline(unsigned char c) {
        return c == ' ' || c == '\t' || c == '\v' || c == '\f';
    }

    std::vector<std::string> pretokenize_ascii(const std::string& text) const {
        std::vector<std::string> parts;
        size_t i = 0;
        while (i < text.size()) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            if (c == '\r' || c == '\n') {
                size_t j = i + 1;
                while (j < text.size()
                       && (text[j] == '\r' || text[j] == '\n')) ++j;
                parts.push_back(text.substr(i, j - i));
                i = j;
            } else if (is_space_no_newline(c)) {
                if (i + 1 < text.size() && is_alpha(static_cast<unsigned char>(text[i + 1]))) {
                    size_t j = i + 1;
                    while (j < text.size() && is_alpha(static_cast<unsigned char>(text[j]))) ++j;
                    parts.push_back(text.substr(i, j - i));
                    i = j;
                } else if (i + 1 < text.size()
                           && !is_space_no_newline(static_cast<unsigned char>(text[i + 1]))
                           && text[i + 1] != '\r' && text[i + 1] != '\n'
                           && !is_alpha(static_cast<unsigned char>(text[i + 1]))
                           && !is_digit(static_cast<unsigned char>(text[i + 1]))) {
                    size_t j = i + 1;
                    while (j < text.size()) {
                        unsigned char d = static_cast<unsigned char>(text[j]);
                        if (is_space_no_newline(d) || d == '\r' || d == '\n'
                            || is_alpha(d) || is_digit(d)) break;
                        ++j;
                    }
                    parts.push_back(text.substr(i, j - i));
                    i = j;
                } else {
                    size_t j = i + 1;
                    while (j < text.size() && is_space_no_newline(static_cast<unsigned char>(text[j]))) ++j;
                    parts.push_back(text.substr(i, j - i));
                    i = j;
                }
            } else if (is_alpha(c)) {
                size_t j = i + 1;
                while (j < text.size() && is_alpha(static_cast<unsigned char>(text[j]))) ++j;
                parts.push_back(text.substr(i, j - i));
                i = j;
            } else if (is_digit(c)) {
                parts.push_back(text.substr(i, 1));
                ++i;
            } else {
                size_t j = i + 1;
                while (j < text.size()) {
                    unsigned char d = static_cast<unsigned char>(text[j]);
                    if (is_space_no_newline(d) || d == '\r' || d == '\n'
                        || is_alpha(d) || is_digit(d)) break;
                    ++j;
                }
                parts.push_back(text.substr(i, j - i));
                i = j;
            }
        }
        return parts;
    }

    std::vector<std::string> byte_symbols(const std::string& part) const {
        std::vector<std::string> out;
        out.reserve(part.size());
        for (unsigned char c : part) out.push_back(byte_encoder[c]);
        return out;
    }

    std::vector<std::string> bpe(const std::string& part) const {
        std::vector<std::string> word = byte_symbols(part);
        if (word.size() <= 1) return word;
        while (word.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            size_t best_pos = word.size();
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                auto it = merge_rank.find(word[i] + "\t" + word[i + 1]);
                if (it != merge_rank.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pos = i;
                }
            }
            if (best_pos == word.size()) break;
            std::vector<std::string> merged;
            merged.reserve(word.size() - 1);
            for (size_t i = 0; i < word.size();) {
                if (i + 1 < word.size()
                    && word[i] == word[best_pos]
                    && word[i + 1] == word[best_pos + 1]) {
                    merged.push_back(word[i] + word[i + 1]);
                    i += 2;
                } else {
                    merged.push_back(word[i]);
                    ++i;
                }
            }
            word.swap(merged);
        }
        return word;
    }

    bool match_special(const std::string& text, size_t pos, std::string& special) const {
        for (const auto& tok : special_tokens) {
            if (tok.empty()) continue;
            if (pos + tok.size() <= text.size()
                && text.compare(pos, tok.size(), tok) == 0) {
                special = tok;
                return true;
            }
        }
        return false;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        std::string normal;
        auto flush_normal = [&]() {
            if (normal.empty()) return;
            for (const auto& part : pretokenize_ascii(normal)) {
                for (const auto& piece : bpe(part)) {
                    auto it = token_to_id.find(piece);
                    if (it != token_to_id.end()) ids.push_back(it->second);
                    else throw std::runtime_error("tokenizer missing BPE piece: " + piece);
                }
            }
            normal.clear();
        };

        for (size_t i = 0; i < text.size();) {
            std::string special;
            if (match_special(text, i, special)) {
                flush_normal();
                ids.push_back(token_to_id.at(special));
                i += special.size();
            } else {
                normal.push_back(text[i]);
                ++i;
            }
        }
        flush_normal();
        return ids;
    }

    std::string decode_token(int id) const {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token.size()) return "";
        const std::string& tok = id_to_token[static_cast<size_t>(id)];
        std::string out;
        for (uint32_t cp : utf8_to_codepoints(tok)) {
            auto it = byte_decoder.find(cp);
            if (it != byte_decoder.end()) {
                out.push_back(static_cast<char>(it->second));
            } else {
                out += utf8_from_codepoint(cp);
            }
        }
        return out;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        for (int id : ids) out += decode_token(id);
        return out;
    }
};

std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to read: " + path.string());
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string regex_group(const std::string& text, const std::string& pat) {
    std::smatch m;
    if (std::regex_search(text, m, std::regex(pat))) return m[1].str();
    return {};
}

fs::path resolve_gguf_from_pack(const fs::path& pack) {
    fs::path sidecar = pack;
    sidecar.replace_extension(".rftmwpk.meta.json");
    if (!fs::exists(sidecar)) {
        sidecar = pack.parent_path() / (pack.stem().string() + ".rftmwpk.meta.json");
    }
    std::string meta = read_text(sidecar);
    std::string gguf_path = regex_group(meta, R"json("gguf_path"\s*:\s*"([^"]+)")json");
    std::string sha = regex_group(meta, R"json("gguf_file_sha256"\s*:\s*"([^"]+)")json");
    if (!gguf_path.empty() && fs::exists(fs::path(gguf_path))) return fs::path(gguf_path);
    if (!sha.empty()) {
#ifdef _WIN32
        const char* user = std::getenv("USERPROFILE");
        fs::path home = user ? fs::path(user) : fs::path();
#else
        const char* user = std::getenv("HOME");
        fs::path home = user ? fs::path(user) : fs::path();
#endif
        fs::path candidate = home / ".ollama" / "models" / "blobs" / ("sha256-" + sha);
        if (fs::exists(candidate)) return candidate;
    }
    throw std::runtime_error("could not resolve GGUF source from sidecar");
}

float fp16_to_float(uint16_t h) {
    return rftmw::inference::fp16_to_f32(h);
}

std::vector<float> tensor_to_f32(const TensorInfo& t) {
    uint64_t n = 1;
    for (uint64_t d : t.ne) n *= d;
    std::vector<float> out(static_cast<size_t>(n));
    if (t.type == GGUF_F32) {
        std::memcpy(out.data(), t.data, static_cast<size_t>(n) * sizeof(float));
    } else if (t.type == GGUF_F16) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(t.data);
        for (uint64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = fp16_to_float(p[i]);
    } else {
        throw std::runtime_error("tensor is not dense f32/f16: " + t.name);
    }
    return out;
}

struct Weight2D {
    const TensorInfo* t = nullptr;
    int in = 0;
    int out = 0;
    int bytes_per_row = 0;
    std::vector<float> dense;

    void bind(const GGUFModel& m, const std::string& name) {
        t = &m.tensor(name);
        if (t->ne.size() < 2) throw std::runtime_error("2D tensor rank error: " + name);
        in = static_cast<int>(t->ne[0]);
        out = static_cast<int>(t->ne[1]);
        if (t->type == GGUF_Q4_K) bytes_per_row = (in / 256) * 144;
        else if (t->type == GGUF_Q6_K) bytes_per_row = (in / 256) * 210;
        else dense = tensor_to_f32(*t);
    }

    void override_dense(std::vector<float>&& data, int rows, int cols, const std::string& label) {
        if (rows != out || cols != in) {
            std::ostringstream oss;
            oss << "fine-tune shape mismatch for " << label
                << ": got (" << rows << "," << cols << ") expected ("
                << out << "," << in << ")";
            throw std::runtime_error(oss.str());
        }
        dense = std::move(data);
    }

    std::vector<float> matmul(const std::vector<float>& x) const {
        std::vector<float> y(static_cast<size_t>(out));
        if (!dense.empty()) {
            rftmw::inference::dense_sgemv_f32(dense.data(), x.data(), y.data(), out, in);
        } else {
            const char* qt = (t->type == GGUF_Q4_K) ? "q4k" : "q6k";
            rftmw::inference::quant_matmul_vec(t->data, x.data(), y.data(), out, bytes_per_row, qt);
        }
        return y;
    }

    void dequant_row(int row, std::vector<float>& y) const {
        y.assign(static_cast<size_t>(in), 0.0f);
        if (!dense.empty()) {
            std::memcpy(y.data(), dense.data() + static_cast<size_t>(row) * in,
                        static_cast<size_t>(in) * sizeof(float));
            return;
        }
        const uint8_t* row_data = t->data + static_cast<size_t>(row) * bytes_per_row;
        int nblocks = (t->type == GGUF_Q4_K) ? bytes_per_row / 144 : bytes_per_row / 210;
        for (int b = 0; b < nblocks; ++b) {
            if (t->type == GGUF_Q4_K) {
                rftmw::inference::dequant_q4k_block(row_data + static_cast<size_t>(b) * 144,
                                                    y.data() + static_cast<size_t>(b) * 256);
            } else {
                rftmw::inference::dequant_q6k_block(row_data + static_cast<size_t>(b) * 210,
                                                    y.data() + static_cast<size_t>(b) * 256);
            }
        }
    }
};

struct Weight3D {
    const TensorInfo* t = nullptr;
    int n_experts = 0;
    int in = 0;
    int out = 0;
    int bytes_per_row = 0;

    void bind(const GGUFModel& m, const std::string& name) {
        t = &m.tensor(name);
        if (t->ne.size() < 3) throw std::runtime_error("3D tensor rank error: " + name);
        in = static_cast<int>(t->ne[0]);
        out = static_cast<int>(t->ne[1]);
        n_experts = static_cast<int>(t->ne[2]);
        if (t->type == GGUF_Q4_K) bytes_per_row = (in / 256) * 144;
        else if (t->type == GGUF_Q6_K) bytes_per_row = (in / 256) * 210;
        else throw std::runtime_error("3D expert tensor must be quantized: " + name);
    }

    std::vector<float> matmul_expert(const std::vector<float>& x, int expert) const {
        std::vector<float> y(static_cast<size_t>(out));
        const size_t expert_stride = static_cast<size_t>(out) * bytes_per_row;
        const uint8_t* raw = t->data + static_cast<size_t>(expert) * expert_stride;
        const char* qt = (t->type == GGUF_Q4_K) ? "q4k" : "q6k";
        rftmw::inference::quant_matmul_vec(raw, x.data(), y.data(), out, bytes_per_row, qt);
        return y;
    }
};

float dot(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

void add_inplace(std::vector<float>& a, const std::vector<float>& b, float scale = 1.0f) {
    for (size_t i = 0; i < a.size(); ++i) a[i] += scale * b[i];
}

float sigmoid(float x) {
    x = std::max(-88.0f, std::min(88.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}

float silu(float x) {
    x = std::max(-88.0f, std::min(88.0f, x));
    return x / (1.0f + std::exp(-x));
}

std::vector<float> rms_norm_vec(const std::vector<float>& x,
                                const std::vector<float>& w,
                                float eps) {
    std::vector<float> y(x.size());
    rftmw::inference::rms_norm(x.data(), w.data(), y.data(), static_cast<int>(x.size()), eps);
    return y;
}

void per_head_rms(float* x, int heads, int dim, const std::vector<float>& w, float eps) {
    for (int h = 0; h < heads; ++h) {
        float* row = x + static_cast<size_t>(h) * dim;
        float ss = 0.0f;
        for (int i = 0; i < dim; ++i) ss += row[i] * row[i];
        float inv = 1.0f / std::sqrt(ss / static_cast<float>(dim) + eps);
        for (int i = 0; i < dim; ++i) row[i] *= inv * w[static_cast<size_t>(i)];
    }
}

std::vector<float> softmax_selected(const std::vector<float>& logits,
                                    const std::vector<int>& idx) {
    float m = -std::numeric_limits<float>::infinity();
    for (int i : idx) m = std::max(m, logits[static_cast<size_t>(i)]);
    std::vector<float> w(idx.size());
    float s = 0.0f;
    for (size_t j = 0; j < idx.size(); ++j) {
        w[j] = std::exp(logits[static_cast<size_t>(idx[j])] - m);
        s += w[j];
    }
    for (float& v : w) v /= s;
    return w;
}

std::vector<int> topk_indices(const std::vector<float>& logits, int k) {
    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    idx.resize(k);
    return idx;
}

struct MoE {
    Weight2D router;
    Weight3D gate_exps;
    Weight3D up_exps;
    Weight3D down_exps;
    Weight2D shared_gate;
    Weight2D shared_up;
    Weight2D shared_down;
    std::vector<float> shared_gate_inp;

    std::vector<float> forward(const std::vector<float>& x, const NativeConfig& cfg) const {
        std::vector<float> logits = router.matmul(x);
        std::vector<int> idx = topk_indices(logits, cfg.experts_per_tok);
        std::vector<float> weights = softmax_selected(logits, idx);
        std::vector<float> out(static_cast<size_t>(cfg.hidden), 0.0f);
        std::vector<std::vector<float>> expert_out(idx.size());

        auto compute_expert = [&](size_t j) {
            std::vector<float> gate = gate_exps.matmul_expert(x, idx[j]);
            std::vector<float> up = up_exps.matmul_expert(x, idx[j]);
            std::vector<float> hidden(gate.size());
            for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = silu(gate[i]) * up[i];
            expert_out[j] = down_exps.matmul_expert(hidden, idx[j]);
        };

        std::vector<std::thread> expert_threads;
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        if (idx.size() > 1 && hw > 2) {
            expert_threads.reserve(idx.size() - 1);
            for (size_t j = 1; j < idx.size(); ++j) {
                expert_threads.emplace_back([&, j]() { compute_expert(j); });
            }
            compute_expert(0);
        } else {
            for (size_t j = 0; j < idx.size(); ++j) compute_expert(j);
        }

        std::vector<float> sg = shared_gate.matmul(x);
        std::vector<float> su = shared_up.matmul(x);
        std::vector<float> sh(sg.size());
        for (size_t i = 0; i < sh.size(); ++i) sh[i] = silu(sg[i]) * su[i];
        std::vector<float> so = shared_down.matmul(sh);
        if (!shared_gate_inp.empty()) {
            float g = sigmoid(dot(shared_gate_inp, x));
            for (float& v : so) v *= g;
        }
        for (auto& th : expert_threads) th.join();
        for (size_t j = 0; j < idx.size(); ++j) add_inplace(out, expert_out[j], weights[j]);
        add_inplace(out, so);
        return out;
    }
};

struct Layer {
    bool full_attention = false;
    std::vector<float> attn_norm;
    std::vector<float> post_norm;

    Weight2D qkv_proj;
    Weight2D attn_gate;
    Weight2D ssm_out;
    Weight2D ssm_alpha;
    Weight2D ssm_beta;
    Weight2D ssm_conv;
    std::vector<float> ssm_a;
    std::vector<float> ssm_dt_bias;
    std::vector<float> ssm_norm;
    std::vector<float> conv_kernel;
    std::vector<float> conv_state;
    std::vector<float> recurrent_state;

    Weight2D q_proj;
    Weight2D k_proj;
    Weight2D v_proj;
    Weight2D o_proj;
    std::vector<float> q_norm;
    std::vector<float> k_norm;
    std::vector<float> k_cache;
    std::vector<float> v_cache;

    MoE moe;

    void bind(const GGUFModel& m, int i, const NativeConfig& cfg, int max_seq) {
        std::string p = "blk." + std::to_string(i) + ".";
        full_attention = ((i + 1) % cfg.full_attn_interval) == 0;
        attn_norm = tensor_to_f32(m.tensor(p + "attn_norm.weight"));
        post_norm = tensor_to_f32(m.tensor(p + "post_attention_norm.weight"));

        if (!full_attention) {
            qkv_proj.bind(m, p + "attn_qkv.weight");
            attn_gate.bind(m, p + "attn_gate.weight");
            ssm_out.bind(m, p + "ssm_out.weight");
            ssm_alpha.bind(m, p + "ssm_alpha.weight");
            ssm_beta.bind(m, p + "ssm_beta.weight");
            ssm_conv.bind(m, p + "ssm_conv1d.weight");
            ssm_a = tensor_to_f32(m.tensor(p + "ssm_a"));
            ssm_dt_bias = tensor_to_f32(m.tensor(p + "ssm_dt.bias"));
            ssm_norm = tensor_to_f32(m.tensor(p + "ssm_norm.weight"));
            conv_kernel = tensor_to_f32(*ssm_conv.t);
            conv_state.assign(static_cast<size_t>(cfg.linear_k_heads * cfg.linear_k_dim * 2
                              + cfg.linear_v_heads * cfg.linear_v_dim) * cfg.ssm_conv_k, 0.0f);
            recurrent_state.assign(static_cast<size_t>(cfg.linear_v_heads)
                                   * cfg.linear_k_dim * cfg.linear_v_dim, 0.0f);
        } else {
            q_proj.bind(m, p + "attn_q.weight");
            k_proj.bind(m, p + "attn_k.weight");
            v_proj.bind(m, p + "attn_v.weight");
            o_proj.bind(m, p + "attn_output.weight");
            q_norm = tensor_to_f32(m.tensor(p + "attn_q_norm.weight"));
            k_norm = tensor_to_f32(m.tensor(p + "attn_k_norm.weight"));
            k_cache.assign(static_cast<size_t>(max_seq) * cfg.attn_kv_heads * cfg.attn_head_dim, 0.0f);
            v_cache.assign(static_cast<size_t>(max_seq) * cfg.attn_kv_heads * cfg.attn_head_dim, 0.0f);
        }

        moe.router.bind(m, p + "ffn_gate_inp.weight");
        moe.gate_exps.bind(m, p + "ffn_gate_exps.weight");
        moe.up_exps.bind(m, p + "ffn_up_exps.weight");
        moe.down_exps.bind(m, p + "ffn_down_exps.weight");
        moe.shared_gate.bind(m, p + "ffn_gate_shexp.weight");
        moe.shared_up.bind(m, p + "ffn_up_shexp.weight");
        moe.shared_down.bind(m, p + "ffn_down_shexp.weight");
        moe.shared_gate_inp = tensor_to_f32(m.tensor(p + "ffn_gate_inp_shexp.weight"));
    }

    std::vector<float> linear_attention(const std::vector<float>& normed,
                                        const NativeConfig& cfg,
                                        int cache_len) {
        std::vector<float> qkv = qkv_proj.matmul(normed);
        std::vector<float> z = attn_gate.matmul(normed);
        std::vector<float> a = ssm_alpha.matmul(normed);
        std::vector<float> b = ssm_beta.matmul(normed);
        int conv_dim = cfg.linear_k_heads * cfg.linear_k_dim * 2
                     + cfg.linear_v_heads * cfg.linear_v_dim;
        if (cache_len == 0) {
            std::fill(conv_state.begin(), conv_state.end(), 0.0f);
            std::fill(recurrent_state.begin(), recurrent_state.end(), 0.0f);
        }
        std::vector<float> y(static_cast<size_t>(cfg.linear_v_heads * cfg.linear_v_dim));
        rftmw::inference::deltanet_step(
            qkv.data(), z.data(), a.data(), b.data(),
            conv_kernel.data(), ssm_a.data(), ssm_dt_bias.data(),
            ssm_norm.empty() ? nullptr : ssm_norm.data(), cfg.rms_eps,
            conv_state.data(), recurrent_state.data(),
            cfg.linear_k_heads, cfg.linear_v_heads,
            cfg.linear_k_dim, cfg.linear_v_dim, cfg.ssm_conv_k,
            y.data());
        return ssm_out.matmul(y);
    }

    static void apply_partial_rope(std::vector<float>& x, int heads, int hd,
                                   int rope_dim, int pos, double theta) {
        int half = rope_dim / 2;
        for (int i = 0; i < half; ++i) {
            double inv_freq = 1.0 / std::pow(theta, static_cast<double>(2 * i) / rope_dim);
            float c = static_cast<float>(std::cos(pos * inv_freq));
            float s = static_cast<float>(std::sin(pos * inv_freq));
            for (int h = 0; h < heads; ++h) {
                float* row = x.data() + static_cast<size_t>(h) * hd;
                float x1 = row[i];
                float x2 = row[half + i];
                row[i] = x1 * c - x2 * s;
                row[half + i] = x2 * c + x1 * s;
            }
        }
    }

    std::vector<float> full_attn_forward(const std::vector<float>& normed,
                                         const NativeConfig& cfg,
                                         int pos,
                                         int cache_len) {
        const int nh = cfg.attn_heads;
        const int nkv = cfg.attn_kv_heads;
        const int hd = cfg.attn_head_dim;
        std::vector<float> q_full = q_proj.matmul(normed);
        std::vector<float> query(static_cast<size_t>(nh * hd));
        std::vector<float> gate(static_cast<size_t>(nh * hd));
        for (int h = 0; h < nh; ++h) {
            std::memcpy(query.data() + static_cast<size_t>(h) * hd,
                        q_full.data() + static_cast<size_t>(h) * hd * 2,
                        static_cast<size_t>(hd) * sizeof(float));
            std::memcpy(gate.data() + static_cast<size_t>(h) * hd,
                        q_full.data() + static_cast<size_t>(h) * hd * 2 + hd,
                        static_cast<size_t>(hd) * sizeof(float));
        }
        std::vector<float> k = k_proj.matmul(normed);
        std::vector<float> v = v_proj.matmul(normed);
        per_head_rms(query.data(), nh, hd, q_norm, cfg.rms_eps);
        per_head_rms(k.data(), nkv, hd, k_norm, cfg.rms_eps);
        apply_partial_rope(query, nh, hd, cfg.rope_dim, pos, cfg.rope_theta);
        apply_partial_rope(k, nkv, hd, cfg.rope_dim, pos, cfg.rope_theta);

        std::memcpy(k_cache.data() + static_cast<size_t>(cache_len) * nkv * hd,
                    k.data(), static_cast<size_t>(nkv * hd) * sizeof(float));
        std::memcpy(v_cache.data() + static_cast<size_t>(cache_len) * nkv * hd,
                    v.data(), static_cast<size_t>(nkv * hd) * sizeof(float));

        std::vector<float> out(static_cast<size_t>(nh * hd), 0.0f);
        int hpg = nh / nkv;
        float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        int seq_len = cache_len + 1;
        std::vector<float> scores(static_cast<size_t>(seq_len));
        std::vector<float> probs(static_cast<size_t>(seq_len));
        for (int kv = 0; kv < nkv; ++kv) {
            for (int ho = 0; ho < hpg; ++ho) {
                int h = kv * hpg + ho;
                const float* qh = query.data() + static_cast<size_t>(h) * hd;
                for (int t = 0; t < seq_len; ++t) {
                    const float* kh = k_cache.data() + (static_cast<size_t>(t) * nkv + kv) * hd;
                    float s = 0.0f;
                    for (int d = 0; d < hd; ++d) s += qh[d] * kh[d];
                    scores[static_cast<size_t>(t)] = s * scale;
                }
                float m = *std::max_element(scores.begin(), scores.end());
                float sum = 0.0f;
                for (int t = 0; t < seq_len; ++t) {
                    probs[static_cast<size_t>(t)] = std::exp(scores[static_cast<size_t>(t)] - m);
                    sum += probs[static_cast<size_t>(t)];
                }
                float* oh = out.data() + static_cast<size_t>(h) * hd;
                for (int t = 0; t < seq_len; ++t) {
                    float w = probs[static_cast<size_t>(t)] / sum;
                    const float* vh = v_cache.data() + (static_cast<size_t>(t) * nkv + kv) * hd;
                    for (int d = 0; d < hd; ++d) oh[d] += w * vh[d];
                }
            }
        }
        for (size_t i = 0; i < out.size(); ++i) out[i] *= sigmoid(gate[i]);
        return o_proj.matmul(out);
    }

    std::vector<float> forward(const std::vector<float>& x,
                               const NativeConfig& cfg,
                               int pos,
                               int cache_len) {
        std::vector<float> h = x;
        std::vector<float> normed = rms_norm_vec(h, attn_norm, cfg.rms_eps);
        std::vector<float> o = full_attention
            ? full_attn_forward(normed, cfg, pos, cache_len)
            : linear_attention(normed, cfg, cache_len);
        add_inplace(h, o);
        normed = rms_norm_vec(h, post_norm, cfg.rms_eps);
        std::vector<float> mo = moe.forward(normed, cfg);
        add_inplace(h, mo);
        return h;
    }
};

struct NativeEngine {
    NativeConfig cfg;
    GGUFModel gguf;
    NativeTokenizer tokenizer;
    Weight2D embed;
    Weight2D lm_head;
    std::vector<float> output_norm;
    std::vector<Layer> layers;

    void load(const fs::path& pack, int max_seq) {
        fs::path gguf_path = resolve_gguf_from_pack(pack);
        std::cerr << "[native] GGUF " << gguf_path.string() << "\n";
        gguf.load(gguf_path);
        tokenizer.init(gguf);
        embed.bind(gguf, "token_embd.weight");
        lm_head.bind(gguf, "output.weight");
        output_norm = tensor_to_f32(gguf.tensor("output_norm.weight"));
        layers.resize(static_cast<size_t>(cfg.layers));
        for (int i = 0; i < cfg.layers; ++i) {
            layers[static_cast<size_t>(i)].bind(gguf, i, cfg, max_seq);
            if ((i + 1) % 5 == 0 || i + 1 == cfg.layers) {
                std::cerr << "[native] loaded layer " << (i + 1) << "/" << cfg.layers << "\r";
            }
        }
        std::cerr << "\n[native] all layers loaded\n";
        apply_finetune_sidecar(pack);
    }

    void apply_finetune_sidecar(const fs::path& pack) {
        fs::path sidecar(pack.string() + ".finetuned_weights.rftft");
        if (!fs::exists(sidecar)) return;

        std::ifstream in(sidecar, std::ios::binary);
        if (!in) throw std::runtime_error("failed to open fine-tune sidecar: " + sidecar.string());

        auto read_bytes = [&](void* dst, size_t bytes, const std::string& what) {
            if (!in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes))) {
                throw std::runtime_error("truncated fine-tune sidecar while reading " + what);
            }
        };
        auto read_u32_local = [&](const std::string& what) {
            uint32_t v = 0;
            read_bytes(&v, sizeof(v), what);
            return v;
        };

        char magic[8] = {};
        read_bytes(magic, sizeof(magic), "magic");
        if (std::memcmp(magic, "RFTFTW1", 7) != 0) {
            throw std::runtime_error("unsupported fine-tune sidecar format: " + sidecar.string());
        }

        uint32_t count = read_u32_local("matrix count");
        int applied = 0;
        int skipped = 0;
        std::regex name_re(R"(layer_(\d+)_(q|k|v|o)_proj)");
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t name_len = read_u32_local("name length");
            if (name_len == 0 || name_len > 256) throw std::runtime_error("invalid fine-tune matrix name length");
            std::string name(name_len, '\0');
            read_bytes(name.data(), name.size(), "matrix name");
            uint32_t rows = read_u32_local("rows");
            uint32_t cols = read_u32_local("cols");
            uint64_t elems = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
            if (elems == 0 || elems > (1ull << 34)) throw std::runtime_error("invalid fine-tune matrix size: " + name);
            std::vector<float> data(static_cast<size_t>(elems));
            read_bytes(data.data(), static_cast<size_t>(elems) * sizeof(float), "matrix data");

            std::smatch m;
            if (!std::regex_match(name, m, name_re)) {
                ++skipped;
                continue;
            }
            int layer_idx = std::stoi(m[1].str());
            std::string proj = m[2].str();
            if (layer_idx < 0 || layer_idx >= cfg.layers) {
                ++skipped;
                continue;
            }
            Layer& layer = layers[static_cast<size_t>(layer_idx)];
            if (!layer.full_attention) {
                ++skipped;
                continue;
            }
            Weight2D* target = nullptr;
            if (proj == "q") target = &layer.q_proj;
            else if (proj == "k") target = &layer.k_proj;
            else if (proj == "v") target = &layer.v_proj;
            else if (proj == "o") target = &layer.o_proj;
            if (!target) {
                ++skipped;
                continue;
            }
            target->override_dense(std::move(data), static_cast<int>(rows), static_cast<int>(cols), name);
            ++applied;
        }
        std::cerr << "[native] applied fine-tune sidecar " << sidecar.string()
                  << " (" << applied << " matrices, " << skipped << " skipped)\n";
    }

    std::vector<float> forward_token(int token, int pos) {
        std::vector<float> h;
        embed.dequant_row(token, h);
        for (int i = 0; i < cfg.layers; ++i) {
            h = layers[static_cast<size_t>(i)].forward(h, cfg, pos, pos);
        }
        h = rms_norm_vec(h, output_norm, cfg.rms_eps);
        return lm_head.matmul(h);
    }
};

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> ids;
    size_t start = 0;
    while (start < s.size()) {
        size_t comma = s.find(',', start);
        std::string part = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!part.empty()) ids.push_back(std::stoi(part));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return ids;
}

std::string format_prompt(const std::string& prompt, const std::string& mode) {
    if (mode == "raw" || mode == "none" || mode == "plain") return prompt;
    if (mode == "final_answer" || mode == "answer_only" || mode == "auto") {
        return "Return only the final answer. Do not include reasoning.\n"
               "Question: " + prompt + "\n"
               "Final answer:";
    }
    if (mode == "ollama" || mode == "qwen35_chat" || mode == "chat") {
        return "<|im_start|>system\n"
               "You are a concise assistant. Answer directly.\n"
               "<|im_end|>\n"
               "<|im_start|>user\n" + prompt + "\n"
               "<|im_end|>\n"
               "<|im_start|>assistant\n";
    }
    if (mode == "qwen35_think" || mode == "think") {
        return "<|im_start|>user\n" + prompt + "\n"
               "<|im_end|>\n"
               "<|im_start|>assistant\n"
               "<think>\n";
    }
    throw std::runtime_error("unknown template mode: " + mode);
}

int argmax(const std::vector<float>& x) {
    return static_cast<int>(std::max_element(x.begin(), x.end()) - x.begin());
}

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::string hex_encode(const std::string& s) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(s.size() * 2);
    for (unsigned char c : s) {
        out.push_back(kHex[(c >> 4) & 0x0f]);
        out.push_back(kHex[c & 0x0f]);
    }
    return out;
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    throw std::runtime_error("invalid hex input");
}

std::string hex_decode(const std::string& h) {
    if ((h.size() & 1u) != 0) throw std::runtime_error("odd-length hex input");
    std::string out;
    out.reserve(h.size() / 2);
    for (size_t i = 0; i < h.size(); i += 2) {
        int hi = hex_value(h[i]);
        int lo = hex_value(h[i + 1]);
        out.push_back(static_cast<char>((hi << 4) | lo));
    }
    return out;
}

std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= line.size()) {
        size_t tab = line.find('\t', start);
        if (tab == std::string::npos) {
            parts.push_back(line.substr(start));
            break;
        }
        parts.push_back(line.substr(start, tab - start));
        start = tab + 1;
    }
    return parts;
}

struct GenerationResult {
    std::vector<int> prompt_ids;
    std::vector<int> out_ids;
    std::string out_text;
    double prefill_s = 0.0;
    double decode_s = 0.0;
};

using StageCallback = std::function<void(const std::string&, int)>;
using TokenCallback = std::function<void(int, const std::string&, int, double)>;

GenerationResult generate_once(NativeEngine& engine,
                               const std::string& prompt,
                               const std::string& template_mode,
                               int max_tokens,
                               int max_seq,
                               StageCallback on_stage,
                               TokenCallback on_token,
                               const std::vector<int>* explicit_ids = nullptr) {
    if (max_tokens < 0) throw std::runtime_error("max_tokens must be >= 0");

    GenerationResult result;
    if (explicit_ids) {
        result.prompt_ids = *explicit_ids;
    } else {
        std::string formatted = format_prompt(prompt, template_mode);
        result.prompt_ids = engine.tokenizer.encode(formatted);
    }
    if (on_stage) on_stage("encoded", static_cast<int>(result.prompt_ids.size()));
    if (static_cast<int>(result.prompt_ids.size()) + max_tokens + 4 > max_seq) {
        throw std::runtime_error("request exceeds native max_seq; reduce history or max tokens");
    }

    std::vector<float> logits;
    auto prefill0 = Clock::now();
    for (size_t i = 0; i < result.prompt_ids.size(); ++i) {
        logits = engine.forward_token(result.prompt_ids[i], static_cast<int>(i));
    }
    result.prefill_s = seconds_since(prefill0);

    if (on_stage) on_stage("decode_start", static_cast<int>(result.prompt_ids.size()));
    auto decode0 = Clock::now();
    for (int i = 0; i < max_tokens; ++i) {
        int next = argmax(logits);
        if (next == 248044 || next == 248046) break;
        result.out_ids.push_back(next);
        std::string token_text = engine.tokenizer.decode_token(next);
        double tok_s = seconds_since(decode0);
        double rate = tok_s > 0.0 ? static_cast<double>(result.out_ids.size()) / tok_s : 0.0;
        if (on_token) on_token(next, token_text, static_cast<int>(result.out_ids.size()), rate);
        logits = engine.forward_token(next, static_cast<int>(result.prompt_ids.size() + result.out_ids.size() - 1));
    }
    result.decode_s = seconds_since(decode0);
    result.out_text = engine.tokenizer.decode(result.out_ids);
    return result;
}

void run_server(NativeEngine& engine, int max_seq) {
    std::cout << "READY\n" << std::flush;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT" || line == "EXIT") {
            std::cout << "BYE\n" << std::flush;
            break;
        }
        try {
            std::vector<std::string> parts = split_tabs(line);
            if (parts.empty() || parts[0] != "GENERATE" || parts.size() < 4) {
                throw std::runtime_error("expected GENERATE\\tmax_tokens\\ttemplate_mode\\thex_prompt");
            }
            int max_tokens = std::stoi(parts[1]);
            std::string template_mode = parts[2];
            std::string prompt = hex_decode(parts[3]);

            GenerationResult r = generate_once(
                engine,
                prompt,
                template_mode,
                max_tokens,
                max_seq,
                [](const std::string& stage, int value) {
                    std::cout << "STAGE\t" << stage << "\t" << value << "\n" << std::flush;
                },
                [](int token_id, const std::string& token_text, int n_output, double tok_per_s) {
                    std::cout << "TOKEN\t" << token_id << "\t" << hex_encode(token_text)
                              << "\t" << n_output << "\t" << tok_per_s << "\n" << std::flush;
                });

            std::cout << "DONE\t" << hex_encode(r.out_text)
                      << "\t" << r.prompt_ids.size()
                      << "\t" << r.prefill_s
                      << "\t" << r.out_ids.size()
                      << "\t" << r.decode_s << "\n" << std::flush;
        } catch (const std::exception& e) {
            std::cout << "ERROR\t" << hex_encode(e.what()) << "\n" << std::flush;
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        fs::path pack = "qwen35-opus-distil.rftmwpk";
        std::string ids_arg;
        std::string prompt_arg;
        std::string template_mode = "final_answer";
        int max_tokens = 2;
        int max_seq_arg = 512;
        bool serve = false;
        bool print_prompt_ids = false;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--pack" && i + 1 < argc) pack = argv[++i];
            else if (a == "--ids" && i + 1 < argc) ids_arg = argv[++i];
            else if (a == "--prompt" && i + 1 < argc) prompt_arg = argv[++i];
            else if (a == "--template-mode" && i + 1 < argc) template_mode = argv[++i];
            else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::stoi(argv[++i]);
            else if (a == "--max-seq" && i + 1 < argc) max_seq_arg = std::stoi(argv[++i]);
            else if (a == "--serve") serve = true;
            else if (a == "--print-prompt-ids") print_prompt_ids = true;
            else if (a == "--help") {
                std::cout << "usage: rftmw_qwen35_native --pack qwen35-opus-distil.rftmwpk "
                          << "[--prompt \"What is 2+2?\"] [--template-mode final_answer] "
                          << "[--ids 1,2,3] [--max-tokens 2] [--max-seq 512] "
                          << "[--serve] [--print-prompt-ids]\n";
                return 0;
            }
        }
        std::vector<int> ids;
        int max_seq_hint = max_seq_arg;
        if (!ids_arg.empty()) {
            ids = parse_ids(ids_arg);
            max_seq_hint = std::max(max_seq_hint, static_cast<int>(ids.size()) + max_tokens + 4);
        }

        NativeEngine engine;
        auto t0 = Clock::now();
        engine.load(pack, max_seq_hint);
        double load_s = seconds_since(t0);
        std::cerr << "[native] load_sec " << load_s << "\n";

        if (serve) {
            run_server(engine, max_seq_hint);
            return 0;
        }

        if (ids.empty()) {
            if (prompt_arg.empty()) prompt_arg = "What is 2+2? Answer with only the final number.";
            std::string formatted = format_prompt(prompt_arg, template_mode);
            ids = engine.tokenizer.encode(formatted);
            if (print_prompt_ids) {
                std::cerr << "[native] PROMPT_IDS";
                for (int id : ids) std::cerr << " " << id;
                std::cerr << "\n";
            }
        }

        GenerationResult r = generate_once(
            engine,
            prompt_arg,
            template_mode,
            max_tokens,
            max_seq_hint,
            nullptr,
            nullptr,
            &ids);

        std::cout << "OUT_IDS";
        for (int id : r.out_ids) std::cout << " " << id;
        std::cout << "\n";
        std::cout << "OUT_TEXT " << r.out_text << "\n";
        std::cout << "LOAD_SEC " << load_s << "\n";
        std::cout << "PREFILL_TOKENS " << r.prompt_ids.size() << "\n";
        std::cout << "PREFILL_SEC " << r.prefill_s << "\n";
        std::cout << "PREFILL_TOK_PER_SEC " << (r.prompt_ids.empty() ? 0.0 : r.prompt_ids.size() / r.prefill_s) << "\n";
        std::cout << "DECODE_TOKENS " << r.out_ids.size() << "\n";
        std::cout << "DECODE_SEC " << r.decode_s << "\n";
        std::cout << "DECODE_TOK_PER_SEC " << (r.out_ids.empty() ? 0.0 : r.out_ids.size() / r.decode_s) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[native:error] " << e.what() << "\n";
        return 1;
    }
}
