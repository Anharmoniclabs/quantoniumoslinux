/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
 *
 * rftmwd.cpp - QuantoniumOS RFTMW middleware daemon (Linux)
 * ==========================================================
 *
 * Session-level service that sits between QuantoniumOS apps and the kernel:
 *
 *   Memory broker   one shared read-only mapping per model/weight pack, page-cache
 *                   residency tracking (mincore), prefetch/pin/evict, an LRU byte
 *                   budget, and eviction driven by kernel PSI memory-pressure events.
 *   Workloads       apps are launched as transient systemd scopes in rftmw.slice with
 *                   memory.high / memory.max / swap / cpu.weight limits, tracked here.
 *   Transforms      the RFTMW C++ engine and NASM kernels are served in-process.
 *
 * Protocol: one request line in, one JSON line out, over a Unix stream socket.
 *   PING | HELP | STATS
 *   PACK.OPEN <path> | PACK.LIST | PACK.PREFETCH <id> | PACK.PIN <id>
 *   PACK.UNPIN <id> | PACK.EVICT <id> | PACK.CLOSE <id> | PACK.BUDGET <mb>
 *   SPAWN <name> [--mem-high MB] [--mem-max MB] [--swap-max MB] [--cpu-weight W] -- <argv...>
 *   PROCS | KILL <name>
 *   RFT.SELFTEST [n] | RFT.FORWARD <x0> <x1> ...
 * Tokens are whitespace separated; "double quotes" and backslash escapes group.
 */

#include "rftmw_core.hpp"
#include "rftmw_mmap.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <climits>
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/signalfd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/timerfd.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" {
void rftmw_dot_f32_asm(const float* w, const float* x, float* out, int cols);
float rftmw_spectral_sparse_dot_asm(const uint16_t* idx, const float* cre, const float* cim,
                                    const float* xre, const float* xim, int n);
}

namespace {

// ===================================================================
// Small utilities
// ===================================================================
void logf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fputs("rftmwd: ", stderr);
    std::vfprintf(stderr, fmt, ap);
    std::fputc('\n', stderr);
    va_end(ap);
}

std::string jesc(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += static_cast<char>(c);
        }
    }
    return o;
}

std::string q(const std::string& s) { return "\"" + jesc(s) + "\""; }

std::string err(const std::string& msg) { return "{\"ok\":false,\"error\":" + q(msg) + "}"; }

bool write_file(const std::string& path, const std::string& value) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
    if (fd < 0) return false;
    ssize_t n = ::write(fd, value.data(), value.size());
    ::close(fd);
    return n == static_cast<ssize_t>(value.size());
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_tok = false, quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '\\' && i + 1 < line.size()) { cur += line[++i]; in_tok = true; continue; }
        if (c == '"') { quoted = !quoted; in_tok = true; continue; }
        if (!quoted && (c == ' ' || c == '\t' || c == '\r')) {
            if (in_tok) { out.push_back(cur); cur.clear(); in_tok = false; }
            continue;
        }
        cur += c;
        in_tok = true;
    }
    if (in_tok) out.push_back(cur);
    return out;
}

bool parse_u64(const std::string& s, uint64_t& v) {
    if (s.empty()) return false;
    char* end = nullptr;
    errno = 0;
    unsigned long long r = std::strtoull(s.c_str(), &end, 10);
    if (errno || *end) return false;
    v = r;
    return true;
}

// ===================================================================
// Memory broker: shared pack mappings
// ===================================================================
struct Pack {
    int id = 0;
    std::string path;
    std::string format;  // rftmwpk2 | gguf | raw
    int fd = -1;
    void* base = nullptr;
    size_t size = 0;
    bool pinned = false;
    double last_used = 0;
    uint64_t tensors = 0, layers = 0, hidden = 0;
};

class PackBroker {
public:
    ~PackBroker() {
        for (auto& kv : packs_) unmap(kv.second);
    }

    uint64_t budget_bytes = 0;  // 0 = unlimited

    std::string open(const std::string& raw_path) {
        char resolved[PATH_MAX];
        if (!::realpath(raw_path.c_str(), resolved)) return err("realpath: " + std::string(std::strerror(errno)));
        for (auto& kv : packs_) {
            if (kv.second.path == resolved) { kv.second.last_used = now_s(); return "{\"ok\":true,\"pack\":" + describe(kv.second) + "}"; }
        }
        Pack p;
        p.path = resolved;
        p.fd = ::open(resolved, O_RDONLY | O_CLOEXEC);
        if (p.fd < 0) return err("open: " + std::string(std::strerror(errno)));
        struct stat st;
        if (fstat(p.fd, &st) < 0 || !S_ISREG(st.st_mode) || st.st_size == 0) {
            ::close(p.fd);
            return err("not a non-empty regular file");
        }
        p.size = static_cast<size_t>(st.st_size);
        // MAP_SHARED + PROT_READ: every process mapping this file shares one page-cache copy.
        p.base = ::mmap(nullptr, p.size, PROT_READ, MAP_SHARED, p.fd, 0);
        if (p.base == MAP_FAILED) {
            ::close(p.fd);
            return err("mmap: " + std::string(std::strerror(errno)));
        }
        ::madvise(p.base, p.size, MADV_RANDOM);  // weights are read by layer, not linearly
        identify(p);
        p.id = next_id_++;
        p.last_used = now_s();
        auto& stored = packs_[p.id] = p;
        logf("pack %d mapped: %s (%zu MiB, %s)", stored.id, stored.path.c_str(), stored.size >> 20, stored.format.c_str());
        enforce_budget(stored.id);
        return "{\"ok\":true,\"pack\":" + describe(stored) + "}";
    }

    std::string list() {
        std::string o = "{\"ok\":true,\"budget_bytes\":" + std::to_string(budget_bytes) + ",\"packs\":[";
        bool first = true;
        for (auto& kv : packs_) {
            if (!first) o += ",";
            first = false;
            o += describe(kv.second);
        }
        return o + "]}";
    }

    std::string prefetch(int id) {
        Pack* p = find(id);
        if (!p) return err("no such pack");
        p->last_used = now_s();
        if (::madvise(p->base, p->size, MADV_WILLNEED) != 0) return err("madvise: " + std::string(std::strerror(errno)));
        ::posix_fadvise(p->fd, 0, 0, POSIX_FADV_WILLNEED);
        enforce_budget(id);
        return "{\"ok\":true,\"pack\":" + describe(*p) + "}";
    }

    std::string pin(int id) {
        Pack* p = find(id);
        if (!p) return err("no such pack");
        if (::mlock(p->base, p->size) != 0)
            return err("mlock: " + std::string(std::strerror(errno)) + " (raise LimitMEMLOCK / memlock rlimit)");
        p->pinned = true;
        p->last_used = now_s();
        return "{\"ok\":true,\"pack\":" + describe(*p) + "}";
    }

    std::string unpin(int id) {
        Pack* p = find(id);
        if (!p) return err("no such pack");
        ::munlock(p->base, p->size);
        p->pinned = false;
        return "{\"ok\":true,\"pack\":" + describe(*p) + "}";
    }

    std::string evict(int id) {
        Pack* p = find(id);
        if (!p) return err("no such pack");
        if (p->pinned) return err("pack is pinned; PACK.UNPIN first");
        size_t before = resident(*p);
        drop(*p);
        return "{\"ok\":true,\"freed_bytes\":" + std::to_string(before - std::min(before, resident(*p))) +
               ",\"pack\":" + describe(*p) + "}";
    }

    std::string close(int id) {
        Pack* p = find(id);
        if (!p) return err("no such pack");
        unmap(*p);
        packs_.erase(id);
        return "{\"ok\":true}";
    }

    uint64_t total_resident() {
        uint64_t t = 0;
        for (auto& kv : packs_) t += resident(kv.second);
        return t;
    }

    size_t count() const { return packs_.size(); }

    // Pressure response: drop the least-recently-used unpinned pack that has been idle
    // for at least min_idle_s. Returns bytes reclaimed (0 if nothing eligible).
    size_t evict_lru(double min_idle_s, int keep_id = -1) {
        Pack* victim = nullptr;
        double t = now_s();
        for (auto& kv : packs_) {
            Pack& p = kv.second;
            if (p.pinned || p.id == keep_id || t - p.last_used < min_idle_s) continue;
            if (resident(p) == 0) continue;
            if (!victim || p.last_used < victim->last_used) victim = &p;
        }
        if (!victim) return 0;
        size_t before = resident(*victim);
        drop(*victim);
        size_t freed = before - std::min(before, resident(*victim));
        logf("evicted pack %d (%s): %zu MiB released", victim->id, victim->path.c_str(), freed >> 20);
        return freed;
    }

    void enforce_budget(int keep_id = -1) {
        if (!budget_bytes) return;
        while (total_resident() > budget_bytes) {
            if (!evict_lru(0.0, keep_id)) break;
        }
    }

private:
    std::map<int, Pack> packs_;
    int next_id_ = 1;

    Pack* find(int id) {
        auto it = packs_.find(id);
        return it == packs_.end() ? nullptr : &it->second;
    }

    static void unmap(Pack& p) {
        if (p.base && p.base != MAP_FAILED) {
            if (p.pinned) ::munlock(p.base, p.size);
            ::munmap(p.base, p.size);
        }
        if (p.fd >= 0) ::close(p.fd);
        p.base = nullptr;
        p.fd = -1;
    }

    static void drop(Pack& p) {
#ifdef MADV_PAGEOUT
        ::madvise(p.base, p.size, MADV_PAGEOUT);
#endif
        ::madvise(p.base, p.size, MADV_DONTNEED);
        // Clean file pages no longer mapped anywhere leave the page cache here.
        ::posix_fadvise(p.fd, 0, 0, POSIX_FADV_DONTNEED);
    }

    static size_t resident(const Pack& p) {
        static const long page = sysconf(_SC_PAGESIZE);
        size_t pages = (p.size + page - 1) / page;
        std::vector<unsigned char> vec(pages);
        if (::mincore(p.base, p.size, vec.data()) != 0) return 0;
        size_t n = 0;
        for (unsigned char v : vec) n += v & 1;
        return std::min(n * static_cast<size_t>(page), p.size);
    }

    static void identify(Pack& p) {
        const char* b = static_cast<const char*>(p.base);
        if (p.size >= sizeof(rftmw::mmap::PackHeader) && std::memcmp(b, "RFTMWPK2", 8) == 0) {
            const auto* h = reinterpret_cast<const rftmw::mmap::PackHeader*>(b);
            p.format = "rftmwpk2";
            p.tensors = h->num_tensors;
            p.layers = h->num_layers;
            p.hidden = h->hidden_size;
        } else if (p.size >= 8 && std::memcmp(b, "RFTMWPK", 7) == 0) {
            p.format = "rftmwpk";
        } else if (p.size >= 4 && std::memcmp(b, "GGUF", 4) == 0) {
            p.format = "gguf";
        } else {
            p.format = "raw";
        }
    }

    std::string describe(const Pack& p) const {
        std::ostringstream o;
        o << "{\"id\":" << p.id << ",\"path\":" << q(p.path) << ",\"format\":" << q(p.format)
          << ",\"size_bytes\":" << p.size << ",\"resident_bytes\":" << resident(p)
          << ",\"pinned\":" << (p.pinned ? "true" : "false")
          << ",\"idle_s\":" << static_cast<long>(now_s() - p.last_used);
        if (p.format == "rftmwpk2")
            o << ",\"tensors\":" << p.tensors << ",\"layers\":" << p.layers << ",\"hidden\":" << p.hidden;
        o << "}";
        return o.str();
    }
};

// ===================================================================
// Kernel memory-pressure (PSI) and meminfo
// ===================================================================
struct Psi {
    double some_avg10 = -1, full_avg10 = -1;
};

Psi read_psi() {
    Psi r;
    std::ifstream f("/proc/pressure/memory");
    std::string kind, avg10;
    while (f >> kind >> avg10) {
        std::string rest;
        std::getline(f, rest);
        if (avg10.rfind("avg10=", 0) != 0) continue;
        double v = std::atof(avg10.c_str() + 6);
        if (kind == "some") r.some_avg10 = v;
        if (kind == "full") r.full_avg10 = v;
    }
    return r;
}

std::map<std::string, uint64_t> read_meminfo() {
    std::map<std::string, uint64_t> m;
    std::ifstream f("/proc/meminfo");
    std::string key;
    uint64_t kb;
    std::string unit;
    while (f >> key >> kb) {
        std::getline(f, unit);
        if (!key.empty() && key.back() == ':') key.pop_back();
        m[key] = kb * 1024;
    }
    return m;
}

// Kernel PSI trigger: POLLPRI fires when tasks stall >= stall_us within window_us.
// Unprivileged triggers need window_us to be a multiple of 2s.
int open_psi_trigger(uint32_t stall_us, uint32_t window_us) {
    int fd = ::open("/proc/pressure/memory", O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return -1;
    char trig[64];
    int n = std::snprintf(trig, sizeof trig, "some %u %u", stall_us, window_us);
    if (::write(fd, trig, n + 1) < 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

// ===================================================================
// Workloads: transient systemd scopes in rftmw.slice
// ===================================================================
// Each SPAWN becomes "qos-<name>.scope" under the user manager's rftmw.slice via
// systemd-run, so workloads are first-class units (systemd-cgls, journald), outlive
// a daemon restart, and share the slice's collective memory ceiling. rftmwd reads
// and tunes their cgroup-v2 files directly.
struct Workload {
    std::string name;
    pid_t pid = -1;
    std::string cgroup;  // scope directory; empty when cgroups are unavailable
    bool running = true;
    bool adopted = false;  // found in the slice at startup, not our child
    int status = 0;
    std::time_t started = 0;
};

class WorkloadManager {
public:
    std::string cgroup_note = "not initialised";
    std::string slice = "rftmw";

    void init() {
        std::ifstream f("/proc/self/cgroup");
        std::string line, own;
        while (std::getline(f, line))
            if (line.rfind("0::", 0) == 0) own = line.substr(3);
        size_t at = own.find("/user@");
        if (own.empty() || at == std::string::npos) {
            cgroup_note = "not under a systemd user manager; workloads run without cgroup limits";
            return;
        }
        size_t end = own.find('/', at + 1);
        slice_dir_ = "/sys/fs/cgroup" + own.substr(0, end) + "/" + slice + ".slice";
        cgroup_note = "ok";
        logf("workloads run as systemd scopes in %s.slice (%s)", slice.c_str(), slice_dir_.c_str());
        adopt_existing();
    }

    bool enabled() const { return !slice_dir_.empty(); }

    // peer: the requesting process; the workload inherits its environment and cwd,
    // so `qos-run app` behaves like running `app` from that shell or compositor.
    std::string spawn(const std::vector<std::string>& args, pid_t peer) {
        // SPAWN <name> [opts] -- argv...
        if (args.size() < 2) return err("usage: SPAWN <name> [--mem-high MB] [--mem-max MB] [--swap-max MB] [--cpu-weight W] -- <argv...>");
        std::string name = args[1];
        if (name.empty() || name.size() > 64 ||
            !std::all_of(name.begin(), name.end(), [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_'; }))
            return err("invalid workload name (use [A-Za-z0-9_-])");
        auto it = work_.find(name);
        if (it != work_.end() && it->second.running) return err("workload already running: " + name);
        uint64_t mem_high = 0, mem_max = 0, cpu_weight = 0, swap_max = UINT64_MAX;
        size_t i = 2;
        for (; i < args.size() && args[i] != "--"; ++i) {
            uint64_t* dst = args[i] == "--mem-high" ? &mem_high : args[i] == "--mem-max" ? &mem_max
                          : args[i] == "--swap-max" ? &swap_max : args[i] == "--cpu-weight" ? &cpu_weight : nullptr;
            if (!dst || i + 1 >= args.size() || !parse_u64(args[i + 1], *dst)) return err("bad option: " + args[i]);
            ++i;
        }
        if (i + 1 >= args.size()) return err("missing -- <argv...>");

        std::vector<std::string> argv;
        Workload w;
        w.name = name;
        if (enabled()) {
            w.cgroup = slice_dir_ + "/qos-" + name + ".scope";
            argv = {"systemd-run", "--user", "--scope", "--quiet", "--collect",
                    "--slice=" + slice + ".slice", "--unit=qos-" + name};
            auto prop = [&](const char* key, uint64_t v) {
                argv.push_back("-p");
                argv.push_back(std::string(key) + "=" + std::to_string(v));
            };
            if (mem_high) prop("MemoryHigh", mem_high << 20);
            if (mem_max) prop("MemoryMax", mem_max << 20);
            // Without a swap cap, MemoryMax only pushes the excess into swap/zram.
            if (swap_max != UINT64_MAX) prop("MemorySwapMax", swap_max << 20);
            if (cpu_weight) prop("CPUWeight", cpu_weight);
            argv.push_back("--");
        }
        argv.insert(argv.end(), args.begin() + i + 1, args.end());

        std::vector<std::string> env;
        std::string cwd;
        if (peer > 0) {
            std::string raw = read_file("/proc/" + std::to_string(peer) + "/environ");
            for (size_t b = 0, e; b < raw.size(); b = e + 1) {
                e = raw.find('\0', b);
                if (e == std::string::npos) e = raw.size();
                if (e > b) env.push_back(raw.substr(b, e - b));
            }
            char buf[PATH_MAX];
            ssize_t n = readlink(("/proc/" + std::to_string(peer) + "/cwd").c_str(), buf, sizeof buf - 1);
            if (n > 0) cwd.assign(buf, n);
        }
        std::vector<char*> cenv;
        for (auto& kv : env) cenv.push_back(const_cast<char*>(kv.c_str()));
        cenv.push_back(nullptr);

        pid_t pid = fork();
        if (pid < 0) return err("fork: " + std::string(std::strerror(errno)));
        if (pid == 0) {
            sigset_t none;
            sigemptyset(&none);
            sigprocmask(SIG_SETMASK, &none, nullptr);
            setsid();
            int devnull = ::open("/dev/null", O_RDONLY);
            if (devnull >= 0) { dup2(devnull, 0); ::close(devnull); }
            if (!cwd.empty() && chdir(cwd.c_str()) != 0) (void)!chdir("/");
            std::vector<char*> cargv;
            for (auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
            cargv.push_back(nullptr);
            // systemd-run --scope registers the scope, then execs the command in place,
            // so this pid becomes the workload itself.
            if (env.empty()) execvp(cargv[0], cargv.data());
            else execvpe(cargv[0], cargv.data(), cenv.data());
            _exit(127);
        }
        w.pid = pid;
        w.started = std::time(nullptr);
        work_[name] = w;
        logf("spawned %s pid %d%s", name.c_str(), pid, w.cgroup.empty() ? " (no cgroup)" : "");
        return "{\"ok\":true,\"workload\":" + describe(work_[name]) + "}";
    }

    std::string kill(const std::string& name) {
        auto it = work_.find(name);
        if (it == work_.end() || !it->second.running) return err("no running workload: " + name);
        Workload& w = it->second;
        // cgroup.kill (Linux 5.14+) takes the whole tree, including forked children.
        if (w.cgroup.empty() || !write_file(w.cgroup + "/cgroup.kill", "1")) {
            ::kill(-w.pid, SIGTERM);  // spawn() made it a session/group leader
            ::kill(w.pid, SIGTERM);
        }
        return "{\"ok\":true}";
    }

    void reap() {
        int status;
        pid_t pid;
        while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
            for (auto& kv : work_) {
                if (kv.second.pid == pid && kv.second.running && !kv.second.adopted) {
                    kv.second.running = false;
                    kv.second.status = status;
                    logf("workload %s exited (%s %d)", kv.first.c_str(),
                         WIFSIGNALED(status) ? "signal" : "code", WIFSIGNALED(status) ? WTERMSIG(status) : WEXITSTATUS(status));
                }
            }
        }
        // Adopted workloads are not our children: track them via cgroup.events.
        for (auto& kv : work_) {
            Workload& w = kv.second;
            if (w.adopted && w.running && !populated(w.cgroup)) {
                w.running = false;
                logf("adopted workload %s ended", kv.first.c_str());
            }
        }
    }

    std::string list() const {
        std::string o = "{\"ok\":true,\"cgroups\":" + q(cgroup_note) + ",\"slice\":" + q(slice + ".slice") + ",\"workloads\":[";
        bool first = true;
        for (auto& kv : work_) {
            if (!first) o += ",";
            first = false;
            o += describe(kv.second);
        }
        return o + "]}";
    }

    size_t running() const {
        size_t n = 0;
        for (auto& kv : work_) n += kv.second.running;
        return n;
    }

private:
    std::string slice_dir_;
    std::map<std::string, Workload> work_;

    static bool populated(const std::string& cg) {
        return read_file(cg + "/cgroup.events").find("populated 1") != std::string::npos;
    }

    // Scopes outlive the daemon: pick up the ones still running after a restart.
    void adopt_existing() {
        DIR* d = opendir(slice_dir_.c_str());
        if (!d) return;
        const std::string pre = "qos-", suf = ".scope";
        while (dirent* e = readdir(d)) {
            std::string dir = e->d_name;
            if (dir.size() <= pre.size() + suf.size() || dir.compare(0, pre.size(), pre) != 0 ||
                dir.compare(dir.size() - suf.size(), suf.size(), suf) != 0)
                continue;
            Workload w;
            w.name = dir.substr(pre.size(), dir.size() - pre.size() - suf.size());
            w.cgroup = slice_dir_ + "/" + dir;
            w.adopted = true;
            if (!populated(w.cgroup)) continue;
            std::istringstream procs(read_file(w.cgroup + "/cgroup.procs"));
            procs >> w.pid;
            w.started = std::time(nullptr);
            logf("re-adopted running workload %s (pid %d)", w.name.c_str(), w.pid);
            work_[w.name] = w;
        }
        closedir(d);
    }

    static std::string describe(const Workload& w) {
        std::ostringstream o;
        o << "{\"name\":" << q(w.name) << ",\"pid\":" << w.pid << ",\"running\":" << (w.running ? "true" : "false");
        if (!w.running && !w.adopted)
            o << ",\"exit\":" << (WIFSIGNALED(w.status) ? -WTERMSIG(w.status) : WEXITSTATUS(w.status));
        if (!w.cgroup.empty()) {
            o << ",\"cgroup\":" << q(w.cgroup);
            for (const char* f : {"memory.current", "memory.peak", "memory.high", "memory.max", "memory.swap.max", "pids.current", "cpu.weight"}) {
                std::string v = trim(read_file(w.cgroup + "/" + f));
                if (v.empty()) continue;
                std::string key = f;
                std::replace(key.begin(), key.end(), '.', '_');
                bool numeric = std::all_of(v.begin(), v.end(), ::isdigit);
                o << ",\"" << key << "\":" << (numeric ? v : q(v));
            }
        }
        if (w.adopted) o << ",\"adopted\":true";
        o << ",\"uptime_s\":" << (std::time(nullptr) - w.started) << "}";
        return o.str();
    }
};

// ===================================================================
// Transform services (RFTMW C++ engine + NASM kernels)
// ===================================================================
bool cpu_has_avx2_fma() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

std::string rft_selftest(size_t n) {
    std::mt19937 rng(42);
    std::normal_distribution<double> nd;
    rftmw::RealVec x(n);
    for (auto& v : x) v = nd(rng);
    rftmw::RFTMWEngine engine;
    auto t0 = std::chrono::steady_clock::now();
    auto X = engine.forward(x);
    auto y = engine.inverse(X);
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    double max_err = 0;
    for (size_t i = 0; i < n; ++i) max_err = std::max(max_err, std::abs(x[i] - y[i]));

    std::ostringstream o;
    o << "{\"ok\":true,\"n\":" << n << ",\"roundtrip_max_abs_err\":" << max_err << ",\"roundtrip_us\":" << us;
    if (cpu_has_avx2_fma()) {
        std::vector<float> w(n), xf(n);
        double ref = 0;
        for (size_t i = 0; i < n; ++i) { w[i] = static_cast<float>(nd(rng)); xf[i] = static_cast<float>(x[i]); ref += double(w[i]) * xf[i]; }
        float got = 0;
        rftmw_dot_f32_asm(w.data(), xf.data(), &got, static_cast<int>(n));
        o << ",\"asm_dot_abs_err\":" << std::abs(got - ref);
    } else {
        o << ",\"asm_dot_abs_err\":null,\"asm_note\":\"CPU lacks AVX2/FMA; NASM kernels skipped\"";
    }
    o << "}";
    return o.str();
}

std::string rft_forward(const std::vector<std::string>& args) {
    rftmw::RealVec x;
    for (size_t i = 1; i < args.size(); ++i) {
        char* end = nullptr;
        double v = std::strtod(args[i].c_str(), &end);
        if (*end) return err("not a number: " + args[i]);
        x.push_back(v);
    }
    if (x.empty() || x.size() > RFTMW_CANONICAL_MAX_N) return err("need 1.." + std::to_string(RFTMW_CANONICAL_MAX_N) + " samples");
    rftmw::RFTMWEngine engine;
    auto X = engine.forward(x);
    std::ostringstream o;
    o.precision(12);
    o << "{\"ok\":true,\"coeffs\":[";
    for (size_t k = 0; k < X.size(); ++k) o << (k ? "," : "") << "[" << X[k].real() << "," << X[k].imag() << "]";
    o << "]}";
    return o.str();
}

// ===================================================================
// Daemon
// ===================================================================
struct Client {
    int fd;
    pid_t peer;
    std::string buf;
};

struct Options {
    std::string socket_path;
    uint64_t budget_mb = 0;
    double psi_threshold = 10.0;  // some avg10 %, used when no PSI trigger is available
    double pressure_idle_s = 30.0;
    std::string slice = "rftmw";
};

std::string default_socket() {
    if (const char* s = std::getenv("RFTMW_SOCKET")) return s;
    if (const char* r = std::getenv("XDG_RUNTIME_DIR")) return std::string(r) + "/rftmw/rftmwd.sock";
    return "/run/rftmw/rftmwd.sock";
}

void sd_notify_ready() {
    const char* ns = std::getenv("NOTIFY_SOCKET");
    if (!ns || !*ns) return;
    int fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return;
    sockaddr_un sa{};
    sa.sun_family = AF_UNIX;
    std::strncpy(sa.sun_path, ns, sizeof(sa.sun_path) - 1);
    if (sa.sun_path[0] == '@') sa.sun_path[0] = '\0';  // abstract namespace
    socklen_t len = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + std::strlen(ns));
    const char msg[] = "READY=1\nSTATUS=RFTMW middleware online";
    sendto(fd, msg, sizeof(msg) - 1, 0, reinterpret_cast<sockaddr*>(&sa), len);
    ::close(fd);
}

class Daemon {
public:
    explicit Daemon(Options o) : opt_(std::move(o)) {
        packs_.budget_bytes = opt_.budget_mb << 20;
        work_.slice = opt_.slice;
    }

    int run() {
        sigset_t mask;
        sigemptyset(&mask);
        sigaddset(&mask, SIGTERM);
        sigaddset(&mask, SIGINT);
        sigaddset(&mask, SIGCHLD);
        sigaddset(&mask, SIGHUP);
        sigprocmask(SIG_BLOCK, &mask, nullptr);
        signal(SIGPIPE, SIG_IGN);
        int sfd = signalfd(-1, &mask, SFD_CLOEXEC | SFD_NONBLOCK);

        work_.init();

        int lfd = listen_socket();
        if (lfd < 0) return 1;

        int tfd = timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC | TFD_NONBLOCK);
        itimerspec its{};
        its.it_interval.tv_sec = its.it_value.tv_sec = 5;
        timerfd_settime(tfd, 0, &its, nullptr);

        psi_fd_ = open_psi_trigger(150000, 2000000);
        logf("memory pressure: %s", psi_fd_ >= 0 ? "PSI trigger armed (150ms / 2s)" : "polling /proc/pressure/memory");
        logf("listening on %s (pack budget %s)", opt_.socket_path.c_str(),
             opt_.budget_mb ? (std::to_string(opt_.budget_mb) + " MiB").c_str() : "unlimited");
        sd_notify_ready();

        bool stop = false;
        while (!stop) {
            std::vector<pollfd> fds = {{sfd, POLLIN, 0}, {lfd, POLLIN, 0}, {tfd, POLLIN, 0}};
            if (psi_fd_ >= 0) fds.push_back({psi_fd_, POLLPRI, 0});
            size_t client_base = fds.size();
            for (auto& c : clients_) fds.push_back({c.fd, POLLIN, 0});

            if (poll(fds.data(), fds.size(), -1) < 0) {
                if (errno == EINTR) continue;
                logf("poll: %s", std::strerror(errno));
                break;
            }
            if (fds[0].revents & POLLIN) {
                signalfd_siginfo si;
                while (read(sfd, &si, sizeof si) == sizeof si) {
                    if (si.ssi_signo == SIGCHLD) work_.reap();
                    else if (si.ssi_signo != SIGHUP) stop = true;
                }
            }
            if (fds[1].revents & POLLIN) accept_clients(lfd);
            if (fds[2].revents & POLLIN) {
                uint64_t ticks;
                (void)!read(tfd, &ticks, sizeof ticks);
                housekeeping();
            }
            if (psi_fd_ >= 0 && (fds[3].revents & POLLPRI)) on_pressure("PSI trigger");
            for (size_t i = fds.size(); i-- > client_base;) {
                if (fds[i].revents & (POLLIN | POLLHUP | POLLERR)) service_client(i - client_base);
            }
        }
        logf("shutting down");
        for (auto& c : clients_) ::close(c.fd);
        ::close(lfd);
        ::unlink(opt_.socket_path.c_str());
        return 0;
    }

private:
    Options opt_;
    PackBroker packs_;
    WorkloadManager work_;
    std::vector<Client> clients_;
    int psi_fd_ = -1;
    uint64_t pressure_events_ = 0;
    uint64_t pressure_freed_ = 0;

    int listen_socket() {
        std::string dir = opt_.socket_path.substr(0, opt_.socket_path.rfind('/'));
        if (!dir.empty()) ::mkdir(dir.c_str(), 0700);
        ::unlink(opt_.socket_path.c_str());
        int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
        sockaddr_un sa{};
        sa.sun_family = AF_UNIX;
        if (opt_.socket_path.size() >= sizeof(sa.sun_path)) { logf("socket path too long"); return -1; }
        std::strcpy(sa.sun_path, opt_.socket_path.c_str());
        mode_t old = umask(0077);
        int rc = bind(fd, reinterpret_cast<sockaddr*>(&sa), sizeof sa);
        umask(old);
        if (rc < 0 || listen(fd, 16) < 0) {
            logf("bind %s: %s", opt_.socket_path.c_str(), std::strerror(errno));
            return -1;
        }
        return fd;
    }

    void accept_clients(int lfd) {
        int fd;
        while ((fd = accept4(lfd, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK)) >= 0) {
            ucred cred{};
            socklen_t len = sizeof cred;
            if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &len) != 0 || (cred.uid != getuid() && cred.uid != 0)) {
                ::close(fd);  // the socket is 0600 already; this is belt and braces
                continue;
            }
            clients_.push_back({fd, cred.pid, {}});
        }
    }

    void service_client(size_t idx) {
        Client& c = clients_[idx];
        char buf[4096];
        ssize_t n = read(c.fd, buf, sizeof buf);
        if (n <= 0) {
            if (n < 0 && errno == EAGAIN) return;
            ::close(c.fd);
            clients_.erase(clients_.begin() + idx);
            return;
        }
        c.buf.append(buf, n);
        if (c.buf.size() > (1 << 16)) {
            ::close(c.fd);
            clients_.erase(clients_.begin() + idx);
            return;
        }
        size_t nl;
        while ((nl = c.buf.find('\n')) != std::string::npos) {
            std::string line = c.buf.substr(0, nl);
            c.buf.erase(0, nl + 1);
            std::string resp = handle(line, c.peer) + "\n";
            const char* p = resp.data();
            size_t left = resp.size();
            while (left) {
                ssize_t w = write(c.fd, p, left);
                if (w < 0 && errno == EAGAIN) { pollfd pf{c.fd, POLLOUT, 0}; poll(&pf, 1, 1000); continue; }
                if (w <= 0) break;
                p += w;
                left -= w;
            }
        }
    }

    void on_pressure(const char* source) {
        ++pressure_events_;
        size_t freed = packs_.evict_lru(opt_.pressure_idle_s);
        pressure_freed_ += freed;
        logf("memory pressure (%s): reclaimed %zu MiB from idle packs", source, freed >> 20);
    }

    void housekeeping() {
        work_.reap();
        packs_.enforce_budget();
        if (psi_fd_ < 0) {
            Psi p = read_psi();
            if (p.some_avg10 >= opt_.psi_threshold) on_pressure("PSI avg10 poll");
        }
    }

    std::string stats() {
        auto mi = read_meminfo();
        Psi p = read_psi();
        std::ostringstream o;
        o << "{\"ok\":true,\"mem_total\":" << mi["MemTotal"] << ",\"mem_available\":" << mi["MemAvailable"]
          << ",\"page_cache\":" << mi["Cached"] << ",\"swap_total\":" << mi["SwapTotal"]
          << ",\"swap_free\":" << mi["SwapFree"] << ",\"psi_some_avg10\":" << p.some_avg10
          << ",\"psi_full_avg10\":" << p.full_avg10 << ",\"psi_trigger\":" << (psi_fd_ >= 0 ? "true" : "false")
          << ",\"pressure_events\":" << pressure_events_ << ",\"pressure_reclaimed_bytes\":" << pressure_freed_
          << ",\"packs\":" << packs_.count() << ",\"packs_resident_bytes\":" << packs_.total_resident()
          << ",\"pack_budget_bytes\":" << packs_.budget_bytes << ",\"workloads_running\":" << work_.running()
          << ",\"cgroups\":" << q(work_.cgroup_note) << ",\"avx2_fma\":" << (cpu_has_avx2_fma() ? "true" : "false")
          << ",\"pid\":" << getpid() << "}";
        return o.str();
    }

    std::string handle(const std::string& line, pid_t peer) {
        auto a = tokenize(line);
        if (a.empty()) return err("empty request");
        std::string cmd = a[0];
        std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::toupper);
        auto id_arg = [&](int& id) {
            uint64_t v;
            if (a.size() < 2 || !parse_u64(a[1], v)) return false;
            id = static_cast<int>(v);
            return true;
        };
        int id;
        if (cmd == "PING") return "{\"ok\":true,\"pong\":true}";
        if (cmd == "HELP")
            return "{\"ok\":true,\"commands\":[\"PING\",\"STATS\",\"PACK.OPEN path\",\"PACK.LIST\",\"PACK.PREFETCH id\","
                   "\"PACK.PIN id\",\"PACK.UNPIN id\",\"PACK.EVICT id\",\"PACK.CLOSE id\",\"PACK.BUDGET mb\","
                   "\"SPAWN name [--mem-high MB] [--mem-max MB] [--swap-max MB] [--cpu-weight W] -- argv...\",\"PROCS\",\"KILL name\","
                   "\"RFT.SELFTEST [n]\",\"RFT.FORWARD x0 x1 ...\"]}";
        if (cmd == "STATS") return stats();
        if (cmd == "PACK.OPEN") return a.size() == 2 ? packs_.open(a[1]) : err("usage: PACK.OPEN <path>");
        if (cmd == "PACK.LIST") return packs_.list();
        if (cmd == "PACK.PREFETCH") return id_arg(id) ? packs_.prefetch(id) : err("usage: PACK.PREFETCH <id>");
        if (cmd == "PACK.PIN") return id_arg(id) ? packs_.pin(id) : err("usage: PACK.PIN <id>");
        if (cmd == "PACK.UNPIN") return id_arg(id) ? packs_.unpin(id) : err("usage: PACK.UNPIN <id>");
        if (cmd == "PACK.EVICT") return id_arg(id) ? packs_.evict(id) : err("usage: PACK.EVICT <id>");
        if (cmd == "PACK.CLOSE") return id_arg(id) ? packs_.close(id) : err("usage: PACK.CLOSE <id>");
        if (cmd == "PACK.BUDGET") {
            uint64_t mb;
            if (a.size() != 2 || !parse_u64(a[1], mb)) return err("usage: PACK.BUDGET <mb> (0 = unlimited)");
            packs_.budget_bytes = mb << 20;
            packs_.enforce_budget();
            return "{\"ok\":true,\"budget_bytes\":" + std::to_string(packs_.budget_bytes) + "}";
        }
        if (cmd == "SPAWN") return work_.spawn(a, peer);
        if (cmd == "PROCS") {
            work_.reap();  // adopted scopes don't raise SIGCHLD; refresh before answering
            return work_.list();
        }
        if (cmd == "KILL") return a.size() == 2 ? work_.kill(a[1]) : err("usage: KILL <name>");
        if (cmd == "RFT.SELFTEST") {
            uint64_t n = 64;
            if (a.size() > 1 && (!parse_u64(a[1], n) || n < 2 || n > 128)) return err("n must be 2..128");
            return rft_selftest(n);
        }
        if (cmd == "RFT.FORWARD") return rft_forward(a);
        return err("unknown command: " + a[0] + " (try HELP)");
    }
};

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    opt.socket_path = default_socket();
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--socket") opt.socket_path = next();
        else if (a == "--pack-budget-mb") opt.budget_mb = std::strtoull(next(), nullptr, 10);
        else if (a == "--psi-threshold") opt.psi_threshold = std::atof(next());
        else if (a == "--pressure-idle-s") opt.pressure_idle_s = std::atof(next());
        else if (a == "--slice") opt.slice = next();
        else if (a == "--version") { std::puts("rftmwd 0.1.0"); return 0; }
        else {
            std::fprintf(stderr,
                         "usage: rftmwd [--socket PATH] [--pack-budget-mb MB] [--psi-threshold PCT]\n"
                         "              [--pressure-idle-s S] [--slice NAME]\n");
            return a == "--help" ? 0 : 2;
        }
    }
    return Daemon(opt).run();
}
