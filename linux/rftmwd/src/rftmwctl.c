/*
 * SPDX-License-Identifier: LicenseRef-SPX-Proprietary
 * Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
 *
 * rftmwctl - command-line client for rftmwd
 *
 *   rftmwctl stats
 *   rftmwctl pack.open ~/models/qwen.rftmwpk
 *   rftmwctl spawn chat --mem-high 6144 -- rftmw_qwen25_native ...
 *
 * Arguments are quoted onto one request line; the JSON reply goes to stdout.
 * Exit status is 1 when the daemon answers {"ok":false,...}, 2 if unreachable.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static void socket_path(char* out, size_t n) {
    const char* s = getenv("RFTMW_SOCKET");
    const char* r = getenv("XDG_RUNTIME_DIR");
    if (s) snprintf(out, n, "%s", s);
    else if (r) snprintf(out, n, "%s/rftmw/rftmwd.sock", r);
    else snprintf(out, n, "/run/rftmw/rftmwd.sock");
}

int main(int argc, char** argv) {
    if (argc < 2 || !strcmp(argv[1], "--help")) {
        fprintf(stderr, "usage: rftmwctl <command> [args...]   (rftmwctl help lists commands)\n");
        return argc < 2 ? 2 : 0;
    }
    char path[sizeof(((struct sockaddr_un*)0)->sun_path)];
    socket_path(path, sizeof path);

    size_t cap = 4096, len = 0;
    char* req = malloc(cap);
    for (int i = 1; i < argc; ++i) {
        size_t need = len + 2 * strlen(argv[i]) + 4;
        if (need > cap) { cap = need * 2; req = realloc(req, cap); }
        if (i > 1) req[len++] = ' ';
        req[len++] = '"';
        for (const char* p = argv[i]; *p; ++p) {
            if (*p == '"' || *p == '\\') req[len++] = '\\';
            req[len++] = *p == '\n' ? ' ' : *p;
        }
        req[len++] = '"';
    }
    req[len++] = '\n';

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un sa = {.sun_family = AF_UNIX};
    snprintf(sa.sun_path, sizeof sa.sun_path, "%s", path);
    if (fd < 0 || connect(fd, (struct sockaddr*)&sa, sizeof sa) != 0) {
        fprintf(stderr, "rftmwctl: cannot reach rftmwd at %s: %s\n", path, strerror(errno));
        return 2;
    }
    for (size_t off = 0; off < len;) {
        ssize_t w = write(fd, req + off, len - off);
        if (w <= 0) { perror("rftmwctl: write"); return 2; }
        off += (size_t)w;
    }

    int failed = 0;
    char buf[8192], prev[8] = {0};
    ssize_t n;
    while ((n = read(fd, buf, sizeof buf)) > 0) {
        fwrite(buf, 1, (size_t)n, stdout);
        /* "ok":false can only appear at the start of the reply. */
        if (!prev[0]) { memcpy(prev, buf, n < 7 ? (size_t)n : 7); failed = !strncmp(buf, "{\"ok\":f", (size_t)(n < 7 ? n : 7)); }
        if (buf[n - 1] == '\n') break;
    }
    close(fd);
    free(req);
    return failed;
}
