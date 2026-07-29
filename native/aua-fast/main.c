/*
 * aua-fast — tiny AOT client for the aua warm daemon.
 *
 * Speaks the same newline-delimited JSON protocol as DaemonClient
 * (see src/android_ui_analyser/daemon.py). Avoids Python import + typer
 * startup (~300–500 ms) when a daemon is already listening.
 *
 * Usage (mirrors a subset of `aua`):
 *   aua-fast ping
 *   aua-fast analyze
 *   aua-fast has "Sign in"
 *   aua-fast tap 4
 *   aua-fast key back
 *   aua-fast input 2 "hello"
 *   aua-fast swipe up
 *   aua-fast devices
 *
 * If the daemon socket is down, exec's the real `aua` on PATH (full CLI).
 * Override socket: AUA_DAEMON_SOCKET=/path/to/daemon.sock
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/socket.h>
#include <sys/un.h>

#define DEFAULT_SOCK "~/.cache/android-ui-analyser/daemon.sock"
#define BUF_CAP (1 << 20) /* 1 MiB response cap */

static void die(const char *msg) {
    fprintf(stderr, "aua-fast: %s\n", msg);
    exit(2);
}

static char *expand_home(const char *path) {
    if (path[0] != '~') {
        return strdup(path);
    }
    const char *home = getenv("HOME");
    if (!home) {
        home = "";
    }
    size_t n = strlen(home) + strlen(path); /* path includes '~' */
    char *out = malloc(n + 1);
    if (!out) {
        die("oom");
    }
    snprintf(out, n + 1, "%s%s", home, path + 1);
    return out;
}

static char *default_socket_path(void) {
    const char *env = getenv("AUA_DAEMON_SOCKET");
    if (env && env[0]) {
        return expand_home(env);
    }
    return expand_home(DEFAULT_SOCK);
}

/* JSON string escape into dst (capacity dst_cap). Returns 0 on success. */
static int json_escape(char *dst, size_t dst_cap, const char *src) {
    size_t j = 0;
    if (j + 1 >= dst_cap) {
        return -1;
    }
    dst[j++] = '"';
    for (const unsigned char *p = (const unsigned char *)src; *p; p++) {
        const char *esc = NULL;
        char tmp[7];
        switch (*p) {
            case '"':
                esc = "\\\"";
                break;
            case '\\':
                esc = "\\\\";
                break;
            case '\n':
                esc = "\\n";
                break;
            case '\r':
                esc = "\\r";
                break;
            case '\t':
                esc = "\\t";
                break;
            default:
                if (*p < 0x20) {
                    snprintf(tmp, sizeof(tmp), "\\u%04x", *p);
                    esc = tmp;
                }
                break;
        }
        if (esc) {
            size_t el = strlen(esc);
            if (j + el + 1 >= dst_cap) {
                return -1;
            }
            memcpy(dst + j, esc, el);
            j += el;
        } else {
            if (j + 2 >= dst_cap) {
                return -1;
            }
            dst[j++] = (char)*p;
        }
    }
    if (j + 1 >= dst_cap) {
        return -1;
    }
    dst[j++] = '"';
    dst[j] = '\0';
    return 0;
}

static int daemon_call(const char *sock_path, const char *request, char **response_out) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if (strlen(sock_path) >= sizeof(addr.sun_path)) {
        close(fd);
        return -1;
    }
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }

    size_t len = strlen(request);
    char *payload = malloc(len + 2);
    if (!payload) {
        close(fd);
        return -1;
    }
    memcpy(payload, request, len);
    payload[len] = '\n';
    payload[len + 1] = '\0';
    ssize_t w = write(fd, payload, len + 1);
    free(payload);
    if (w < 0) {
        close(fd);
        return -1;
    }

    char *buf = malloc(BUF_CAP);
    if (!buf) {
        close(fd);
        return -1;
    }
    size_t n = 0;
    while (n + 1 < BUF_CAP) {
        ssize_t r = read(fd, buf + n, BUF_CAP - 1 - n);
        if (r < 0) {
            free(buf);
            close(fd);
            return -1;
        }
        if (r == 0) {
            break;
        }
        n += (size_t)r;
        if (memchr(buf, '\n', n)) {
            break;
        }
    }
    close(fd);
    buf[n] = '\0';
    char *nl = strchr(buf, '\n');
    if (nl) {
        *nl = '\0';
    }
    *response_out = buf;
    return 0;
}

/* Extract JSON value for key "result" via brace/bracket matching. Caller frees. */
static char *extract_result_json(const char *resp) {
    const char *key = strstr(resp, "\"result\"");
    if (!key) {
        return NULL;
    }
    const char *p = key + 8;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ':') {
        p++;
    }
    if (*p == '\0') {
        return NULL;
    }
    if (*p == 'n' && strncmp(p, "null", 4) == 0) {
        return strdup("null");
    }
    if (*p != '{' && *p != '[') {
        /* scalar: copy until comma or end-object */
        const char *start = p;
        if (*p == '"') {
            p++;
            while (*p && *p != '"') {
                if (*p == '\\' && p[1]) {
                    p += 2;
                } else {
                    p++;
                }
            }
            if (*p == '"') {
                p++;
            }
        } else {
            while (*p && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && *p != '\n') {
                p++;
            }
        }
        size_t n = (size_t)(p - start);
        char *out = malloc(n + 1);
        if (!out) {
            return NULL;
        }
        memcpy(out, start, n);
        out[n] = '\0';
        return out;
    }
    char open = *p;
    char close = (open == '{') ? '}' : ']';
    int depth = 0;
    int in_str = 0;
    const char *start = p;
    for (; *p; p++) {
        if (in_str) {
            if (*p == '\\' && p[1]) {
                p++;
                continue;
            }
            if (*p == '"') {
                in_str = 0;
            }
            continue;
        }
        if (*p == '"') {
            in_str = 1;
            continue;
        }
        if (*p == open) {
            depth++;
        } else if (*p == close) {
            depth--;
            if (depth == 0) {
                p++;
                size_t n = (size_t)(p - start);
                char *out = malloc(n + 1);
                if (!out) {
                    return NULL;
                }
                memcpy(out, start, n);
                out[n] = '\0';
                return out;
            }
        }
    }
    return NULL;
}

static int response_ok(const char *resp) {
    /* Prefer an explicit "ok": false before true. */
    if (strstr(resp, "\"ok\":false") || strstr(resp, "\"ok\": false")) {
        return 0;
    }
    if (strstr(resp, "\"ok\":true") || strstr(resp, "\"ok\": true")) {
        return 1;
    }
    return 0;
}

static int exit_code_for_error(const char *resp) {
    if (strstr(resp, "\"code\":\"usage\"")) {
        return 2;
    }
    if (strstr(resp, "\"code\":\"device\"") || strstr(resp, "wait_timeout")) {
        return 3;
    }
    if (strstr(resp, "provider")) {
        return 4;
    }
    if (strstr(resp, "\"code\":\"config\"")) {
        return 5;
    }
    if (strstr(resp, "selector_not_found")) {
        return 6;
    }
    if (strstr(resp, "selector_ambiguous")) {
        return 7;
    }
    if (strstr(resp, "expectation_failed")) {
        return 8;
    }
    return 1;
}

static void fallback_exec_aua(int argc, char **argv) {
    /* Rebuild argv with program name "aua". */
    char **nargv = calloc((size_t)argc + 1, sizeof(char *));
    if (!nargv) {
        die("oom");
    }
    nargv[0] = "aua";
    for (int i = 1; i < argc; i++) {
        nargv[i] = argv[i];
    }
    execvp("aua", nargv);
    fprintf(stderr, "aua-fast: daemon down and `aua` not on PATH (%s)\n", strerror(errno));
    exit(3);
}

static int streq(const char *a, const char *b) {
    return a && b && strcmp(a, b) == 0;
}

/* Build daemon request JSON into buf. Returns 0 on success. */
static int build_request(int argc, char **argv, char *buf, size_t cap) {
    if (argc < 2) {
        return -1;
    }
    const char *cmd = argv[1];
    char esc[4096];

    if (streq(cmd, "ping")) {
        snprintf(buf, cap, "{\"cmd\":\"ping\",\"args\":{}}");
        return 0;
    }
    if (streq(cmd, "analyze") || streq(cmd, "devices") || streq(cmd, "list_devices")) {
        const char *dcmd = streq(cmd, "devices") ? "list_devices" : cmd;
        /* Skip global flags we don't understand; pass source=hierarchy by default for analyze. */
        if (streq(dcmd, "analyze")) {
            snprintf(buf, cap, "{\"cmd\":\"analyze\",\"args\":{\"source\":\"auto\",\"record\":true}}");
        } else {
            snprintf(buf, cap, "{\"cmd\":\"%s\",\"args\":{}}", dcmd);
        }
        return 0;
    }
    if (streq(cmd, "has") && argc >= 3) {
        if (json_escape(esc, sizeof(esc), argv[2]) != 0) {
            return -1;
        }
        snprintf(buf, cap, "{\"cmd\":\"has\",\"args\":{\"text\":%s}}", esc);
        return 0;
    }
    if (streq(cmd, "tap") && argc >= 3) {
        snprintf(buf, cap, "{\"cmd\":\"tap\",\"args\":{\"element_id\":%d,\"observe\":true}}", atoi(argv[2]));
        return 0;
    }
    if (streq(cmd, "key") && argc >= 3) {
        if (json_escape(esc, sizeof(esc), argv[2]) != 0) {
            return -1;
        }
        snprintf(buf, cap, "{\"cmd\":\"key\",\"args\":{\"name\":%s,\"observe\":true}}", esc);
        return 0;
    }
    if (streq(cmd, "input") && argc >= 4) {
        char esc2[4096];
        if (json_escape(esc, sizeof(esc), argv[3]) != 0) {
            return -1;
        }
        snprintf(
            buf,
            cap,
            "{\"cmd\":\"input\",\"args\":{\"element_id\":%d,\"text\":%s,\"observe\":true}}",
            atoi(argv[2]),
            esc
        );
        (void)esc2;
        return 0;
    }
    if (streq(cmd, "swipe") && argc >= 3) {
        if (json_escape(esc, sizeof(esc), argv[2]) != 0) {
            return -1;
        }
        snprintf(
            buf,
            cap,
            "{\"cmd\":\"swipe\",\"args\":{\"direction\":%s,\"observe\":true,\"verify\":false}}",
            esc
        );
        return 0;
    }
    if (streq(cmd, "wait") && argc >= 3) {
        /* aua-fast wait <text>  OR  aua-fast wait --for <text> */
        const char *text = argv[2];
        if (streq(argv[2], "--for") && argc >= 4) {
            text = argv[3];
        }
        if (json_escape(esc, sizeof(esc), text) != 0) {
            return -1;
        }
        snprintf(buf, cap, "{\"cmd\":\"wait\",\"args\":{\"for_\":%s,\"observe\":false}}", esc);
        return 0;
    }
    return -1; /* unsupported → Python fallback */
}

int main(int argc, char **argv) {
    if (argc < 2 || streq(argv[1], "-h") || streq(argv[1], "--help")) {
        fprintf(
            stderr,
            "aua-fast — fast daemon client (no Python import)\n"
            "  aua-fast ping|analyze|devices|has <text>|tap <id>|key <name>|\n"
            "           input <id> <text>|swipe <dir>|wait [--for] <text>\n"
            "Falls back to `aua` on PATH when the daemon is down.\n"
            "Socket: $AUA_DAEMON_SOCKET or %s\n",
            DEFAULT_SOCK
        );
        return argc < 2 ? 2 : 0;
    }

    char req[8192];
    if (build_request(argc, argv, req, sizeof(req)) != 0) {
        /* Unknown shape — let the full CLI handle flags / rare commands. */
        fallback_exec_aua(argc, argv);
    }

    char *sock = default_socket_path();
    char *resp = NULL;
    if (daemon_call(sock, req, &resp) != 0) {
        free(sock);
        fallback_exec_aua(argc, argv);
    }
    free(sock);

    if (!response_ok(resp)) {
        /* Structured error on stderr; empty/partial stdout. */
        fputs(resp, stderr);
        fputc('\n', stderr);
        int code = exit_code_for_error(resp);
        free(resp);
        return code;
    }

    char *result = extract_result_json(resp);
    int has_miss = streq(argv[1], "has") &&
        (strstr(resp, "\"found\":false") || strstr(resp, "\"found\": false"));

    if (result) {
        fputs(result, stdout);
        fputc('\n', stdout);
        free(result);
    } else {
        fputs(resp, stdout);
        fputc('\n', stdout);
    }
    free(resp);
    return has_miss ? 1 : 0;
}
