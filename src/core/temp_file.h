// temp_file.h — one way to make a private temporary file.
//
// This exists because there were two copies of it and I fixed only one. The
// server wrote uploaded pages to /tmp/crispembed_doc_<pid>_<n>.img and the OCR
// orchestrator wrote cleaned pages to /tmp/crispembed_ocr_<pid>_<counter>.png;
// both built a PREDICTABLE name and then fopen(..., "wb")'d it, which follows a
// symlink planted at that path and leaves the file world-readable under a
// default umask. The content in both cases is the user's scanned document.
// Fixing one instance and missing the other is exactly what a shared helper
// prevents, so there is now one implementation and a test for it.
//
// mkstemp creates the file itself — unpredictable name, O_EXCL, mode 0600. The
// descriptor is closed and the path returned, which is safe for callers that
// must write by name (stb_image_write, fopen): the file already exists and is
// owned by us, so there is nothing left for a symlink to redirect.

#pragma once

#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
// Plain <windows.h> pulls in the legacy <winsock.h>, which then collides with
// <winsock2.h> in any TU that also does networking (server.cpp includes this
// header before httplib.h). The build passes -DWIN32_LEAN_AND_MEAN globally;
// repeat it here so an out-of-tree consumer of this header cannot inherit the
// clash. Guarded so the two definitions cannot conflict (C4005).
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace core_tmp {

// Returns "" on failure. `suffix` may be empty; when given (e.g. ".png") it is
// preserved on the end of the name, which callers that dispatch on extension
// depend on.
inline std::string make_private(const char * suffix = nullptr) {
    const char * dir = std::getenv("TMPDIR");
    if (!dir || !*dir) dir = std::getenv("TEMP");
    if (!dir || !*dir) dir = "/tmp";

#ifdef _WIN32
    // Windows temp directories are per-user, and GetTempFileName creates the
    // file exclusively and hands back a unique name.
    char path[MAX_PATH];
    if (GetTempFileNameA(dir, "crsp", 0, path) == 0) return std::string();
    std::string out(path);
    if (suffix && *suffix) {
        const std::string renamed = out + suffix;
        if (MoveFileA(out.c_str(), renamed.c_str())) out = renamed;
    }
    return out;
#else
    std::string tmpl(dir);
    if (!tmpl.empty() && tmpl.back() == '/') tmpl.pop_back();
    tmpl += "/crispembed_XXXXXX";
    const int suffix_len = (suffix && *suffix) ? (int)std::string(suffix).size() : 0;
    if (suffix_len) tmpl += suffix;

    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const int fd = suffix_len ? mkstemps(buf.data(), suffix_len) : mkstemp(buf.data());
    if (fd < 0) return std::string();
    close(fd);
    return std::string(buf.data());
#endif
}

} // namespace core_tmp
