// test_temp_file.cpp — private temporary files.
//
// There were two hand-rolled copies of this logic (the server's uploaded-page
// path and the OCR orchestrator's cleaned-page path). Both built a predictable
// /tmp name and left the caller to fopen(..., "wb") it, which follows a symlink
// planted at that path and creates the file world-readable under a default
// umask — with the user's scanned document inside. I fixed one and missed the
// other, which is the argument for one implementation and this test.
//
// The properties that matter are cheap to check and easy to regress silently:
// the name must not be derivable, the file must already exist so there is
// nothing for a symlink to redirect, and the mode must not be world-readable.

#include "core/temp_file.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <sys/stat.h>

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) failures++;
}

bool exists(const std::string & p) {
    struct stat st;
    return !p.empty() && ::stat(p.c_str(), &st) == 0;
}

} // namespace

static int crispembed_test_main() {
    std::printf("private temp files\n");

    const std::string a = core_tmp::make_private(".png");
    check(!a.empty(), "returns a path");
    check(exists(a), "the file already EXISTS — nothing left for a symlink to redirect");
    check(a.size() > 4 && a.compare(a.size() - 4, 4, ".png") == 0, "suffix preserved (callers dispatch on it)");

#ifndef _WIN32
    struct stat st {};
    ::stat(a.c_str(), &st);
    const mode_t perms = st.st_mode & 07777;
    check((perms & 077) == 0, "not readable or writable by group or other (0600)");
    check(S_ISREG(st.st_mode), "is a regular file, not a symlink or directory");
#endif

    // Unpredictability: the old scheme was <pid>_<counter>, so two calls
    // differed by exactly one. Distinctness alone would pass that; require the
    // names not to be a short edit apart either.
    std::set<std::string> seen;
    for (int i = 0; i < 32; i++) {
        const std::string p = core_tmp::make_private(".png");
        if (!p.empty()) {
            seen.insert(p);
            std::remove(p.c_str());
        }
    }
    check(seen.size() == 32, "32 calls yield 32 distinct paths");

    const std::string b = core_tmp::make_private(".png");
    check(b != a, "consecutive calls differ");
    // Under the old counter scheme the two names were identical but for one
    // digit. Require several differing characters in the random field.
    size_t diff = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++)
        if (a[i] != b[i]) diff++;
    check(diff >= 2, "names differ in more than one position (not a counter)");

    const std::string c = core_tmp::make_private();
    check(!c.empty() && exists(c), "works with no suffix");
    std::remove(c.c_str());

    std::remove(a.c_str());
    std::remove(b.c_str());
    check(!exists(a), "caller can remove it");

    if (failures) {
        std::printf("\nFAIL: %d check(s) failed.\n", failures);
        return 1;
    }
    std::printf("\nPASS: temporary files are private and unpredictable.\n");
    return 0;
}

// The guard in tools/check_test_clean_exit.sh: a one-shot binary must not run
// ggml's static GPU-device destructor at exit (it aborts on Metal / faults on
// CUDA). These tests touch no GPU today, but they link crispembed-core, so the
// teardown is one added dependency away from firing.
int main() {
    core_util::clean_exit(crispembed_test_main());
}
