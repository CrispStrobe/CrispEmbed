#include "tesseract_dawg.h"

#include <cstdio>
#include <cstring>

int main() {
    // magic=42, unicharset_size=2, one forward edge, letter=1, marker+EOW.
    const char * valid = "KgACAAAAAQAAAAsAAAAAAAAA";
    char compact[64];
    int out = 0;
    for (const char * p = valid; *p; ++p) {
        if (*p != ' ') compact[out++] = *p;
    }
    compact[out] = '\0';
    char error[128];
    if (!tesseract_dawg_validate_base64(compact, error, sizeof(error))) {
        std::fprintf(stderr, "valid DAWG rejected: %s\n", error);
        return 1;
    }
    if (tesseract_dawg_validate_base64("AAAA", error, sizeof(error))) {
        std::fprintf(stderr, "invalid DAWG accepted\n");
        return 1;
    }
    std::puts("tesseract DAWG validation: PASS");
    return 0;
}
