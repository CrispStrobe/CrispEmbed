#include "tesseract_dawg.h"

#include <cstdio>
#include <cstring>

int main() {
    // magic=42, unicharset_size=3, two forward edges: 1 -> 2 (word end).
    const char * valid = "KgADAAAAAgAAACUAAAAAAAAAFgAAAAAAAAA=";
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
    const int one[] = { 1 };
    const int two[] = { 2 };
    const int word[] = { 1, 2 };
    if (tesseract_dawg_contains_base64(compact, one, 1) || !tesseract_dawg_has_prefix_base64(compact, one, 1) ||
        !tesseract_dawg_contains_base64(compact, word, 2) || tesseract_dawg_has_prefix_base64(compact, two, 1)) {
        std::fprintf(stderr, "DAWG exact-word lookup failed\n");
        return 1;
    }
    if (tesseract_dawg_validate_base64("AAAA", error, sizeof(error))) {
        std::fprintf(stderr, "invalid DAWG accepted\n");
        return 1;
    }
    tesseract_dawg_context * ctx = tesseract_dawg_init_base64(valid, error, sizeof(error));
    if (!ctx || !tesseract_dawg_context_has_prefix(ctx, one, 1) || !tesseract_dawg_context_contains(ctx, word, 2) ||
        tesseract_dawg_context_contains(ctx, one, 1)) {
        std::fprintf(stderr, "cached DAWG lookup failed: %s\n", error);
        tesseract_dawg_free(ctx);
        return 1;
    }
    tesseract_dawg_free(ctx);
    if (tesseract_dawg_validate_base64("K===", error, sizeof(error)) ||
        tesseract_dawg_validate_base64("KgADAAAAAgAAACUAAAAAAAAAFgAAAAAAAAA=A", error, sizeof(error))) {
        std::fprintf(stderr, "malformed base64 accepted\n");
        return 1;
    }
    std::puts("tesseract DAWG validation: PASS");
    return 0;
}
