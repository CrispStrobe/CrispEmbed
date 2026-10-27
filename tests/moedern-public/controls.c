/* Model-free C ABI controls for the MoEdern public draft; plain C consumer. */
#include "moedern_text_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL %s\n", m); ++failures; } else printf("ok %s\n", m); } while (0)
int main(void) {
    float score = 7.0f;
    unsetenv("CRISPEMBED_MOEDERN_PUBLIC_TEXT");
    CHECK(sizeof(crispembed_moedern_result) == 8 + 5 * 4 + 4 + 7 * sizeof(void *) || sizeof(void *) != 8,
          "result struct layout (LP64)");
    CHECK(!crispembed_moedern_init("missing.gguf", 2), "gate off rejects init");
    CHECK(strstr(crispembed_moedern_error(NULL), "gate is OFF") != NULL, "gate-off error is thread-local and explicit");
    setenv("CRISPEMBED_MOEDERN_PUBLIC_TEXT", "0", 1);
    CHECK(!crispembed_moedern_init("missing.gguf", 2), "gate=0 stays off");
    setenv("CRISPEMBED_MOEDERN_PUBLIC_TEXT", "1", 1);
    CHECK(!crispembed_moedern_init("no-such-moedern.gguf", 2), "missing model rejected");
    CHECK(crispembed_moedern_error(NULL)[0] != 0, "missing-model error present");
    CHECK(!crispembed_moedern_encode(NULL, "x", 0), "null context encode rejected");
    CHECK(strstr(crispembed_moedern_error(NULL), "null") != NULL, "null-context error");
    CHECK(!crispembed_moedern_score(NULL, "q", "d", &score) && score == 0.0f, "failed score clears output");
    crispembed_moedern_free(NULL);
    printf("%s %d\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
