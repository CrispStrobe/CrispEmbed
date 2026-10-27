/* Actual C consumer: hex-encoded frozen raw text in, complete typed output out. */
#include "moedern_text_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static char * unhex(const char * h) {
    size_t n = strlen(h) / 2;
    char * s = malloc(n + 1);
    for (size_t i = 0; i < n; ++i) sscanf(h + 2 * i, "%2hhx", (unsigned char *)&s[i]);
    s[n] = 0;
    return s;
}
static void u8s(const char * k, const uint8_t * v, int n) {
    printf(",\"%s\":[", k);
    for (int i = 0; i < n; ++i) printf(i ? ",%u" : "%u", v[i]);
    printf("]");
}
static void f32s(const char * k, const float * v, int n) {
    printf(",\"%s\":[", k);
    for (int i = 0; i < n; ++i) printf(i ? ",%.9g" : "%.9g", v[i]);
    printf("]");
}
int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "model cases.tsv pairs.tsv\n");
        return 2;
    }
    crispembed_moedern_context * ctx = crispembed_moedern_init(argv[1], 2);
    if (!ctx) {
        fprintf(stderr, "%s\n", crispembed_moedern_error(NULL));
        return 1;
    }
    if (crispembed_moedern_encode(ctx, "x", 7) || !crispembed_moedern_error(ctx)[0]) {
        fprintf(stderr, "invalid role accepted\n");
        return 1;
    }
    char line[1 << 17], id[256], role[32], hex[1 << 17];
    FILE * f = fopen(argv[2], "r");
    while (fgets(line, sizeof line, f) && sscanf(line, "%255s %31s %131071s", id, role, hex) >= 2) {
        char * text = unhex(strcmp(hex, "-") ? hex : "");
        const crispembed_moedern_result * r = crispembed_moedern_encode(ctx, text, strcmp(role, "query") ? 1 : 0);
        if (!r) {
            fprintf(stderr, "%s: %s\n", id, crispembed_moedern_error(ctx));
            return 1;
        }
        printf("{\"kind\":\"case\",\"id\":\"%s\",\"role\":\"%s\",\"ids\":[", id, role);
        for (int i = 0; i < r->n_tokens; ++i) printf(i ? ",%d" : "%d", r->ids[i]);
        printf("]");
        u8s("supplied_mask", r->supplied_mask, r->n_tokens);
        u8s("effective_mask", r->effective_mask, r->n_tokens);
        u8s("scoring_mask", r->scoring_mask, r->n_tokens);
        u8s("expansion_positions", r->expansion_positions, r->expansion_count);
        f32s("all_vectors", r->all_vectors, r->n_tokens * 128);
        f32s("selected_vectors", r->selected_vectors, r->n_selected * 128);
        printf("}\n");
        free(text);
        hex[0] = 0;
    }
    fclose(f);
    f = fopen(argv[3], "r");
    char qh[1 << 16], dh[1 << 16];
    while (fgets(line, sizeof line, f) && sscanf(line, "%255s %65535s %65535s", id, qh, dh) == 3) {
        char *q = unhex(strcmp(qh, "-") ? qh : ""), *d = unhex(strcmp(dh, "-") ? dh : "");
        float s = -1;
        if (!crispembed_moedern_score(ctx, q, d, &s)) {
            fprintf(stderr, "%s\n", crispembed_moedern_error(ctx));
            return 1;
        }
        printf("{\"kind\":\"score\",\"id\":\"%s\",\"score\":%.9g}\n", id, s);
        free(q);
        free(d);
    }
    fclose(f);
    float s = 5;
    if (crispembed_moedern_score(ctx, "q", NULL, &s) || s != 0) {
        fprintf(stderr, "failed score not cleared\n");
        return 1;
    }
    crispembed_moedern_free(ctx);
    return 0;
}
