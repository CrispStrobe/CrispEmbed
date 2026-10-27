/* CrispEmbed MoEdern C-API timing consumer (no Python/ctypes copies).
 * usage: bench_ours MODEL THREADS WORKLOAD.tsv REPS   (tsv: role<TAB>hex-utf8, "-" = empty)
 * Warm-up (first 16 of each role) excluded. Prints one JSON object. */
#include "moedern_text_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static char * unhex(const char * h) {
    if (!strcmp(h, "-")) h = "";
    size_t n = strlen(h) / 2; char * s = malloc(n + 1);
    for (size_t i = 0; i < n; ++i) sscanf(h + 2 * i, "%2hhx", (unsigned char *) &s[i]);
    s[n] = 0; return s;
}
int main(int argc, char ** argv) {
    if (argc != 5) { fprintf(stderr, "MODEL THREADS WORKLOAD.tsv REPS\n"); return 2; }
    int threads = atoi(argv[2]), reps = atoi(argv[4]);
    static char line[1 << 18], role[32], hex[1 << 18];
    char * texts[4096]; int roles[4096], n = 0;
    FILE * f = fopen(argv[3], "r"); if (!f) return 2;
    while (n < 4096 && fgets(line, sizeof line, f) && sscanf(line, "%31s %262143s", role, hex) == 2) {
        roles[n] = strcmp(role, "query") ? 1 : 0; texts[n++] = unhex(hex);
    }
    fclose(f);
    double t = now();
    crispembed_moedern_context * ctx = crispembed_moedern_init(argv[1], threads);
    if (!ctx) { printf("{\"error\":\"init failed\"}\n"); fprintf(stderr, "%s\n", crispembed_moedern_error(NULL)); return 1; }
    double load = now() - t;
    for (int pass = 0; pass < 2; ++pass) {  /* warm-up: first 16 of each role */
        int seen = 0;
        for (int i = 0; i < n && seen < 16; ++i) if (roles[i] == pass) { crispembed_moedern_encode(ctx, texts[i], pass); ++seen; }
    }
    long doc_tokens = 0, query_tokens = 0; int max_doc = 0, n_docs = 0, n_queries = 0;
    printf("{\"load_seconds\":%.6f,\"reps\":[", load);
    for (int r = 0; r < reps; ++r) {
        double sec[2] = {0, 0};
        for (int pass = 1; pass >= 0; --pass) {  /* documents first, then queries */
            double t0 = now();
            for (int i = 0; i < n; ++i) if (roles[i] == pass) {
                const crispembed_moedern_result * x = crispembed_moedern_encode(ctx, texts[i], pass);
                if (!x) { fprintf(stderr, "%s\n", crispembed_moedern_error(ctx)); return 1; }
                if (r == 0) {
                    if (pass) { doc_tokens += x->n_tokens; ++n_docs; if (x->n_tokens > max_doc) max_doc = x->n_tokens; }
                    else { query_tokens += x->n_tokens; ++n_queries; }
                }
            }
            sec[pass] = now() - t0;
        }
        printf(r ? ",{\"documents_seconds\":%.6f,\"queries_seconds\":%.6f}" : "{\"documents_seconds\":%.6f,\"queries_seconds\":%.6f}", sec[1], sec[0]);
    }
    crispembed_moedern_free(ctx);
    struct rusage u; getrusage(RUSAGE_SELF, &u);
    printf("],\"n_documents\":%d,\"n_queries\":%d,\"document_tokens\":%ld,\"query_tokens\":%ld,\"max_document_tokens\":%d,"
           "\"doc_batch\":1,\"query_batch\":1,\"process_peak_rss_kib\":%ld}\n", n_docs, n_queries, doc_tokens, query_tokens, max_doc, u.ru_maxrss);
    return 0;
}
