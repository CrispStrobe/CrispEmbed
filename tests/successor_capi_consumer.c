// Compile against the shipped header and link only the shipped import/shared library.
#include "crispembed.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    crispembed_set_offline(1);
    crispembed_context * ctx = crispembed_init(argv[1], 2);
    if (!ctx) return 2;
    const crispembed_hparams * hp = crispembed_get_hparams(ctx);
    int dim = 0;
    const float * vector = crispembed_encode(ctx, "A small reproducible embedding consumer.", &dim);
    if (!hp || hp->n_embd != 384 || hp->n_head != 12 || hp->n_layer != 6 || !vector || dim != 384) {
        crispembed_free(ctx);
        return 1;
    }
    printf("{\"hparams_size\":%zu,\"prefix_offsets\":[%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu],\"vector\":[", sizeof(*hp),
           offsetof(crispembed_hparams, n_vocab), offsetof(crispembed_hparams, n_max_tokens),
           offsetof(crispembed_hparams, n_embd), offsetof(crispembed_hparams, n_head),
           offsetof(crispembed_hparams, n_layer), offsetof(crispembed_hparams, n_intermediate),
           offsetof(crispembed_hparams, n_output), offsetof(crispembed_hparams, layer_norm_eps));
    for (int i = 0; i < dim; i++) printf(i ? ",%.9g" : "%.9g", vector[i]);
    puts("]}");
    crispembed_free(ctx);
    fflush(NULL);
    _Exit(0); // Test process only: avoid process-global ggml GPU static teardown.
}
