// tests/test_pix2struct_dec.cpp -- Pix2Struct decoder step 0 parity.
#include "pix2struct.h"
#include "crispembed_diff.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s model.gguf enc-ref.gguf dec-ref.gguf\n", argv[0]);
        return 1;
    }
    printf("Pix2Struct -- decoder step 0 parity\n");

    crispembed_diff::Ref enc_ref, dec_ref;
    enc_ref.load(argv[2]);
    dec_ref.load(argv[3]);

    pix2struct_context * ctx = pix2struct_init(argv[1], 1);
    if (!ctx) { printf("Load failed\n"); return 1; }

    // Encode patches (proven correct)
    auto [patches, pn] = enc_ref.get_f32("flattened_patches");
    auto sh = enc_ref.shape("flattened_patches");
    int n_patches = (int)sh[1];
    int out_dim = 0;
    pix2struct_encode_patches(ctx, patches, n_patches, &out_dim);
    printf("Encoder: %d patches, %d dim\n", n_patches, out_dim);

    // Run decoder step 0: input token = 0 (decoder_start_token_id)
    // We need to call decoder_step directly, but it's static.
    // For now, use the generate function (which needs to be implemented)
    // TODO: expose decoder step for testing

    printf("(Decoder test skeleton -- needs exposed decoder API)\n");
    pix2struct_free(ctx);
    return 0;
}
