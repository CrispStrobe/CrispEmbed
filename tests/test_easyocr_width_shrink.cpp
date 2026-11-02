// Regression test: EasyOCR recognizer output must not depend on the widths the
// context was built for earlier.
//
// The CUDA audit (PR #98) found that x86 CPU dropped every page line whose
// recognizer canvas was narrower than the previous one. Root cause: both
// BiLSTM layers name their initial state tensors identically, the host reset
// them by name, and ggml_graph_get_tensor() returns only the first match. The
// second layer's h0/c0 were never written, so they held whatever the compute
// buffer held: zero on a fresh allocation, stale data when glibc reused the
// freed block of the larger graph.
//
// The test visits line crops in DECREASING canvas width on one context (each
// width change rebuilds the graph), then recognizes every (image, width) pair
// on a fresh context, and requires identical text and confidence, with
// non-empty text. By default it poisons each compute buffer with NaN bytes
// (CRISPEMBED_EASYOCR_POISON_COMPUTE=1), so an unwritten tensor fails on every
// platform instead of only where the allocator happens to hand back stale
// memory. Set CRISPEMBED_EASYOCR_POISON_COMPUTE=0 to run without poisoning.
//
// usage: test-easyocr-width-shrink <easyocr.gguf> <line.png> [line.png ...]
#include "easyocr_ocr.h"
#include "easyocr_postprocess.h"
#include "core/clean_exit.h"
#include "stb_image.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct image {
    std::string path;
    int w = 0, h = 0, ch = 0;
    std::vector<unsigned char> px;
    int canvas = 0;
};

struct outcome {
    std::string text;
    float confidence = 0.0f;
    bool ok = false;
};

outcome recognize(easyocr_ocr_context * ctx, const image & im, int width) {
    outcome o;
    if (!easyocr_ocr_set_width(ctx, width)) return o;
    int len = 0;
    const char * text = easyocr_ocr_recognize(ctx, im.px.data(), im.w, im.h, im.ch, &len);
    if (!text) return o;
    o.text.assign(text, (size_t)len);
    o.confidence = easyocr_ocr_last_confidence(ctx);
    o.ok = true;
    return o;
}

void set_env_default(const char * name, const char * value) {
    if (std::getenv(name)) return;
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
}

} // namespace

static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <easyocr.gguf> <line.png> [line.png ...]\n", argv[0]);
        return 2;
    }
    set_env_default("CRISPEMBED_EASYOCR_POISON_COMPUTE", "1");
    const char * poison = std::getenv("CRISPEMBED_EASYOCR_POISON_COMPUTE");
    const char * reset = std::getenv("CRISPEMBED_EASYOCR_LSTM_STATE_RESET");
    std::printf("poison=%s lstm_state_reset=%s\n", poison ? poison : "", reset ? reset : "(default)");

    std::vector<image> images;
    for (int i = 2; i < argc; ++i) {
        image im;
        im.path = argv[i];
        unsigned char * px = stbi_load(argv[i], &im.w, &im.h, &im.ch, 3);
        if (!px) {
            std::fprintf(stderr, "cannot read %s\n", argv[i]);
            return 3;
        }
        im.ch = 3;
        im.px.assign(px, px + (size_t)im.w * im.h * 3);
        stbi_image_free(px);
        // Recognizer input height is 64, so this is the pipeline's own canvas.
        im.canvas = easyocr_postprocess::recognizer_canvas_width(im.w, im.h);
        images.push_back(std::move(im));
    }

    // Visit plan: the narrowest line on an oversized canvas, then every line at
    // its own canvas widest first, then the narrowest line at 1.5x and 1x its
    // canvas. Almost every step shrinks the graph, as on a real page.
    std::vector<size_t> order(images.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return images[a].canvas > images[b].canvas; });
    struct step {
        size_t img;
        int width;
    };
    std::vector<step> plan;
    const size_t narrow = order.back();
    const int base = images[narrow].canvas;
    const int widest = images[order.front()].canvas;
    // Start wider than anything below so the first narrow step already shrinks.
    plan.push_back({ narrow, std::max(widest, base) + 256 });
    for (size_t i : order) plan.push_back({ i, images[i].canvas });
    plan.push_back({ narrow, ((base * 3 / 2) + 63) / 64 * 64 });
    plan.push_back({ narrow, base });

    easyocr_ocr_context * shared = easyocr_ocr_init(argv[1], 1);
    if (!shared) return 4;
    std::vector<outcome> reused;
    for (const auto & s : plan) reused.push_back(recognize(shared, images[s.img], s.width));
    easyocr_ocr_free(shared);

    int failures = 0;
    for (size_t k = 0; k < plan.size(); ++k) {
        const auto & s = plan[k];
        easyocr_ocr_context * fresh = easyocr_ocr_init(argv[1], 1);
        if (!fresh) return 5;
        const outcome expected = recognize(fresh, images[s.img], s.width);
        easyocr_ocr_free(fresh);
        const outcome & got = reused[k];
        const bool pass = expected.ok && got.ok && !got.text.empty() && got.text == expected.text &&
                          got.confidence == expected.confidence;
        std::printf("step=%zu image=%s canvas=%d reused=\"%s\" conf=%.6f fresh=\"%s\" conf=%.6f %s\n", k,
                    images[s.img].path.c_str(), s.width, got.text.c_str(), got.confidence, expected.text.c_str(),
                    expected.confidence, pass ? "PASS" : "FAIL");
        if (!pass) ++failures;
    }
    std::printf("easyocr-width-shrink steps=%zu failures=%d %s\n", plan.size(), failures, failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
