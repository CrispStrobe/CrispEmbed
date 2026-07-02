// tests/test_face_diff.cpp — SCRFD face-detection parity guardrail.
// Usage: ./test-face-diff scrfd.gguf face-ref.gguf [image.png]
//
// Compares crispembed_detect_faces() against an independent insightface-SCRFD reference
// (tools/dump_face_reference.py over det_10g.onnx). The regression signal is (a) the face
// count and (b) the max bbox/landmark pixel error — a graph-scramble regression yields
// wrong/zero detections. We emit a synthetic `detection: cos_min=... max_abs=...` line
// (1.0 pass / 0.0 fail) so tests/regression/run_one.py's diff parser consumes it.

#include "crispembed.h"
#include "core/clean_exit.h"
#include "crispembed_diff.h"
#include <cstdio>
#include <cmath>
#include <vector>

#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"
static int n_pass = 0, n_fail = 0;
static void check(const char * n, bool c) {
    if (c) {
        printf("  %s[PASS]%s %s\n", GREEN, RESET, n);
        n_pass++;
    } else {
        printf("  %s[FAIL]%s %s\n", RED, RESET, n);
        n_fail++;
    }
}

static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s scrfd.gguf ref.gguf [image.png]\n", argv[0]);
        return 1;
    }
    const char * candidates[] = {
        argc > 3 ? argv[3] : nullptr,
        "tests/regression/images/face.png",
        "../tests/regression/images/face.png",
        "../../tests/regression/images/face.png",
    };
    const char * image = nullptr;
    for (const char * c : candidates) {
        if (!c) continue;
        if (FILE * f = fopen(c, "rb")) {
            fclose(f);
            image = c;
            break;
        }
    }
    if (!image) {
        fprintf(stderr, "face image not found\n");
        return 1;
    }

    printf("SCRFD face detection — parity test\n");
    printf("  Model: %s\n  Ref:   %s\n  Image: %s\n\n", argv[1], argv[2], image);

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;
    auto rd = ref.get_f32("detection"); // [ref_n * 15]
    const float * ref_det = rd.first;
    int ref_n = ref_det ? (int)(rd.second / 15) : 0; // 15 floats per face
    check("ref loaded", ref_det != nullptr && ref_n > 0);
    if (!ref_det || ref_n <= 0) return 1;

    crispembed_face_context * ctx = crispembed_face_init(argv[1], 0);
    check("model loads", ctx != nullptr);
    if (!ctx) return 1;

    int n = 0;
    const crispembed_face_detection * dets = crispembed_detect_faces(ctx, image, 0.5f, 640, &n);
    printf("  detected %d face(s) (ref %d)\n", n, ref_n);
    check("face count matches ref", n == ref_n);

    // Match the highest-confidence (primary) detection against ref face 0, compare
    // bbox(4) + conf(1) + landmarks(10) and report the worst coordinate error.
    float max_px = 0.0f, conf_err = 0.0f;
    bool ok = (n == ref_n) && dets != nullptr && n > 0;
    if (ok) {
        int best = 0;
        float bc = dets[0].confidence;
        for (int i = 1; i < n; i++)
            if (dets[i].confidence > bc) {
                bc = dets[i].confidence;
                best = i;
            }
        const auto & d = dets[best];
        float cpp[15] = { d.x,
                          d.y,
                          d.w,
                          d.h,
                          d.confidence,
                          d.landmarks[0],
                          d.landmarks[1],
                          d.landmarks[2],
                          d.landmarks[3],
                          d.landmarks[4],
                          d.landmarks[5],
                          d.landmarks[6],
                          d.landmarks[7],
                          d.landmarks[8],
                          d.landmarks[9] };
        for (int j = 0; j < 15; j++) {
            float e = std::fabs(cpp[j] - ref_det[j]);
            if (j == 4)
                conf_err = e;
            else if (e > max_px)
                max_px = e;
        }
        printf("  bbox/landmark max error: %.2f px   conf error: %.4f\n", max_px, conf_err);
    }
    // 12 px tolerance: letterbox/resize differs slightly between insightface and the engine;
    // a real scramble craters the box hundreds of px off (or n_faces wrong).
    bool pass = ok && max_px < 12.0f && conf_err < 0.10f;
    printf("  detection: cos_min=%.6f max_abs=%.4f  %s\n", pass ? 1.0f : 0.0f, max_px, pass ? "PASS" : "FAIL");
    check("detection within tolerance", pass);

    crispembed_face_free(ctx);
    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
