// tests/test_kie_pipeline.cpp — KIE pipeline integration test.
//
// Usage:
//   ./test-kie-pipeline <ner_model.gguf> <det_model.gguf> <rec_model.gguf> <image>
//
// Runs the full KIE pipeline: OCR + NER → structured field extraction.
// Prints extracted fields with bounding boxes and confidence scores.

#include "crispembed.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <ner.gguf> <det.gguf> <rec.gguf> <image> [labels]\n", argv[0]);
        fprintf(stderr, "  labels: comma-separated field names (default: total,date,vendor)\n");
        return 1;
    }

    const char * ner_model = argv[1];
    const char * det_model = argv[2];
    const char * rec_model = argv[3];
    const char * image     = argv[4];
    const char * labels_str = argc > 5 ? argv[5] : "total,date,vendor";

    printf("KIE pipeline test\n");
    printf("  NER model: %s\n", ner_model);
    printf("  Det model: %s\n", det_model);
    printf("  Rec model: %s\n", rec_model);
    printf("  Image:     %s\n", image);
    printf("  Labels:    %s\n\n", labels_str);

    printf("Loading KIE pipeline...\n");
    void * ctx = crispembed_kie_init(det_model, rec_model, ner_model, 4);
    if (!ctx) {
        fprintf(stderr, "ERROR: failed to init KIE pipeline\n");
        return 1;
    }

    // Parse comma-separated labels
    const char * labels[32];
    int n_labels = 0;
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s", labels_str);
    char * tok = strtok(buf, ",");
    while (tok && n_labels < 32) {
        while (*tok == ' ') tok++;
        labels[n_labels++] = tok;
        tok = strtok(nullptr, ",");
    }

    printf("Extracting fields...\n\n");
    crispembed_kie_result res = crispembed_kie_extract(
        ctx, image, labels, n_labels, 0.3f);

    printf("OCR: %d regions, confidence=%.2f\n", res.n_ocr_regions, res.ocr_confidence);
    if (res.ocr_text && res.ocr_text[0]) {
        printf("OCR text: %s\n\n", res.ocr_text);
    }

    printf("%d fields extracted:\n", res.n_fields);
    for (int i = 0; i < res.n_fields; i++) {
        printf("  %s = \"%s\"  (score=%.3f, bbox=[%.0f,%.0f,%.0f,%.0f])\n",
               res.fields[i].label, res.fields[i].value,
               res.fields[i].score,
               res.fields[i].x, res.fields[i].y,
               res.fields[i].w, res.fields[i].h);
    }

    crispembed_kie_free(ctx);
    printf("\nDone.\n");
    return 0;
}
