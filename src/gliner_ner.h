// gliner_ner.h — GLiNER zero-shot NER via LFM2.5 bidirectional backbone.
//
// Architecture:
//   LFM2.5-350M bidirectional (16 layers: 10 ShortConv + 6 GQA)
//   + layer fusion (attention-weighted sum) + BiLSTM
//   + GLiNER head (span_rep markerV1 + prompt_rep + dot-product scorer)
//
// Usage:
//   void * ctx = gliner_ner_init("gliner-lfm-f32.gguf", 4);
//   gliner_ner_entity * ents;
//   int n = gliner_ner_extract(ctx, "Barack Obama was born in Hawaii",
//               (const char*[]){"person","location"}, 2, 0.5f, &ents);
//   gliner_ner_free(ctx);

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gliner_ner_entity {
    int    start_char;    // character offset in input text
    int    end_char;      // character offset (exclusive)
    const char * text;    // extracted span (owned by ctx, valid until next call)
    const char * label;   // entity label string (owned by ctx, valid until next call)
    float  score;         // confidence [0, 1]
} gliner_ner_entity;

// Load GLiNER model from GGUF.
void * gliner_ner_init(const char * model_path, int n_threads);

// Free all resources.
void gliner_ner_free(void * ctx);

// Extract named entities. Returns count, fills *out_entities with pointer
// to array (owned by ctx, valid until next call or free).
// labels: array of entity type strings (e.g. "person", "organization")
// n_labels: number of entity types
// threshold: confidence threshold (0.0-1.0, recommended 0.5)
int gliner_ner_extract(void * ctx,
                       const char * text,
                       const char ** labels, int n_labels,
                       float threshold,
                       gliner_ner_entity ** out_entities);

#ifdef __cplusplus
}
#endif
