// oxr_api_internal.h — private OXR public-API context shared by oxr_api.cpp
// (recognition) and oxr_document_api.cpp (native document route).
#pragma once
#include "crispembed.h"
#include "oxr_document.h"
#include "oxr_ocr.h"

#include <memory>
#include <string>

struct crispembed_oxr_context {
    oxr_ocr_context * model = nullptr;
    std::string json, error;
    // Native document route (CRISPEMBED_OXR_NATIVE_DOC=1): last document and trace.
    std::string document_json, trace_json, rendered;
    oxr_document::result document;
    bool has_document = false;
    // Optional native D-FINE layout provider (layout_detect context, owned via
    // a type-erased deleter so this header needs no layout_detect symbols).
    std::shared_ptr<void> layout_model;
    ~crispembed_oxr_context() { oxr_ocr_free(model); }
};

// One native request through the resident model; returns the result JSON or throws.
std::string oxr_predict_json(crispembed_oxr_context * ctx, const uint8_t * pixels, int width, int height, int channels,
                             const char * prompt, int max_tokens, float repetition_penalty, bool & eos_out,
                             bool & capped_out, std::string & text_out);
