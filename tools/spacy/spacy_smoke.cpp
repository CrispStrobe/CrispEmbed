// spacy_smoke — C-ABI-only analysis runner for targets without Python (iOS
// Simulator, Android emulator). Reads one UTF-8 text per line, analyzes each
// with the admitted model and prints one JSON object per line:
//   {"text":..,"tokens":[{"text","tag","pos","morph","lemma","dep","head","sent_start"}],
//    "entities":[[start,end,label],...]}
// tests/spacy_smoke_compare.py compares this output with original spaCy.
//
// usage: spacy_smoke MODEL MODEL_SHA RESOURCES RESOURCES_SHA TEXTS [NER NER_SHA]
// Enables the library through crispembed_spacy_set_enabled (no setenv).
#include "spacy_pipeline.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void json_string(std::string & out, const char * data, size_t length) {
    out += '"';
    for (size_t i = 0; i < length; ++i) {
        const unsigned char c = static_cast<unsigned char>(data[i]);
        if (c == '"' || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c < 0x20) {
            char buffer[8];
            std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
            out += buffer;
        } else {
            out += static_cast<char>(c);
        }
    }
    out += '"';
}

void field(std::string & out, const char * name, const crispembed_spacy_text_view & view) {
    out += '"';
    out += name;
    out += "\":";
    json_string(out, view.data, view.length);
    out += ',';
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 6 && argc != 8) {
        std::fprintf(stderr, "usage: %s MODEL MODEL_SHA RESOURCES RESOURCES_SHA TEXTS [NER NER_SHA]\n", argv[0]);
        return 2;
    }
    const bool ner = argc == 8;
    crispembed_spacy_set_enabled(CRISPEMBED_SPACY_ENABLE_ANALYSIS | (ner ? CRISPEMBED_SPACY_ENABLE_NER : 0u));
    char error[1024] = { 0 };
    crispembed_spacy_context * context =
        crispembed_spacy_create_with_resources(argv[1], argv[2], argv[3], argv[4], error, sizeof(error));
    if (!context) {
        std::fprintf(stderr, "create failed: %s\n", error);
        return 3;
    }
    if (ner && crispembed_spacy_attach_ner(context, argv[6], argv[7], error, sizeof(error)) != CRISPEMBED_SPACY_OK) {
        std::fprintf(stderr, "attach_ner failed: %s\n", error);
        crispembed_spacy_free(context);
        return 3;
    }
    std::ifstream input(argv[5], std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "cannot read %s\n", argv[5]);
        crispembed_spacy_free(context);
        return 3;
    }
    std::string line;
    int status = 0;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        crispembed_spacy_document * document = nullptr;
        if (crispembed_spacy_analyze_utf8(context, line.data(), line.size(), &document, nullptr, nullptr, nullptr,
                                          nullptr, error, sizeof(error)) != CRISPEMBED_SPACY_OK) {
            std::fprintf(stderr, "analyze failed: %s\n", error);
            status = 4;
            break;
        }
        std::string out = "{\"text\":";
        json_string(out, line.data(), line.size());
        out += ",\"tokens\":[";
        const size_t count = crispembed_spacy_document_token_count(document);
        for (size_t i = 0; i < count; ++i) {
            crispembed_spacy_token_view token;
            crispembed_spacy_document_token(document, i, &token);
            out += i ? ",{" : "{";
            field(out, "text", token.text);
            field(out, "tag", token.tag);
            field(out, "pos", token.pos);
            field(out, "morph", token.morph);
            field(out, "lemma", token.lemma);
            field(out, "dep", token.dependency);
            out += "\"head\":" + std::to_string(token.head) + ",\"sent_start\":" +
                   ((token.flags & CRISPEMBED_SPACY_SENTENCE_START) ? "true" : "false") + "}";
        }
        out += "],\"entities\":[";
        const size_t entities = crispembed_spacy_document_entity_count(document);
        for (size_t i = 0; i < entities; ++i) {
            crispembed_spacy_entity_view entity;
            crispembed_spacy_document_entity(document, i, &entity);
            out += i ? ",[" : "[";
            out += std::to_string(entity.token_start) + "," + std::to_string(entity.token_end) + ",";
            json_string(out, entity.label.data, entity.label.length);
            out += "]";
        }
        out += "]}\n";
        std::fwrite(out.data(), 1, out.size(), stdout);
        crispembed_spacy_document_free(document);
    }
    crispembed_spacy_free(context);
    return status;
}
