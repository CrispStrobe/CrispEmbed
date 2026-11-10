// Test-only pointer overlays; production profile/recipe admission remains unchanged.
#include "lighton4b_native_context.h"
#include "lighton4b_head_bf16.h"
#include "lighton4b_image_compose.h"
#include "lighton4b_frontend.h"
#include "lighton4b_mixer_partition_policy.h"
#include "core/sha256.h"
#include "core/clean_exit.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
using namespace qwen35_ocr;
static int run(int argc, char ** argv) {
    if (argc != 5) return 2; // Q4, Q8, case-list, output root
    const std::string out = argv[4];
    auto context = lighton4b_native_context::open(argv[1]);
    if (!context) return 3;
    const auto & model = context->model();
    auto base = model.text.decoder;
    auto fields = sensitivity::slots(base);
    if (fields.size() != 248 || base.head != base.embedding || base.head->type != GGML_TYPE_Q8_0) return 4;
    ggml_context * headers = nullptr;
    auto * metadata = gguf_init_from_file(argv[2], { true, &headers });
    if (!metadata || !headers || !lighton4b_gguf_inspection::open(argv[2])) return 5;
    auto * donorctx = ggml_init({ 1024 * 1024, nullptr, true });
    auto * backend = ggml_backend_cpu_init();
    if (!donorctx || !backend) return 6;
    std::map<std::string, ggml_tensor *> donor;
    std::map<std::string, size_t> offsets;
    for (auto & s : fields) {
        auto * old = *s.target;
        auto * original = ggml_get_tensor(headers, s.name.c_str());
        if (!old || std::string(ggml_get_name(old)) != s.name || old->type != GGML_TYPE_Q4_K || !original ||
            original->type != GGML_TYPE_Q8_0)
            return 7;
        for (int d = 0; d < 4; ++d)
            if (old->ne[d] != original->ne[d]) return 8;
        auto * t = ggml_new_tensor_2d(donorctx, GGML_TYPE_Q8_0, original->ne[0], original->ne[1]);
        ggml_set_name(t, s.name.c_str());
        donor[s.name] = t;
        auto id = gguf_find_tensor(metadata, s.name.c_str());
        if (id < 0) return 9;
        offsets[s.name] = gguf_get_data_offset(metadata) + gguf_get_tensor_offset(metadata, id);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(donorctx, backend);
    if (!buffer) return 10;
    std::ifstream expected(out + "/policy-payloads.tsv");
    std::ifstream payload(argv[2], std::ios::binary);
    std::ofstream hashes(out + "/loaded-payloads.tsv");
    for (auto & s : fields) {
        auto * t = donor.at(s.name);
        std::vector<uint8_t> bytes(ggml_nbytes(t));
        payload.seekg(offsets[s.name]);
        payload.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
        if (!payload) return 11;
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        if (std::memcmp(t->data, bytes.data(), bytes.size())) return 12;
        std::string expected_name, q4hash, q8hash;
        if (!(expected >> expected_name >> q4hash >> q8hash) || expected_name != s.name ||
            q4hash != core_sha256::bytes((*s.target)->data, ggml_nbytes(*s.target)) ||
            q8hash != core_sha256::bytes(t->data, ggml_nbytes(t)))
            return 29;
        hashes << s.name << '\t' << core_sha256::bytes((*s.target)->data, ggml_nbytes(*s.target)) << '\t'
               << core_sha256::bytes(t->data, ggml_nbytes(t)) << '\n';
    }
    hashes.close();
    std::string extra;
    if (expected >> extra) return 30;
    auto * tokenmeta = core_gguf::open_metadata(argv[1]);
    lighton4b::frontend_tokenizer tokenizer;
    if (!tokenmeta || !tokenizer.load(tokenmeta)) return 13;
    core_gguf::free_metadata(tokenmeta);
    std::ifstream cases(argv[3]);
    std::string caseid, rgbpath;
    int h, w;
    while (cases >> caseid >> rgbpath >> h >> w) {
        if (caseid != "simple_table" && caseid != "arabic_printed_line") return 14;
        const auto root = out + "/" + caseid;
        auto emit = [&](const std::string & name, const std::vector<float> & x) {
            std::ofstream f(root + "/" + name + ".f32", std::ios::binary);
            f.write(reinterpret_cast<const char *>(x.data()), x.size() * 4);
            if (!f) throw std::runtime_error("emit");
        };
        auto idswrite = [&](const std::string & name, const std::vector<int> & x) {
            std::ofstream f(root + "/" + name + ".ids");
            for (int v : x) f << v << '\n';
        };
        std::ifstream input(rgbpath, std::ios::binary);
        std::vector<uint8_t> rgb((std::istreambuf_iterator<char>(input)), {});
        if (rgb.size() != size_t(h) * w * 3) return 15;
        image_inputs image;
        if (!lighton4b_preprocess_rgb(rgb.data(), h, w, image)) return 16;
        emit("processor.patches", image.patches);
        idswrite("processor.grid", image.grid);
        std::vector<int> ids, mods;
        std::string rendered;
        if (!tokenizer.image_prompt("", image.grid[1] * image.grid[2] / 4, ids, mods, rendered)) return 17;
        idswrite("prompt", ids);
        idswrite("modalities", mods);
        vision_outputs vision;
        if (!lighton4b_image_vision_cpu(model.vision, model.vision_parameters, image, 4, vision)) return 18;
        emit("vision.merged", vision.merged);
        std::vector<float> embedded;
        if (!embed_cpu(base.embedding, ids, embedded)) return 19;
        lighton4b_image_prefill prefill;
        if (!lighton4b_image_prefill_cpu(embedded, mods, image.grid, vision.merged, prefill)) return 20;
        vision = {};
        image.patches.clear();
        embedded.clear();
        for (auto g : { sensitivity::group::q4, sensitivity::group::linear, sensitivity::group::attention,
                        sensitivity::group::mixers, sensitivity::group::mixers_down, sensitivity::group::q8 }) {
            const auto started = std::chrono::steady_clock::now();
            auto weights = base;
            auto targets = sensitivity::slots(weights);
            int changes = 0;
            for (auto & s : targets)
                if (sensitivity::selected(g, s.role)) {
                    *s.target = donor.at(s.name);
                    ++changes;
                }
            if (weights.head != base.head || weights.embedding != base.embedding || weights.norm != base.norm)
                return 21;
            for (size_t i = 0; i < fields.size(); ++i)
                if (*targets[i].target !=
                    (sensitivity::selected(g, fields[i].role) ? donor.at(fields[i].name) : *fields[i].target))
                    return 22;
            lighton4b_decoder_outputs initial;
            if (!lighton4b_prefill_cpu(weights, prefill.hidden.data(), prefill.positions.axes.data(), int(ids.size()),
                                       4, nullptr, initial))
                return 23;
            for (size_t i = 0; i < initial.hidden.size(); ++i)
                emit(std::string(sensitivity::name(g)) + ".layer." + std::to_string(i), initial.hidden[i]);
            std::vector<float> logits;
            if (!project_rows_cpu(weights.head, initial.norm.data() + (ids.size() - 1) * 2560, 2560, 1, 4, 256, logits))
                return 24;
            lighton4b_head_bf16(logits);
            emit(std::string(sensitivity::name(g)) + ".first", logits);
            int first = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
            std::vector<int> prefix{ first };
            if (first != 248046) {
                std::vector<float> x, pos;
                if (!embed_cpu(weights.embedding, { first }, x) ||
                    !cached_positions(initial.cache.tokens, 1, prefill.positions.rope_delta, pos))
                    return 25;
                lighton4b_decoder_outputs next;
                if (!lighton4b_decoder_hidden_cpu(weights, x.data(), pos.data(), 1, 4, &initial.cache, next)) return 26;
                if (!project_rows_cpu(weights.head, next.norm.data(), 2560, 1, 4, 256, logits)) return 27;
                lighton4b_head_bf16(logits);
                prefix.push_back(int(std::max_element(logits.begin(), logits.end()) - logits.begin()));
            }
            idswrite(std::string(sensitivity::name(g)) + ".prefix", prefix);
            std::string decoded;
            if (!tokenizer.decode(prefix, decoded)) return 31;
            std::ofstream(root + "/" + sensitivity::name(g) + ".decoded.txt") << decoded;
            std::ofstream receipt(root + "/" + sensitivity::name(g) + ".txt");
            receipt << changes << ' '
                    << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << '\n';
            std::cout << caseid << ' ' << sensitivity::name(g) << " completed changes=" << changes << std::endl;
        }
    }
    if (!cases.eof()) return 28;
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(donorctx);
    ggml_free(headers);
    gguf_free(metadata);
    return 0;
}
int main(int argc, char ** argv) {
    try {
        core_util::clean_exit(run(argc, argv));
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 99;
    }
}
