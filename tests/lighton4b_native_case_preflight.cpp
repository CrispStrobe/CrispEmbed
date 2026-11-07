#include "core/clean_exit.h"
#include "lighton4b_native_context.h"
#include "lighton4b_image_compose.h"
#include "lighton4b_frontend.h"
#include <fstream>
#include <iterator>
#include <iostream>
using namespace qwen35_ocr;
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 7 || !core_env::on("CRISPEMBED_LIGHTON4B_FROZEN_CASES")) return 2;
    const int height = std::stoi(argv[3]), width = std::stoi(argv[4]);
    const std::string recipe = argv[6], out = argv[5];
    if (height < 1 || width < 1 || height > 4096 || width > 4096) return 2;
    auto header = lighton4b_gguf_inspection::open(argv[1]);
    if (!header) return 3;
    lighton4b_quant_recipe policy;
    const std::string tag = recipe == "q8_0" || recipe == "q4_k" ? "" : recipe;
    if (header->mixed_quant_recipe() != tag || !lighton4b_select_recipe(header->model().text.decoder, tag, policy)) return 3;
    if (recipe == "q8_0" && policy != lighton4b_quant_recipe::q8_0) return 3;
    if (recipe == "q4_k" && policy != lighton4b_quant_recipe::q4_k) return 3;
    if (policy == lighton4b_quant_recipe::q4km_q6_q8head && !header->q6_profile_header()) return 3;
    if (policy == lighton4b_quant_recipe::imatrix_q4km_q6head && !header->imatrix_q4km_profile_header()) return 3;
    if (policy == lighton4b_quant_recipe::upstream_physical_q4km && !header->upstream_physical_profile_header()) return 3;
    if (policy == lighton4b_quant_recipe::gdn_out_q8 && !header->gdn_out_recipe_metadata_matches()) return 3;
    if (policy == lighton4b_quant_recipe::gdn_out_q6 && !header->gdn_out_q6_metadata_matches()) return 3;
    auto * meta = core_gguf::open_metadata(argv[1]);
    lighton4b::frontend_tokenizer tokenizer;
    if (!meta || !tokenizer.load(meta)) return 4;
    core_gguf::free_metadata(meta);
    std::ifstream input(argv[2], std::ios::binary);
    std::vector<uint8_t> rgb((std::istreambuf_iterator<char>(input)), {});
    if (rgb.size() != size_t(height) * width * 3) return 5;
    image_inputs image;
    if (!lighton4b_preprocess_rgb(rgb.data(), height, width, image)) return 6;
    std::vector<int> ids, mods;
    std::string rendered;
    if (!tokenizer.image_prompt("", image.grid[1] * image.grid[2] / 4, ids, mods, rendered)) return 7;
    std::ofstream patches(out + "/processor.patches.f32", std::ios::binary);
    patches.write((const char *)image.patches.data(), image.patches.size() * 4);
    if (!patches) return 8;
    for (auto item : { std::pair<const char *, const std::vector<int> *>{ "prompt", &ids },
                       { "modalities", &mods },
                       { "grid", &image.grid } }) {
        std::ofstream f(out + "/" + item.first + ".ids");
        for (int id : *item.second) f << id << '\n';
        if (!f) return 8;
    }
    std::ofstream(out + "/rendered.txt") << rendered;
    std::cout << "NATIVE_CASE_PREFLIGHT payload_bytes=0 model_forward_calls=0 input_tokens=" << ids.size()
              << " patches=" << image.patches.size() / 1536 << "\n";
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
