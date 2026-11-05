#include "core/clean_exit.h"
#include "lighton4b_frontend.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <cstdlib>
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 4) return 2;
    auto * meta = core_gguf::open_metadata(argv[1]);
    if (!meta) return 2;
    lighton4b::frontend_tokenizer tokenizer;
    setenv("CRISPEMBED_LIGHTON4B_FRONTEND", "0", 1);
    if (tokenizer.load(meta)) return 3;
    setenv("CRISPEMBED_LIGHTON4B_FRONTEND", "1", 1);
    if (!tokenizer.load(meta)) return 4;
    std::vector<int> ids, modalities;
    std::string rendered;
    if (!tokenizer.image_prompt("", 140, ids, modalities, rendered)) return 5;
    std::ofstream output(std::string(argv[3]) + "/prompt.ids");
    for (auto id : ids) output << id << '\n';
    output.close();
    std::ofstream mods(std::string(argv[3]) + "/modalities.ids");
    for (auto id : modalities) mods << id << '\n';
    mods.close();
    std::ofstream(std::string(argv[3]) + "/rendered.txt") << rendered;
    std::ifstream generated(argv[2]);
    std::vector<int> generated_ids;
    int id;
    while (generated >> id) generated_ids.push_back(id);
    std::string decoded;
    if (!tokenizer.decode(generated_ids, decoded)) return 6;
    std::ofstream(std::string(argv[3]) + "/decoded.txt") << decoded;
    std::ifstream cases(std::string(argv[3]) + "/text-cases.txt");
    std::ofstream case_ids(std::string(argv[3]) + "/text-cases.ids");
    std::string text;
    while (std::getline(cases, text)) {
        std::vector<int> encoded;
        if (!tokenizer.encode(text, encoded)) return 12;
        for (int token : encoded) case_ids << token << ' ';
        case_ids << '\n';
    }
    if (!tokenizer.is_stop(248046) || tokenizer.is_stop(248044)) return 7;
    if (tokenizer.image_prompt("changed", 140, ids, modalities, rendered)) return 8;
    if (tokenizer.image_prompt("", 0, ids, modalities, rendered)) return 9;
    gguf_set_val_u32(meta, "qwen35_ocr.format_version", 4);
    if (tokenizer.load(meta)) return 10;
    gguf_set_val_u32(meta, "qwen35_ocr.format_version", 5);
    gguf_set_val_str(meta, "general.source.huggingface.revision", "wrong");
    if (tokenizer.load(meta)) return 11;
    core_gguf::free_metadata(meta);
    std::cout << "LIGHTON4B_TOKENIZER controls=7 passed=1\n";
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
