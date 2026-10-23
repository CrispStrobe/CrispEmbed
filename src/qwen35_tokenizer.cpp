#include "qwen35_frontend.h"
#include "core/bpe.h"
#include "core/nfc.h"
#include "core/env_gate.h"
#include <algorithm>
#include <utility>
namespace qwen35_ocr {
namespace {
const char * ovis_template =
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
    "Extract all readable content from the image in natural human reading order and output the result as a single "
    "Markdown document. For charts or images, represent them using an HTML image tag: "
    "<img src=\"images/bbox_{left}_{top}_{right}_{bottom}.jpg\" />, where left, top, right, bottom are bounding box "
    "coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. Preserve the original text without translation or paraphrasing."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
std::string replace_invalid_utf8(const std::string & input) {
    std::string result;
    for (size_t i = 0; i < input.size();) {
        const unsigned char lead = input[i];
        if (lead < 128) {
            result.push_back(input[i++]);
            continue;
        }
        const int count = lead >= 0xc2 && lead <= 0xdf   ? 2
                          : lead >= 0xe0 && lead <= 0xef ? 3
                          : lead >= 0xf0 && lead <= 0xf4 ? 4
                                                         : 0;
        if (!count) {
            result += "\xef\xbf\xbd";
            ++i;
            continue;
        }
        int valid = 1;
        while (valid < count && i + valid < input.size()) {
            const unsigned char c = input[i + valid];
            if (c < 0x80 || c > 0xbf ||
                (valid == 1 && ((lead == 0xe0 && c < 0xa0) || (lead == 0xed && c >= 0xa0) ||
                                (lead == 0xf0 && c < 0x90) || (lead == 0xf4 && c >= 0x90))))
                break;
            ++valid;
        }
        if (valid == count)
            result.append(input, i, count);
        else
            result += "\xef\xbf\xbd";
        i += valid;
    }
    return result;
}
} // namespace

bool frontend_tokenizer::enabled() const {
    return core_env::on("CRISPEMBED_QWEN35_OCR") && (!ovis_ || core_env::on("CRISPEMBED_OVISOCR2_FRONTEND"));
}
bool frontend_tokenizer::is_stop(int id) const {
    return enabled() && std::find(stops_.begin(), stops_.end(), id) != stops_.end();
}
bool frontend_tokenizer::load(gguf_context * meta) {
    *this = {};
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !meta ||
        core_gguf::kv_str(meta, "tokenizer.ggml.pre", "") != "qwen35")
        return false;
    const auto source = core_gguf::kv_str(meta, "general.source.huggingface.repository", "");
    const auto revision = core_gguf::kv_str(meta, "general.source.huggingface.revision", "");
    const bool lighton =
        source == "lightonai/LightOnOCR-3-0.8B" && revision == "36d8636925327c671bd8e479c517a004f7566bd9";
    const bool ovis = source == "ATH-MaaS/OvisOCR2" && revision == "cbdede82e048659b83310476ff807da825f8fe78";
    if (!lighton && !ovis) return false;
    if (ovis && (!core_env::on("CRISPEMBED_OVISOCR2_FRONTEND") ||
                 core_gguf::kv_u32(meta, "qwen35_ocr.format_version", 0) != 2 ||
                 core_gguf::kv_str(meta, "qwen35_ocr.profile", "") != "ovisocr2" ||
                 core_gguf::kv_str(meta, "qwen35_ocr.checkpoint_sha256", "") !=
                     "9270560288656ece5cb3a6989001afcf5af8d223bceed4a423c33a008861d009" ||
                 core_gguf::kv_str(meta, "qwen35_ocr.profile.rendered_prompt", "") != ovis_template ||
                 core_gguf::kv_bool(meta, "qwen35_ocr.native_alias_qualified", true) ||
                 core_gguf::kv_bool(meta, "qwen35_ocr.image_max_qualified", true) ||
                 core_gguf::kv_u32(meta, "qwen35_ocr.image_min_pixels", 0) != 448 * 448 ||
                 core_gguf::kv_u32(meta, "qwen35_ocr.image_max_pixels", 0) != 2880 * 2880 ||
                 core_gguf::kv_u32(meta, "qwen35_ocr.primary_eos_id", 0) != 248046 ||
                 core_gguf::kv_u32(meta, "tokenizer.ggml.eos_token_id", 0) != 248046 ||
                 core_gguf::kv_i32_array(meta, "qwen35_ocr.stop_token_ids") != std::vector<int>{ 248044, 248046 } ||
                 core_gguf::kv_str_array(meta, "qwen35_ocr.missing_source_files") !=
                     std::vector<std::string>{ "processor_config.json", "generation_config.json" }))
        return false;
    const std::string actual_pattern =
        ovis
            ? R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"
            : R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    if (core_gguf::kv_str(meta, "qwen35_ocr.normalizer", "") != "NFC" ||
        core_gguf::kv_str(meta, "qwen35_ocr.pretokenizer_pattern", "") != actual_pattern)
        return false;
    frontend_tokenizer result;
    result.ovis_ = ovis;
    if (ovis) result.image_template_ = ovis_template;
    result.vocab_ = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
    result.added_ = core_gguf::kv_i32_array(meta, "qwen35_ocr.added_token_ids");
    result.special_ = core_gguf::kv_i32_array(meta, "qwen35_ocr.special_token_ids");
    const int generation_eos = int(core_gguf::kv_u32(meta, "qwen35_ocr.generation_eos_id", 0));
    if (generation_eos != (ovis ? 248044 : 248046)) return false;
    result.eos_ = 248046;
    result.stops_ = ovis ? std::vector<int>{ 248044, 248046 } : std::vector<int>{ 248046 };
    const auto merges = core_gguf::kv_str_array(meta, "tokenizer.ggml.merges");
    if (result.vocab_.size() != 248320 || merges.empty() || result.added_.empty() || result.added_.size() > 256 ||
        result.special_.size() > result.added_.size())
        return false;
    if (ovis && (result.vocab_[248044] != "<|endoftext|>" || result.vocab_[248046] != "<|im_end|>")) return false;
    for (size_t i = 0; i < result.vocab_.size(); ++i)
        if (!result.vocab_[i].empty() && !result.tokens_.emplace(result.vocab_[i], int32_t(i)).second) return false;
    for (size_t i = 0; i < merges.size(); ++i)
        if (merges[i].find(' ') == std::string::npos || !result.merges_.emplace(merges[i], int32_t(i)).second)
            return false;
    std::vector<int> seen;
    for (int id : result.added_) {
        if (id < 0 || size_t(id) >= result.vocab_.size() || result.vocab_[id].empty() ||
            std::find(seen.begin(), seen.end(), id) != seen.end())
            return false;
        seen.push_back(id);
    }
    for (int id : result.special_)
        if (std::find(result.added_.begin(), result.added_.end(), id) == result.added_.end()) return false;
    if (result.eos_ != 248046 ||
        std::find(result.special_.begin(), result.special_.end(), result.eos_) == result.special_.end())
        return false;
    for (int stop : result.stops_)
        if (std::find(result.special_.begin(), result.special_.end(), stop) == result.special_.end()) return false;
    for (const auto & pair : std::vector<std::pair<std::string, int>>{
             { "<|image_pad|>", 248056 }, { "<|vision_start|>", 248053 }, { "<|vision_end|>", 248054 } })
        if (result.vocab_[pair.second] != pair.first) return false;
    for (int byte = 0; byte < 256; ++byte) {
        const char raw = char(byte);
        if (!result.tokens_.count(core_bpe::bytes_to_unicode(&raw, 1))) return false;
    }
    *this = std::move(result);
    return true;
}
bool frontend_tokenizer::encode(const std::string & text, std::vector<int> & ids) const {
    ids.clear();
    if (!enabled() || vocab_.empty() || text.size() > 65536) return false;
    for (size_t start = 0; start < text.size();) {
        size_t next = text.size();
        int added = -1;
        for (int id : added_) {
            const size_t found = text.find(vocab_[id], start);
            if (found != std::string::npos &&
                (found < next || (found == next && (added < 0 || vocab_[id].size() > vocab_[added].size())))) {
                next = found;
                added = id;
            }
        }
        const auto ordinary = core_bpe::tokenize_pretokenized(
            tokens_, merges_,
            core_bpe::bytelevel_pretokenize(core_nfc::normalize(replace_invalid_utf8(text.substr(start, next - start))),
                                            1, ovis_ || core_env::on("CRISPEMBED_QWEN35_TOKENIZER_MARKS")));
        ids.insert(ids.end(), ordinary.begin(), ordinary.end());
        if (added < 0) break;
        ids.push_back(added);
        start = next + vocab_[added].size();
    }
    if (ids.size() > 4096) {
        ids.clear();
        return false;
    }
    return true;
}
bool frontend_tokenizer::decode(const std::vector<int> & ids, std::string & text) const {
    text.clear();
    if (!enabled() || vocab_.empty() || ids.size() > 4096) return false;
    for (int id : ids) {
        if (id < 0 || size_t(id) >= vocab_.size()) {
            text.clear();
            return false;
        }
        if (std::find(special_.begin(), special_.end(), id) == special_.end())
            text += core_bpe::unicode_to_bytes(vocab_[id]);
    }
    text = replace_invalid_utf8(text);
    return true;
}
bool frontend_tokenizer::image_prompt(const std::string & prompt, int image_tokens, std::vector<int> & ids,
                                      std::vector<int> & modalities, std::string & rendered) const {
    ids.clear();
    modalities.clear();
    rendered.clear();
    if (!enabled() || image_tokens < 1 || image_tokens > 1024 || (ovis_ && !prompt.empty())) return false;
    rendered = ovis_ ? image_template_
                     : "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" + prompt +
                           "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    std::vector<int> initial;
    if (!encode(rendered, initial) || std::count(initial.begin(), initial.end(), 248056) != 1 ||
        initial.size() + image_tokens - 1 > 4096) {
        rendered.clear();
        return false;
    }
    for (int id : initial) {
        if (id == 248056) {
            ids.insert(ids.end(), image_tokens, id);
            modalities.insert(modalities.end(), image_tokens, 1);
        } else {
            ids.push_back(id);
            modalities.push_back(0);
        }
    }
    return true;
}
} // namespace qwen35_ocr
