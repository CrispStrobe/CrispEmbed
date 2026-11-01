// Original Big-LaMa reference replay. The model-free --contracts mode never
// downloads/loads weights. Heavy execution belongs on the hosted workflow.
#include "lama.h"
#include "lama_schema.h"
#include "core/clean_exit.h"
#include "core/gguf_loader.h"
#include "core/sha256.h"
#include "crispembed_diff.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CRISPEMBED_LAMA_TESTING
#error Original replay requires the test-only LaMa observer API.
#endif

namespace {
using latent_inpaint::Tensor;
namespace fs = std::filesystem;
using Shapes = std::map<std::string, std::array<int64_t, 3>>;
struct Fixture {
    int width, height;
    const char * rgb_sha256;
    const char * mask_sha256;
};
// Frozen owned input identities from docs/lama-source-contract.json. A bundle
// cannot select new inputs merely by declaring its own matching file hashes.
const std::map<std::string, Fixture> cases{
    { "subtitle_bar",
      { 256, 256, "c695ac3e9c60b2ca5346e9d328684dc516e7e3b7f8c08b478683c3017dfc353b",
        "a63eee585dc8caf6706aba81ebb4f44f2448059db9a454446e9c66966e545f5f" } },
    { "edge_touch",
      { 257, 249, "3abe857fb91421161aeeeeeff8b9ad07130bd8766b4f88cebb780ee744b1fde2",
        "589eabd49200586229a2538256484bfec7d90bf194e4eacf4065fced734a17ec" } },
    { "center",
      { 128, 160, "b71b252dd7068b3746fa13233171143c3403618f6a124cd6d7fe993f983aa9cc",
        "4faf67e30979cb0ca045b6cc929f92b09b20f9c9c6999da6eed173a28e8de0d0" } },
    { "empty",
      { 64, 64, "be3762811f94a14ce464caa2dad6858eeda3814fb3acb54b1490340099eef8d0",
        "ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7" } },
    { "full",
      { 128, 128, "599b2a32b0354047fa3b3ebec3f4fa05293dc0394c433e28cb24e9e5e4f61d39",
        "0fbba07a833d4dcfc7024eaf313661a0ba8f80a05c6d29b8801c612e10e60dee" } },
    { "one_pixel",
      { 63, 65, "61b760a168ee3bc1a49c96c4351bb32b3b896160c6a77724d545ff4973d2d0de",
        "4addb4d0f10b8bec3e6363b27d26c4adbe4381c1607266374dc3af91aab1ec9b" } },
    { "wide",
      { 320, 96, "cc7ac6c83bcbcace73bac0504de91b854aba76471f2c629412e15e959457abd8",
        "68c810741178d9666e2dfcb46a416e134421089640c144ffd9406ec5e6492a49" } },
    { "thin_strip",
      { 256, 256, "dbf1db6f80816d566c4d1facd6eceaf0c6c116b2839ae920c3074f61ec2f7bfb",
        "73b78bb20eabe842de2b1a709f82fd6ad6e85549bf0e00c6d12bbe9a48d4b230" } }
};

void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}
bool digest_valid(const std::string & value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(),
                                             [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_LAMA", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_LAMA", value, 1);
    else
        unsetenv("CRISPEMBED_LAMA");
#endif
}
Shapes shapes(int width, int height) {
    const int w = (width + 7) / 8 * 8, h = (height + 7) / 8 * 8;
    Shapes result{ { "input", { w, h, 4 } },
                   { "prediction", { w, h, 3 } },
                   { "generator.model.23", { w / 8, h / 8, 512 } },
                   { "generator.model.26", { w / 4, h / 4, 256 } },
                   { "generator.model.29", { w / 2, h / 2, 128 } },
                   { "generator.model.32", { w, h, 64 } } };
    for (int block = 1; block <= 4; ++block) {
        const int divisor = 1 << (block - 1), channels = block == 4 ? 128 : 64 << (block - 1);
        result["generator.model." + std::to_string(block) + ".local"] = { w / divisor, h / divisor, channels };
        if (block == 4) result["generator.model.4.global"] = { w / 8, h / 8, 384 };
    }
    for (int block = 5; block <= 22; ++block) {
        const std::string base = "generator.model." + std::to_string(block);
        result[base + ".local"] = { w / 8, h / 8, 128 };
        result[base + ".global"] = { w / 8, h / 8, 384 };
        for (const char * conv : { "conv1", "conv2" }) {
            const std::string name = base + "." + conv;
            result[name + ".local"] = { w / 8, h / 8, 128 };
            result[name + ".global"] = { w / 8, h / 8, 384 };
            result[name + ".ffc.convg2g.fu.rfft"] = { w / 16 + 1, h / 8, 384 };
            result[name + ".ffc.convg2g.fu.spectral"] = { w / 16 + 1, h / 8, 384 };
            result[name + ".ffc.convg2g.fu.irfft"] = { w / 8, h / 8, 192 };
        }
    }
    return result;
}

void verified_file(const fs::path & path, const std::string & digest, uintmax_t maximum) {
    require(digest_valid(digest) && fs::is_regular_file(path) && fs::file_size(path) > 0 &&
                fs::file_size(path) <= maximum,
            "invalid file, digest or byte bound: " + path.string());
    require(core_sha256::file(path.string()) == digest, "file digest differs: " + path.string());
}
std::vector<uint8_t> read_bytes(const fs::path & path, size_t count, const std::string & digest) {
    verified_file(path, digest, count);
    require(fs::file_size(path) == count, "image byte count differs");
    std::vector<uint8_t> result(count);
    std::ifstream stream(path, std::ios::binary);
    require(bool(stream.read(reinterpret_cast<char *>(result.data()), std::streamsize(count))),
            "cannot read image bytes");
    return result;
}
void write_bytes(const fs::path & path, const void * data, size_t bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(static_cast<const char *>(data), std::streamsize(bytes));
    require(bool(stream), "cannot write numerical evidence");
}

struct Metrics {
    double maximum = 0, squared_error = 0, original_squared = 0, native_squared = 0, dot = 0;
};
struct Comparison {
    crispembed_diff::Ref & reference;
    const Shapes & required;
    fs::path output;
    std::set<std::string> seen;
    size_t failed = 0;
    std::string first_failure;
    void compare(const std::string & name, const Tensor & value) {
        const auto expected = required.find(name);
        require(expected != required.end() && seen.insert(name).second,
                "unexpected or duplicated native stage: " + name);
        const auto original = reference.get_f32(name);
        require(latent_inpaint::finite(value) && original.first && original.second == value.v.size() &&
                    expected->second == std::array<int64_t, 3>{ value.w, value.h, value.c },
                "native stage shape/value differs: " + name);
        Metrics metrics;
        for (size_t i = 0; i < value.v.size(); ++i) {
            const double actual = value.v[i], wanted = original.first[i], error = actual - wanted;
            require(std::isfinite(wanted), "nonfinite original reference");
            metrics.maximum = std::max(metrics.maximum, std::abs(error));
            metrics.squared_error += error * error;
            metrics.original_squared += wanted * wanted;
            metrics.native_squared += actual * actual;
            metrics.dot += actual * wanted;
        }
        const double relative = std::sqrt(metrics.squared_error) / std::max(std::sqrt(metrics.original_squared), 1e-12);
        const double denominator = std::sqrt(metrics.original_squared * metrics.native_squared);
        const double cosine = denominator > 0 ? metrics.dot / denominator : (metrics.squared_error == 0 ? 1.0 : 0.0);
        const bool pass = metrics.maximum <= (name == "input"        ? 0.0
                                              : name == "prediction" ? 0.0002
                                                                     : 0.002) &&
                          relative <= 0.0003 && cosine >= 0.99999;
        if (!pass) {
            ++failed;
            if (first_failure.empty()) first_failure = name;
        }
        std::printf("LAMA_STAGE {\"stage\":\"%s\",\"max_abs\":%.9g,\"relative_l2\":%.9g,\"cosine\":%.12g,"
                    "\"native_l2\":%.9g,\"original_l2\":%.9g,\"passed\":%s}\n",
                    name.c_str(), metrics.maximum, relative, cosine, std::sqrt(metrics.native_squared),
                    std::sqrt(metrics.original_squared), pass ? "true" : "false");
        write_bytes(output / (name + ".f32"), value.v.data(), value.v.size() * sizeof(float));
    }
};

bool contracts() {
    const auto required = shapes(257, 249);
    require(required.size() == 227 &&
                required.at("generator.model.5.conv1.ffc.convg2g.fu.rfft") == std::array<int64_t, 3>{ 17, 32, 384 },
            "reference coverage/odd-width contract differs");
    std::string error;
    gate("0");
    std::unique_ptr<lama::context, decltype(&lama::free)> model(lama::init("unused", std::string(64, '0'), 1, error),
                                                                lama::free);
    require(!model && error.find("CRISPEMBED_LAMA") != std::string::npos, "disabled gate did not reject before I/O");
    Tensor output(1, 1, 1);
    output.v[0] = 123;
    require(!lama::run_reference(nullptr, Tensor{}, output, error) && output.v == std::vector<float>{ 123 },
            "null reference call did not preserve destination");
    std::printf("LAMA_CONTRACT {\"required_stages\":227,\"default_off\":true,\"passed\":true}\n");
    return true;
}

int run(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--contracts") return contracts() ? 0 : 1;
    const std::set<std::string> keys{ "--model", "--model-sha256", "--reference", "--reference-sha256", "--output" };
    std::map<std::string, std::string> arguments;
    for (int i = 1; i < argc; i += 2) {
        require(i + 1 < argc && keys.count(argv[i]) && arguments.emplace(argv[i], argv[i + 1]).second,
                "usage: test-lama-diff --model FILE --model-sha256 SHA --reference CASE/reference.gguf "
                "--reference-sha256 SHA --output NEW_DIRECTORY");
    }
    require(arguments.size() == keys.size(), "all model/reference/output arguments are required");
    const fs::path reference_path = arguments.at("--reference"), input_dir = reference_path.parent_path();
    verified_file(reference_path, arguments.at("--reference-sha256"), UINT64_C(512) * 1024 * 1024);
    std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> metadata(
        core_gguf::open_metadata(reference_path.string().c_str()), &core_gguf::free_metadata);
    require(bool(metadata), "cannot read reference metadata");
    auto text = [&](const char * key) { return core_gguf::kv_str(metadata.get(), key, ""); };
    auto integer = [&](const char * key) {
        const int64_t id = gguf_find_key(metadata.get(), key);
        require(id >= 0 && gguf_get_kv_type(metadata.get(), id) == GGUF_TYPE_UINT32, "wrong reference integer type");
        return gguf_get_val_u32(metadata.get(), id);
    };
    require(text("general.architecture") == "big_lama-reference" && integer("lama.reference.schema") == 1 &&
                text("lama.source.commit") == lama::detail::source_commit() &&
                text("lama.contract.sha256") == lama::detail::source_contract_sha256() &&
                text("lama.checkpoint.sha256") == lama::detail::checkpoint_sha256() &&
                text("lama.checkpoint.container") == "official-zip" && text("lama.reference.precision") == "float32" &&
                text("lama.reference.hook_neutrality") == "bit-exact",
            "original reference provenance differs");
    const auto fixture = cases.find(text("lama.reference.case"));
    require(fixture != cases.end(), "unknown reference fixture");
    const int width = fixture->second.width, height = fixture->second.height;
    require(text("lama.reference.rgb.sha256") == fixture->second.rgb_sha256 &&
                text("lama.reference.mask.sha256") == fixture->second.mask_sha256,
            "reference input identity differs from the frozen owned fixture");
    require(integer("lama.reference.width") == uint32_t(width) && integer("lama.reference.height") == uint32_t(height),
            "fixture dimensions differ");
    const auto required = shapes(width, height);
    require(integer("lama.reference.stage_count") == required.size() &&
                size_t(gguf_get_n_tensors(metadata.get())) == required.size(),
            "original stage coverage is incomplete");
    for (const auto & stage : required) {
        const int64_t id = gguf_find_tensor(metadata.get(), stage.first.c_str());
        require(id >= 0 && gguf_get_tensor_type(metadata.get(), id) == GGML_TYPE_F32, "missing/non-F32 original stage");
        const int64_t * ne = gguf_get_tensor_ne(metadata.get(), id);
        require(ne[0] == stage.second[0] && ne[1] == stage.second[1] && ne[2] == stage.second[2] && ne[3] == 1,
                "original stage shape differs: " + stage.first);
    }
    const size_t pixels = size_t(width) * height;
    const auto rgb = read_bytes(input_dir / "input.rgb8", pixels * 3, fixture->second.rgb_sha256);
    const auto mask = read_bytes(input_dir / "fill-mask.u8", pixels, fixture->second.mask_sha256);
    const auto expected_rgb =
        read_bytes(input_dir / "composited.rgb8", pixels * 3, text("lama.reference.composited.sha256"));
    crispembed_diff::Ref reference;
    require(reference.load(reference_path.string()), "cannot read reference tensors");
    const fs::path output = arguments.at("--output");
    require(!fs::exists(output) && fs::create_directories(output), "evidence output must be a new directory");
    gate("1");
    std::string error;
    std::unique_ptr<lama::context, decltype(&lama::free)> model(
        lama::init(arguments.at("--model"), arguments.at("--model-sha256"), 2, error), lama::free);
    require(bool(model), error);
    const Tensor input = lama::detail::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size(), width, height);
    Tensor baseline, prediction;
    require(lama::run_reference(model.get(), input, baseline, error), error);
    Comparison comparison{ reference, required, output, {}, 0, {} };
    lama::set_observer(
        model.get(),
        [](void * opaque, const std::string & name, const Tensor & tensor) {
            static_cast<Comparison *>(opaque)->compare(name, tensor);
        },
        &comparison);
    require(lama::run_reference(model.get(), input, prediction, error), error);
    lama::set_observer(model.get(), nullptr, nullptr);
    require(baseline.v.size() == prediction.v.size() &&
                std::memcmp(baseline.v.data(), prediction.v.data(), baseline.v.size() * sizeof(float)) == 0,
            "native observer changed generated float bytes");
    require(comparison.seen.size() == required.size(), "native stage coverage is incomplete");
    std::vector<uint8_t> actual(rgb.size(), 213);
    require(lama::process_rgb8(model.get(), rgb.data(), rgb.size(), mask.data(), mask.size(), width, height,
                               actual.data(), actual.size(), error),
            error);
    std::vector<uint8_t> composed(rgb.size());
    lama::detail::composite_rgb8(prediction, rgb.data(), rgb.size(), mask.data(), mask.size(), width, height,
                                 composed.data(), composed.size());
    require(actual == composed, "public wrapper differs from observed generator and compositor");
    size_t known_errors = 0, masked_count = 0;
    double absolute = 0, squared = 0;
    int maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!mask[i / 3])
            known_errors += actual[i] != rgb[i] || expected_rgb[i] != rgb[i];
        else {
            const int delta = std::abs(int(actual[i]) - int(expected_rgb[i]));
            maximum = std::max(maximum, delta);
            absolute += delta;
            squared += delta * delta;
            ++masked_count;
        }
    }
    const double mean = masked_count ? absolute / masked_count : 0;
    const double psnr = squared ? 10 * std::log10(255.0 * 255.0 * masked_count / squared) : 999;
    gate("0");
    require(!lama::run_reference(model.get(), input, prediction, error) && prediction.v == baseline.v,
            "runtime disabled gate did not reject transactionally");
    const bool pass = comparison.failed == 0 && known_errors == 0 && maximum <= 1 && mean <= 0.1 && psnr >= 50;
    write_bytes(output / "native.rgb8", actual.data(), actual.size());
    std::vector<uint8_t> difference(actual.size());
    for (size_t i = 0; i < actual.size(); ++i)
        difference[i] = uint8_t(std::min(255, 16 * std::abs(int(actual[i]) - int(expected_rgb[i]))));
    write_bytes(output / "difference-x16.rgb8", difference.data(), difference.size());
    std::printf("LAMA_FINAL {\"case\":\"%s\",\"stages\":%zu,\"failed_stages\":%zu,\"first_failure\":\"%s\","
                "\"known_errors\":%zu,\"masked_max_abs\":%d,\"masked_mean_abs\":%.9g,\"masked_psnr\":%.9g,"
                "\"native_hook_neutrality\":true,\"passed\":%s}\n",
                fixture->first.c_str(), comparison.seen.size(), comparison.failed, comparison.first_failure.c_str(),
                known_errors, maximum, mean, psnr, pass ? "true" : "false");
    return pass ? 0 : 1;
}
} // namespace

int main(int argc, char ** argv) {
    int result = 2;
    try {
        result = run(argc, argv);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "LaMa replay failed: %s\n", error.what());
    }
    core_util::clean_exit(result);
}
