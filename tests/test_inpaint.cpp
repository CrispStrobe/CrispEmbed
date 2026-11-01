// Model-free contracts for the real tiling/composition and model file boundary.
// Hosted builds execute these; no checkpoint or inference backend is needed.
#include "inpaint.h"
#include "core/clean_exit.h"
#include "core/sha256.h"
#include "core/temp_file.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0, checks = 0;

void check(bool ok, const char * description) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", description);
    }
}

void set_gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_MIGAN", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_MIGAN", value, 1);
    else
        unsetenv("CRISPEMBED_MIGAN");
#endif
}

struct fixture {
    int width, height;
    std::vector<uint8_t> rgb, mask, output;
    fixture(int w, int h) : width(w), height(h), rgb(size_t(w) * h * 3), mask(size_t(w) * h), output(rgb.size(), 213) {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const size_t pixel = size_t(y) * w + x;
                rgb[3 * pixel] = uint8_t(x % 256);
                rgb[3 * pixel + 1] = uint8_t(y % 256);
                rgb[3 * pixel + 2] = uint8_t((3 * x + 5 * y) % 256);
            }
    }
};

struct processor_state {
    int calls = 0;
    int fail_at = 0;
    bool throw_error = false;
    bool origin_color = false;
    const fixture * small = nullptr;
    bool padding_matches = true;
};

bool synthetic_tile(void * opaque, const uint8_t * rgb, const uint8_t * mask, uint8_t * output, std::string & error) {
    auto & state = *static_cast<processor_state *>(opaque);
    ++state.calls;
    if (state.throw_error) throw std::runtime_error("synthetic tile exception");
    if (state.calls == state.fail_at) {
        error = "synthetic tile failure";
        return false;
    }
    if (state.small) {
        const fixture & source = *state.small;
        for (int y = 0; y < inpaint::tile_size; ++y)
            for (int x = 0; x < inpaint::tile_size; ++x) {
                const size_t from =
                    size_t(std::min(y, source.height - 1)) * source.width + std::min(x, source.width - 1);
                const size_t to = size_t(y) * inpaint::tile_size + x;
                state.padding_matches &=
                    mask[to] == source.mask[from] && std::memcmp(rgb + 3 * to, source.rgb.data() + 3 * from, 3) == 0;
            }
    }
    const uint8_t first = state.origin_color ? (rgb[0] == 0 ? 10 : 210) : 17;
    for (size_t i = 0; i < size_t(inpaint::tile_size) * inpaint::tile_size; ++i) {
        output[3 * i] = first;
        output[3 * i + 1] = 91;
        output[3 * i + 2] = 203;
    }
    return true;
}

bool run(fixture & image, processor_state & state, std::string & error) {
    return inpaint::detail::process_tiles_rgb8(synthetic_tile, &state, image.rgb.data(), image.rgb.size(),
                                               image.mask.data(), image.mask.size(), image.width, image.height,
                                               image.output.data(), image.output.size(), error);
}

bool matches_constant(const fixture & image) {
    constexpr std::array<uint8_t, 3> fill = { 17, 91, 203 };
    for (size_t i = 0; i < image.mask.size(); ++i)
        for (size_t channel = 0; channel < 3; ++channel)
            if (image.output[3 * i + channel] != (image.mask[i] ? fill[channel] : image.rgb[3 * i + channel]))
                return false;
    return true;
}

void image_contracts() {
    std::string error;
    {
        fixture image(257, 259);
        processor_state state;
        error = "stale";
        check(run(image, state, error) && error.empty(), "empty mask succeeds and clears prior error");
        check(state.calls == 0 && image.output == image.rgb, "empty mask copies bytes without a model call");
    }
    {
        fixture image(256, 256);
        for (size_t i = 0; i < image.mask.size(); i += 71) image.mask[i] = 255;
        processor_state state;
        check(run(image, state, error), "native256 sparse mask succeeds");
        check(state.calls == 1 && matches_constant(image), "native256 exactly preserves the tile's uint8 output");
    }
    {
        fixture image(17, 23);
        image.mask.front() = image.mask.back() = 255;
        processor_state state;
        state.small = &image;
        check(run(image, state, error), "small edge-mask image succeeds");
        check(state.calls == 1 && state.padding_matches && matches_constant(image),
              "small RGB and mask edge padding matches all256x256 positions");
    }
    {
        fixture image(448, 100);
        const size_t overlap = size_t(50) * image.width + 223;
        image.mask[overlap] = 255;
        processor_state state;
        state.origin_color = true;
        check(run(image, state, error) && state.calls == 2, "both native tiles contribute to an overlap pixel");
        // x223 receives local223 (weight33) and local31 (weight32).
        check(image.output[3 * overlap] == 108 && image.output[3 * overlap + 1] == 91 &&
                  image.output[3 * overlap + 2] == 203,
              "overlap composition has the expected weighted rounded RGB");
        bool preserved = true;
        for (size_t i = 0; i < image.mask.size(); ++i)
            if (!image.mask[i]) preserved &= std::memcmp(image.output.data() + 3 * i, image.rgb.data() + 3 * i, 3) == 0;
        check(preserved, "overlap composition leaves every known RGB byte intact");
    }
    {
        fixture image(640, 640);
        image.mask[0] = 255;
        processor_state state;
        check(run(image, state, error) && state.calls == 1 && matches_constant(image),
              "all tiles outside the mask are skipped");
    }
    // These dimensions exercise full masks, fractional final origins, three
    // intersecting tiles per axis and reuse of every ring-buffer row.
    for (const auto & dimensions : { std::make_pair(1, 1), std::make_pair(255, 257), std::make_pair(257, 257),
                                     std::make_pair(449, 449), std::make_pair(641, 385), std::make_pair(1025, 513) }) {
        fixture image(dimensions.first, dimensions.second);
        std::fill(image.mask.begin(), image.mask.end(), 255);
        processor_state state;
        check(run(image, state, error) && matches_constant(image),
              "all full-mask pixels covered across row-band reuse");
    }
    {
        fixture image(449, 257);
        for (size_t i = 0; i < image.mask.size(); i += 13) image.mask[i] = 255;
        processor_state state;
        const std::vector<uint8_t> original = image.rgb;
        check(inpaint::detail::process_tiles_rgb8(synthetic_tile, &state, image.rgb.data(), image.rgb.size(),
                                                  image.mask.data(), image.mask.size(), image.width, image.height,
                                                  image.rgb.data(), image.rgb.size(), error),
              "in-place output succeeds");
        image.output = image.rgb;
        image.rgb = original;
        check(matches_constant(image), "in-place processing reads original frame and preserves known pixels");
    }
    {
        fixture image(449, 257);
        std::fill(image.mask.begin(), image.mask.end(), 255);
        const std::vector<uint8_t> before = image.output;
        processor_state state;
        state.fail_at = 2;
        check(!run(image, state, error) && error.find("synthetic tile failure") != std::string::npos &&
                  error.find("(192,0)") != std::string::npos,
              "tile failure reports its coordinates and cause");
        check(image.output == before, "late tile failure leaves caller output unchanged");
        state = processor_state{};
        state.throw_error = true;
        check(!run(image, state, error) && image.output == before &&
                  error.find("synthetic tile exception") != std::string::npos,
              "tile exception is converted to an error without partial output");
        state = processor_state{};
        check(run(image, state, error) && matches_constant(image), "a failed image does not poison a subsequent call");
    }
}

void invalid_inputs() {
    fixture image(8, 8);
    image.mask[0] = 255;
    processor_state state;
    std::string error;
    const std::vector<uint8_t> before = image.output;
    auto attempt = [&](const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask, size_t mask_bytes, int w, int h,
                       uint8_t * output, size_t output_bytes) {
        return inpaint::detail::process_tiles_rgb8(synthetic_tile, &state, rgb, rgb_bytes, mask, mask_bytes, w, h,
                                                   output, output_bytes, error);
    };
    check(!attempt(nullptr, image.rgb.size(), image.mask.data(), image.mask.size(), 8, 8, image.output.data(),
                   image.output.size()),
          "null RGB rejects");
    check(!attempt(image.rgb.data(), image.rgb.size(), nullptr, image.mask.size(), 8, 8, image.output.data(),
                   image.output.size()),
          "null mask rejects");
    check(!attempt(image.rgb.data(), image.rgb.size(), image.mask.data(), image.mask.size(), 8, 8, nullptr,
                   image.output.size()),
          "null output rejects");
    check(!attempt(image.rgb.data(), image.rgb.size() - 1, image.mask.data(), image.mask.size(), 8, 8,
                   image.output.data(), image.output.size()),
          "short RGB rejects");
    check(!attempt(image.rgb.data(), image.rgb.size(), image.mask.data(), image.mask.size() + 1, 8, 8,
                   image.output.data(), image.output.size()),
          "long mask rejects");
    check(!attempt(image.rgb.data(), image.rgb.size(), image.mask.data(), image.mask.size(), 8, 8, image.output.data(),
                   image.output.size() - 1),
          "short output rejects");
    for (const auto & dimensions : { std::make_pair(0, 8), std::make_pair(8, -1), std::make_pair(INT_MAX, INT_MAX),
                                     std::make_pair(32769, 1), std::make_pair(32768, 32768) })
        check(!attempt(image.rgb.data(), image.rgb.size(), image.mask.data(), image.mask.size(), dimensions.first,
                       dimensions.second, image.output.data(), image.output.size()),
              "invalid or oversized dimensions reject before pixel access");
    image.mask[0] = 127;
    check(!run(image, state, error) && error.find("binary") != std::string::npos, "nonbinary mask rejects");
    check(state.calls == 0 && image.output == before, "input failures make no model calls and preserve output");
}

void hash_contracts() {
    const char * empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const char * abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const std::string boundary = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    check(core_sha256::bytes(nullptr, 0) == empty, "SHA-256 empty FIPS vector");
    check(core_sha256::bytes("abc", 3) == abc, "SHA-256 abc FIPS vector");
    check(core_sha256::bytes(boundary.data(), boundary.size()) ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          "SHA-256 two-block padding FIPS vector");
    check(core_sha256::bytes(nullptr, 1).empty(), "SHA-256 rejects null nonempty payload");
    const std::string million(1000000, 'a');
    const char * million_hash = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
    check(core_sha256::bytes(million.data(), million.size()) == million_hash, "SHA-256 million-a FIPS vector");

    const std::string path = core_tmp::make_private();
    check(!path.empty(), "hash test has a private temporary file");
    if (path.empty()) return;
    check(core_sha256::file(path) == empty, "SHA-256 hashes an empty file");
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream.write(million.data(), std::streamsize(million.size()));
        check(bool(stream), "hash test file write succeeds");
    }
    check(core_sha256::file(path) == million_hash, "streaming file hash spans multiple64KiB reads");
    set_gate("1");
    std::string error;
    inpaint::context * model = inpaint::init(path.c_str(), 1, error);
    check(!model && error.find("SHA-256 mismatch") != std::string::npos,
          "wrong model file fails its immutable receipt before GGUF loading");
    inpaint::free(model);
    std::remove(path.c_str());
    check(core_sha256::file(path).empty(), "SHA-256 reports missing file");
    std::error_code ec;
    const auto directory = std::filesystem::temp_directory_path(ec);
    check(!ec && core_sha256::file(directory.string()).empty(), "SHA-256 reports file read failure on directory");
    model = inpaint::init(path.c_str(), 1, error);
    check(!model && error.find("cannot read") != std::string::npos, "missing model has an actionable file error");
    inpaint::free(model);
}

void gate_contracts() {
    const char * prior = std::getenv("CRISPEMBED_MIGAN");
    const bool existed = prior != nullptr;
    const std::string saved = prior ? prior : "";
    std::string error;
    for (const char * value : { static_cast<const char *>(nullptr), "", "0" }) {
        set_gate(value);
        inpaint::context * model = inpaint::init(nullptr, 0, error);
        check(!model && error == "MI-GAN requires CRISPEMBED_MIGAN=1", "init gate rejects before model access");
        inpaint::free(model);
        check(!inpaint::process_rgb8(nullptr, nullptr, 0, nullptr, 0, 0, 0, nullptr, 0, error) &&
                  error == "MI-GAN requires CRISPEMBED_MIGAN=1",
              "processing gate rejects before context and pixel access");
    }
    set_gate("1");
    check(!inpaint::process_rgb8(nullptr, nullptr, 0, nullptr, 0, 0, 0, nullptr, 0, error) &&
              error.find("initialized model") != std::string::npos,
          "enabled processing rejects a null context");
    for (int threads : { 0, 65 }) {
        inpaint::context * model = inpaint::init("unused.gguf", threads, error);
        check(!model && error.find("1..64") != std::string::npos, "invalid thread counts reject before model access");
        inpaint::free(model);
    }
    set_gate(existed ? saved.c_str() : nullptr);
}

} // namespace

static int crispembed_test_main() {
    image_contracts();
    invalid_inputs();
    hash_contracts();
    gate_contracts();
    std::printf("INPAINT_CONTRACT {\"checks\":%d,\"failures\":%d,\"passed\":%s}\n", checks, failures,
                failures ? "false" : "true");
    return failures ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
