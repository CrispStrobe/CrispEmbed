// Test-only original-module capture driver. Compile with CRISPEMBED_LAMA_TESTING
// and link the observer archive; this binary is never a production dependency.
#include "lama.h"
#include "core/clean_exit.h"
#include "core/sha256.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
std::string quote(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\')
            out << '\\' << c;
        else if (c < 32)
            out << "\\u" << std::hex << std::setfill('0') << std::setw(4) << unsigned(c) << std::dec;
        else
            out << c;
    }
    out << '"';
    return out.str();
}

int integer(const std::string & value, int minimum, int maximum) {
    if (value.empty() || !std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }))
        throw std::runtime_error("expected a bounded decimal integer");
    const unsigned long parsed = std::stoul(value);
    if (parsed < unsigned(minimum) || parsed > unsigned(maximum))
        throw std::runtime_error("integer outside supported bounds");
    return int(parsed);
}

std::vector<uint8_t> read_bytes(const fs::path & path, size_t expected) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != std::streamoff(expected))
        throw std::runtime_error("input file size differs: " + path.filename().string());
    std::vector<uint8_t> data(expected);
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char *>(data.data()), std::streamsize(data.size())))
        throw std::runtime_error("cannot read fixture: " + path.filename().string());
    return data;
}

void write_bytes(const fs::path & path, const void * data, size_t size) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(static_cast<const char *>(data), std::streamsize(size));
    stream.close();
    if (!stream) throw std::runtime_error("cannot write native capture: " + path.filename().string());
}

void require(bool result, const std::string & error) {
    if (!result) throw std::runtime_error(error.empty() ? "native validation failed" : error);
}

using model_ptr = std::unique_ptr<lama::context, decltype(&lama::free)>;

int execute(int argc, char ** argv) {
    std::map<std::string, std::string> args;
    const std::set<std::string> valid = {
        "--model", "--input-dir", "--output-dir", "--width", "--height", "--threads"
    };
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (!valid.count(key) || i + 1 >= argc || args.count(key))
            throw std::runtime_error("unknown, duplicate, or incomplete option: " + key);
        args[key] = argv[++i];
    }
    for (const char * key : { "--model", "--input-dir", "--output-dir", "--width", "--height" })
        if (!args.count(key)) throw std::runtime_error(std::string("missing option: ") + key);
    const uint16_t endian = 1;
    require(*reinterpret_cast<const uint8_t *>(&endian) == 1, "capture format requires a little-endian host");
    const int width = integer(args["--width"], lama::min_dimension, lama::max_dimension);
    const int height = integer(args["--height"], lama::min_dimension, lama::max_dimension);
    const int threads = args.count("--threads") ? integer(args["--threads"], 1, 64) : 1;
    int pw, ph;
    std::string error;
    require(lama::padded_dimensions(width, height, pw, ph, error), error);
    // Original captures are intentionally bounded. Full-frame quality belongs
    // in a separate hosted panel after these exact architectural checks pass.
    require(size_t(pw) * ph <= 256 * 256, "native stage capture is limited to 65,536 padded pixels");
    const size_t pixels = size_t(width) * height, padded = size_t(pw) * ph;
    const fs::path input_dir(args["--input-dir"]), output_dir(args["--output-dir"]);
    require(fs::is_directory(input_dir), "input directory does not exist");
    require(!fs::exists(output_dir), "output directory must not exist");
    const std::vector<uint8_t> rgb = read_bytes(input_dir / "input.rgb", 3 * pixels);
    const std::vector<uint8_t> mask = read_bytes(input_dir / "mask.gray", pixels);
    const std::vector<uint8_t> expected_input =
        read_bytes(input_dir / "generator.input.f32", 4 * padded * sizeof(float));
    std::vector<float> input(4 * padded), observed(3 * padded), raw(3 * padded), repeat(3 * padded);
    require(lama::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size(), width, height, input.data(),
                                  input.size(), error),
            error);
    require(std::memcmp(input.data(), expected_input.data(), expected_input.size()) == 0,
            "earliest divergence: original versus native generator.input is not byte-exact");
    model_ptr model(lama::init(args["--model"].c_str(), threads, error), lama::free);
    require(bool(model), error);
    require(fs::create_directories(output_dir), "cannot create a new output directory");
    std::vector<std::string> records;
    std::set<std::string> names;
    size_t graph_bytes = 0;
    const lama::test::observer observer = [&](const std::string & name, const float * data, size_t count,
                                              const std::array<int64_t, 4> & shape) {
        require(names.insert(name).second, "duplicate native observed stage: " + name);
        std::ostringstream filename;
        filename << std::setfill('0') << std::setw(4) << records.size() << ".f32";
        write_bytes(output_dir / filename.str(), data, count * sizeof(float));
        std::ostringstream record;
        record << "{\"stage\":" << quote(name) << ",\"file\":" << quote(filename.str()) << ",\"shape\":[" << shape[3]
               << ',' << shape[2] << ',' << shape[1] << ',' << shape[0]
               << "],\"sha256\":" << quote(core_sha256::bytes(data, count * sizeof(float))) << '}';
        records.push_back(record.str());
        return true;
    };
    require(lama::test::run_observed(model.get(), input.data(), input.size(), pw, ph, observed.data(), observed.size(),
                                     observer, graph_bytes, error),
            error);
    require(lama::run(model.get(), input.data(), input.size(), pw, ph, raw.data(), raw.size(), error), error);
    require(std::memcmp(observed.data(), raw.data(), raw.size() * sizeof(float)) == 0,
            "observer retention changed raw generator output");
    require(lama::run(model.get(), input.data(), input.size(), pw, ph, repeat.data(), repeat.size(), error), error);
    require(std::memcmp(raw.data(), repeat.data(), raw.size() * sizeof(float)) == 0,
            "same-shape cached graph changed repeated raw output");
    model.reset();
    model.reset(lama::init(args["--model"].c_str(), threads, error));
    require(bool(model), error);
    require(lama::run(model.get(), input.data(), input.size(), pw, ph, repeat.data(), repeat.size(), error), error);
    require(std::memcmp(raw.data(), repeat.data(), raw.size() * sizeof(float)) == 0,
            "fresh context changed raw output");
    std::vector<uint8_t> output(rgb.size()), processed(rgb.size());
    require(lama::composite_rgb8(raw.data(), raw.size(), rgb.data(), rgb.size(), mask.data(), mask.size(), width,
                                 height, output.data(), output.size(), error),
            error);
    require(lama::process_rgb8(model.get(), rgb.data(), rgb.size(), mask.data(), mask.size(), width, height,
                               processed.data(), processed.size(), error),
            error);
    require(processed == output, "image wrapper composition differs from raw graph composition");
    for (size_t i = 0; i < pixels; ++i)
        if (mask[i] == 0)
            require(std::memcmp(output.data() + 3 * i, rgb.data() + 3 * i, 3) == 0,
                    "composition changed a known pixel");
    write_bytes(output_dir / "output.rgb", output.data(), output.size());
    const std::string model_sha = core_sha256::file(args["--model"]);
    require(model_sha.size() == 64, "cannot hash model provenance");
    std::ostringstream manifest;
    manifest << "{\n  \"schema\":\"crispembed.lama.native.v1\",\n  \"case\":" << quote(input_dir.filename().string())
             << ",\n  \"width\":" << width << ",\n  \"height\":" << height << ",\n  \"padded_width\":" << pw
             << ",\n  \"padded_height\":" << ph << ",\n  \"model_sha256\":" << quote(model_sha)
             << ",\n  \"graph_buffer_bytes\":" << graph_bytes
             << ",\n  \"output_sha256\":" << quote(core_sha256::bytes(output.data(), output.size()))
             << ",\n  \"controls\":{\"input_exact\":true,\"observer_matches_unobserved\":true,"
                "\"repeated_graph_exact\":true,\"fresh_context_exact\":true,\"process_matches_raw\":true,"
                "\"known_pixels_exact\":true},\n  \"records\":[\n";
    for (size_t i = 0; i < records.size(); ++i)
        manifest << "    " << records[i] << (i + 1 == records.size() ? "\n" : ",\n");
    manifest << "  ]\n}\n";
    const std::string serialized = manifest.str();
    write_bytes(output_dir / "manifest.json", serialized.data(), serialized.size());
    std::fprintf(stderr, "Big-LaMa native capture: %zu stages; repeat/fresh/observer/image controls passed\n",
                 records.size());
    return 0;
}

int crispembed_test_main(int argc, char ** argv) {
    try {
        return execute(argc, argv);
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "Big-LaMa native capture failed: %s\n", exception.what());
        std::fprintf(stderr, "Usage: lama-native --model MODEL.gguf --input-dir CASE_DIR --output-dir NEW_DIR "
                             "--width W --height H [--threads N]\n");
        return 1;
    }
}
} // namespace

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
