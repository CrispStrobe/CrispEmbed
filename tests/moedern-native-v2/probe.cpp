// Test-only standalone probe. Raw array/member hashes are added independently
// by tools/moedern-colbert/native.py after the process has exited successfully.
#include "moedern_colbert.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>

namespace mc = moedern_colbert;
static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_MOEDERN_COLBERT", value);
#else
    setenv("CRISPEMBED_MOEDERN_COLBERT", value, 1);
#endif
}
static bool equal(const mc::result & a, const mc::result & b) {
    return a.supplied_mask == b.supplied_mask && a.effective_mask == b.effective_mask &&
           a.vectors.size() == b.vectors.size() &&
           std::memcmp(a.vectors.data(), b.vectors.data(), a.vectors.size() * sizeof(float)) == 0;
}
static void bytes(const std::filesystem::path & file, const void * data, size_t count) {
    std::ofstream out(file, std::ios::binary);
    out.write(static_cast<const char *>(data), std::streamsize(count));
    require(bool(out), "Raw-array write failed");
}
static std::vector<float> read_vectors(const char * file) {
    std::ifstream in(file, std::ios::binary | std::ios::ate);
    const auto n = in.tellg();
    require(in && n > 0 && n <= 512 * 128 * 4 && size_t(n) % (128 * 4) == 0, "Bad bounded scoring array");
    std::vector<float> data(size_t(n) / 4);
    in.seekg(0);
    in.read(reinterpret_cast<char *>(data.data()), n);
    require(bool(in), "Scoring array read failed");
    return data;
}
int main(int argc, char ** argv) {
    try {
        if (argc == 5 && std::string(argv[1]) == "--maxsim") {
            const auto q = read_vectors(argv[2]), d = read_vectors(argv[3]);
            float score = 0;
            require(mc::maxsim(q, d, score), "Actual native MaxSim failed");
            bytes(argv[4], &score, sizeof(score));
            std::cout << "PASS actual native MaxSim " << score << '\n';
            return 0;
        }
        require(argc == 4, "usage: moedern-native-probe MODEL TOKENS OUTPUT");
        std::ifstream in(argv[2]);
        std::string magic, kind, label;
        size_t count = 0, skip_count = 0;
        in >> magic >> kind >> count;
        require(magic == "MOEDERN_TOKENS_V1" && count >= 3 && count <= 512, "Bad bounded token header");
        const auto task = kind == "query" ? mc::role::query : mc::role::document;
        require(kind == "query" || kind == "document", "Bad role");
        std::vector<int32_t> ids(count);
        std::vector<uint8_t> supplied(count);
        in >> label;
        require(label == "ids", "Missing IDs");
        for (auto & x : ids) in >> x;
        in >> label;
        require(label == "mask", "Missing mask");
        for (auto & x : supplied) {
            int n = -1;
            in >> n;
            require(n == 0 || n == 1, "Bad mask");
            x = uint8_t(n);
        }
        in >> label >> skip_count;
        require(label == "skip" && skip_count <= 30002, "Bad skiplist");
        std::vector<int32_t> skip(skip_count);
        for (auto & x : skip) in >> x;
        require(bool(in), "Truncated token record");
        in >> label;
        require(in.eof(), "Trailing token record");
        const std::filesystem::path out(argv[3]);
        require(!std::filesystem::exists(out), "Output must be fresh");
        std::filesystem::create_directories(out);
        std::string error;
        gate("0");
        std::unique_ptr<mc::context, decltype(&mc::free)> disabled(mc::init(argv[1], 2, error), mc::free);
        require(!disabled && error.find("OFF") != std::string::npos, "Literal0 init did not reject");
        gate("1");
        std::unique_ptr<mc::context, decltype(&mc::free)> ctx(mc::init(argv[1], 2, error), mc::free);
        require(bool(ctx), error.c_str());
        mc::result ordinary, observed, repeated;
        auto encode = [&](mc::context * c, mc::result & result, const mc::observer & observer = mc::observer{}) {
            if (!mc::encode_tokens(c, ids.data(), supplied.data(), count, task, result, observer))
                throw std::runtime_error(mc::error(c));
        };
        encode(ctx.get(), ordinary);
        std::ofstream manifest(out / "stages.tsv");
        manifest << "stage\ttype\tshape\tfile\tbytes\n";
        size_t index = 0, total = 0;
        std::set<std::string> stages;
        auto dump = [&](const std::string & stage, const mc::tensor_view & view) {
            require(stages.insert(stage).second, "Duplicate stage");
            size_t elements = 1;
            for (auto n : view.shape) {
                require(n > 0 && n <= 512, "Bad dump dimension");
                elements *= size_t(n);
            }
            require(view.bytes == elements * (view.type == mc::scalar::f32 ? 4 : 8), "Dump bytes/shape differ");
            total += view.bytes;
            require(total <= 128 * 1024 * 1024, "Bounded dump exceeded");
            const std::string filename = std::to_string(index++) + (view.type == mc::scalar::f32 ? ".f32" : ".i64");
            bytes(out / filename, view.data, view.bytes);
            manifest << stage << '\t' << (view.type == mc::scalar::f32 ? "f32" : "i64") << '\t';
            for (size_t i = 0; i < view.shape.size(); ++i) {
                if (i) manifest << ',';
                manifest << view.shape[i];
            }
            manifest << '\t' << filename << '\t' << view.bytes << '\n';
            return bool(manifest);
        };
        encode(ctx.get(), observed, dump);
        require(equal(ordinary, observed), "Ordinary/dump bits differ");
        encode(ctx.get(), repeated);
        require(equal(ordinary, repeated), "Repeat bits differ");
        mc::result rejected;
        require(!mc::encode_tokens(ctx.get(), ids.data(), supplied.data(), count, task, rejected,
                                   [](const std::string &, const mc::tensor_view &) { return false; }),
                "Observer rejection ignored");
        require(rejected.vectors.empty(), "Rejected output leaked stale vectors");
        require(mc::encode_tokens(ctx.get(), ids.data(), supplied.data(), count, task, repeated) &&
                    equal(ordinary, repeated),
                "Observer failure recovery differs");
        gate("0");
        require(!mc::encode_tokens(ctx.get(), ids.data(), supplied.data(), count, task, rejected) &&
                    rejected.vectors.empty(),
                "Gate0 encode did not reject");
        gate("1");
        require(mc::encode_tokens(ctx.get(), ids.data(), supplied.data(), count, task, repeated) &&
                    equal(ordinary, repeated),
                "Gate recovery differs");
        {
            std::unique_ptr<mc::context, decltype(&mc::free)> other(mc::init(argv[1], 1, error), mc::free);
            require(bool(other), error.c_str());
            encode(other.get(), repeated);
            // Threads may change numerical reduction; context isolation is checked
            // by reusing the original after the second owner has been destroyed.
        }
        require(mc::encode_tokens(ctx.get(), ids.data(), supplied.data(), count, task, repeated) &&
                    equal(ordinary, repeated),
                "Other-context destruction invalidated original");
        std::vector<uint8_t> scoring;
        require(mc::scoring_mask(ids, supplied, task, skip, scoring, error), error.c_str());
        std::vector<float> selected;
        require(mc::select_rows(ordinary.vectors, scoring, selected), "Scoring slice failed");
        bytes(out / "all.f32", ordinary.vectors.data(), ordinary.vectors.size() * 4);
        bytes(out / "selected.f32", selected.data(), selected.size() * 4);
        bytes(out / "scoring.u8", scoring.data(), scoring.size());
        std::ofstream(out / "controls.txt") << "ordinary_dump_bit_exact=true\nrepeat_bit_exact=true\nobserver_"
                                               "rejection_recovery=true\ngate0_recovery=true\ncontext_ownership=true\n";
        std::cout << "PASS bounded native case " << kind << " tokens=" << count << " stages=" << index
                  << " selected=" << selected.size() / 128 << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "REJECTED " << e.what() << '\n';
        return 1;
    }
}
