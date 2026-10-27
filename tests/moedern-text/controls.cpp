// Weight-free test source. Execute on hosted CI only, not on the shared host.
#include "moedern_text_engine_internal.h"
#include "core/clean_exit.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace mc = moedern_text_detail;
static void require(bool ok, const char * text) {
    if (!ok) throw std::runtime_error(text);
}
static void gate(const char * v) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_MOEDERN_COLBERT", v);
#else
    setenv("CRISPEMBED_MOEDERN_COLBERT", v, 1);
#endif
}
static int controls_main(int argc, char ** argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--load") {
            gate("1");
            std::string failure;
            auto * ctx = mc::init(argv[2], 2, failure);
            const bool valid = std::string(argv[3]) == "OK";
            if (valid) {
                require(ctx, "Valid typed fixture rejected");
                mc::free(ctx);
            } else {
                if (ctx) mc::free(ctx);
                require(!ctx && failure.find(argv[3]) != std::string::npos,
                        "Invalid typed fixture was accepted or failed at wrong guard");
            }
            std::cout << "PASS typed loader control " << argv[3] << '\n';
            return 0;
        }
        require(argc == 1, "Unexpected control arguments");
        std::string error;
        for (const auto * g : { "", "0" }) {
            gate(g);
            auto * c = mc::init("must-not-be-opened.gguf", 2, error);
            require(!c && error.find("OFF") != std::string::npos, "Default/literal0 gate failed");
        }
        std::vector<int32_t> ids(32, 4);
        ids[0] = 2;
        ids[1] = 30000;
        ids[2] = 9;
        ids[3] = 3;
        std::vector<uint8_t> mask(32, 0);
        for (int i = 0; i < 4; ++i) mask[i] = 1;
        std::vector<uint8_t> scoring;
        require(mc::scoring_mask(ids, mask, mc::role::query, { 9 }, scoring, error), "Query scoring failed");
        require(scoring == std::vector<uint8_t>(32, 1), "Expansion must score without changing supplied mask");
        require(mask[4] == 0, "Input mask mutated");
        std::vector<int32_t> doc{ 2, 30001, 9, 10, 3 };
        std::vector<uint8_t> ones(5, 1);
        require(mc::scoring_mask(doc, ones, mc::role::document, { 9 }, scoring, error) &&
                    scoring == std::vector<uint8_t>({ 1, 1, 0, 1, 1 }),
                "Document skiplist mixed with attention");
        require(!mc::scoring_mask(doc, ones, mc::role::query, {}, scoring, error), "Role marker mismatch accepted");
        doc[2] = 30002;
        require(!mc::scoring_mask(doc, ones, mc::role::document, {}, scoring, error), "Out-of-range token accepted");
        std::vector<float> all(3 * 128, 0);
        all[0] = 1;
        all[128 + 1] = 1;
        all[256] = -.5f;
        all[256 + 1] = .5f;
        std::vector<float> selected;
        require(mc::select_rows(all, { 1, 0, 1 }, selected) && selected.size() == 256 && selected[128] == -.5f,
                "Scoring slice order differs");
        float score = 0;
        std::vector<float> q(256, 0), d(256, 0);
        q[0] = 1;
        q[129] = 1;
        d[0] = .5f;
        d[1] = .25f;
        d[128] = -.5f;
        d[129] = .75f;
        require(mc::maxsim(q, d, score) && score == 1.25f, "MaxSim must sum query-row maxima");
        require(!mc::maxsim({}, d, score), "Empty MaxSim accepted");
        q[0] = NAN;
        require(!mc::maxsim(q, d, score), "Nonfinite MaxSim accepted");
        std::cout << "PASS default-off/role/three-mask/scoring/MaxSim controls\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

int main(int argc, char ** argv) {
    core_util::clean_exit(controls_main(argc, argv));
}
