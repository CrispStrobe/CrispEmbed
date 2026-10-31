// Shared OTSL helper: every profile against golden outputs.
//
// tests/data/otsl_profiles_golden.tsv holds, per input, the output of the
// MinerU basic and Marker OTSL converters *before* they moved into
// otsl_table.h, plus the pinned mineru-vl-utils (docling) converter output.
// Passing proves the extraction left both callers byte-identical and that the
// upstream profile reproduces the Python helper. The summary also counts how
// often the profiles disagree (their documented semantic differences).
#include "marker_document.h"
#include "mineru_layout.h"
#include "otsl_table.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string unhex(const std::string & h) {
    std::string o;
    for (size_t i = 0; i + 1 < h.size(); i += 2) o.push_back(static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16)));
    return o;
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "tests/data/otsl_profiles_golden.tsv";
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return 2;
    }
    std::string line;
    long n = 0, bad = 0, basic_vs_up = 0, marker_vs_up = 0;
    while (std::getline(in, line)) {
        std::vector<std::string> f;
        std::stringstream ss(line);
        std::string x;
        while (std::getline(ss, x, '\t')) f.push_back(x);
        while (f.size() < 5) f.push_back("");
        const std::string input = unhex(f[0]), basic = unhex(f[1]), marker = unhex(f[2]), warn = unhex(f[3]),
                          upstream = unhex(f[4]);
        ++n;
        std::vector<std::string> w;
        const std::string m = crisp_marker::otsl_to_html(input, w);
        std::string wj;
        for (auto & s : w) wj += s + "\x1e";
        const bool ok = otsl_table::to_html(input, otsl_table::profile::mineru_basic_v1) == basic &&
                        mineru::otsl_to_html(input) == basic && m == marker && wj == warn &&
                        otsl_table::to_html(input, otsl_table::profile::upstream_docling) == upstream;
        if (!ok) {
            if (++bad <= 5) std::fprintf(stderr, "FAIL row %ld: %s\n", n, f[0].c_str());
        }
        basic_vs_up += basic != upstream;
        marker_vs_up += marker != upstream;
    }
    std::printf("test-otsl-table: %ld rows, %ld failures; profile differences vs upstream: basic %ld, marker %ld\n", n,
                bad, basic_vs_up, marker_vs_up);
    return bad || n == 0 ? 1 : 0;
}
