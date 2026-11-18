// Development-only bridge to the actual header; no replacement cosine scorer.
#include "crispembed_diff.h"

extern "C" int spacy_diff(const char * reference, const char * name, const float * values, size_t count,
                           double * metrics) {
    crispembed_diff::Ref ref;
    if (!ref.load(reference)) return 1;
    const auto report = ref.compare(name, values, count);
    if (!report.found || report.n_elem != count) return 2;
    metrics[0] = report.max_abs;
    metrics[1] = report.mean_abs;
    metrics[2] = report.rms;
    metrics[3] = report.cos_min;
    metrics[4] = report.cos_global;
    metrics[5] = report.mine_norm;
    metrics[6] = report.ref_norm;
    return 0;
}
