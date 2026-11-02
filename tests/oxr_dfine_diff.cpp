// Stage-by-stage OXR-1.0 D-FINE layout parity against the original PyTorch
// service (tools/oxr_dfine_reference.py dump), plus final-detection and
// same-host speed comparison. Runs layout_detect's dfine architecture.
//
//   CRISPEMBED_OXR=1 CRISPEMBED_ACCEPT_OXR_LICENSE=1 test-oxr-dfine-diff \
//       --model oxr-dfine-f32.gguf --reference ref.gguf --image page.png \
//       [--inject-input] [--threads 4] [--repeat 3] [--max-rel 1e-3] [--box-tol 0.5]
//
// Every stage reports max_abs, mean_abs, max |ref|, rel = max_abs / max|ref|
// and cosine; the earliest stage whose rel exceeds --max-rel is reported as
// the earliest divergence (budgets are unchanged per run; failures are kept).
// --inject-input feeds the reference's own preprocessed tensor, isolating the
// network from the resize. Output: one JSON document on stdout.
#include "crispembed_diff.h"
#include "core/clean_exit.h"
#include "core/python_float_repr.h"
#include "layout_detect.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

extern "C" {
unsigned char * stbi_load(char const * filename, int * x, int * y, int * channels_in_file, int desired_channels);
void stbi_image_free(void * retval_from_stbi_load);
}

namespace {

struct stage_cmp {
    std::string name;
    bool found = false, finite = true;
    size_t count = 0;
    double max_abs = 0, mean_abs = 0, ref_max = 0, rel = 0, cosine = 0;
    size_t worst = 0;
};

std::string fmt(double v) {
    if (!std::isfinite(v)) return "null";
    return core_pyrepr::python_float(v);
}

stage_cmp compare(const std::string & name, const layout_detect::stage_map & native, const crispembed_diff::Ref & ref,
                  bool allow_nonfinite_ref) {
    stage_cmp r;
    r.name = name;
    auto it = native.find(name);
    auto expected = ref.get_f32(name);
    if (it == native.end() || !expected.first || expected.second != it->second.data.size()) return r;
    r.found = true;
    r.count = expected.second;
    const float * a = it->second.data.data();
    const float * b = expected.first;
    double dot = 0, na = 0, nb = 0, sum = 0;
    for (size_t i = 0; i < r.count; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            if (allow_nonfinite_ref && a[i] == b[i]) continue;
            r.finite = false;
            continue;
        }
        const double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > r.max_abs) {
            r.max_abs = d;
            r.worst = i;
        }
        sum += d;
        r.ref_max = std::max(r.ref_max, std::fabs((double)b[i]));
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    r.mean_abs = sum / std::max<size_t>(1, r.count);
    r.rel = r.ref_max > 0 ? r.max_abs / r.ref_max : r.max_abs;
    r.cosine = na > 0 && nb > 0 ? dot / std::sqrt(na * nb) : (na == nb ? 1.0 : 0.0);
    return r;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int run(int argc, char ** argv) {
    std::string model, reference, image, dump_dir;
    int threads = 4, repeat = 3;
    bool inject = false;
    double max_rel = 5e-3, box_tol = 0.5, score_tol = 1e-3;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--model")
            model = next();
        else if (a == "--reference")
            reference = next();
        else if (a == "--image")
            image = next();
        else if (a == "--threads")
            threads = std::stoi(next());
        else if (a == "--repeat")
            repeat = std::stoi(next());
        else if (a == "--max-rel")
            max_rel = std::stod(next());
        else if (a == "--box-tol")
            box_tol = std::stod(next());
        else if (a == "--score-tol")
            score_tol = std::stod(next());
        else if (a == "--inject-input")
            inject = true;
        else if (a == "--dump-dir")
            dump_dir = next();
        else
            throw std::invalid_argument("unknown argument " + a);
    }
    if (model.empty() || image.empty()) throw std::invalid_argument("--model and --image are required");
    layout_detect::context * ctx = nullptr;
    if (!layout_detect::load(&ctx, model.c_str(), threads) || layout_detect::arch(ctx) != layout_detect::arch_id::dfine)
        throw std::runtime_error("D-FINE model did not load (gates, path or architecture)");
    int w = 0, h = 0, c = 0;
    unsigned char * px = stbi_load(image.c_str(), &w, &h, &c, 3);
    if (!px) throw std::runtime_error("cannot decode " + image);
    std::vector<unsigned char> rgb(px, px + (size_t)w * h * 3);
    stbi_image_free(px);

    crispembed_diff::Ref ref;
    const bool have_ref = !reference.empty() && ref.load(reference);
    if (!reference.empty() && !have_ref) throw std::runtime_error("reference could not be loaded");
    if (inject && !have_ref) throw std::invalid_argument("--inject-input needs --reference");

    layout_detect::stage_map stages;
    layout_detect::set_stage_capture(ctx, &stages);
    std::vector<layout_detect::candidate> cands;
    bool ok;
    if (inject) {
        auto in = ref.get_f32("input");
        ok = in.first && in.second == (size_t)3 * 960 * 960 &&
             layout_detect::detect_candidates_preprocessed(ctx, in.first, w, h, cands);
    } else {
        ok = layout_detect::detect_candidates(ctx, rgb.data(), w, h, 3, cands);
    }
    layout_detect::set_stage_capture(ctx, nullptr);
    if (!dump_dir.empty()) {
        // <name>.f32 raw little-endian float32 plus index.json {name: shape (ggml order)}.
        std::string index = "{";
        bool first = true;
        for (const auto & kv : stages) {
            FILE * f = std::fopen((dump_dir + "/" + kv.first + ".f32").c_str(), "wb");
            if (!f) throw std::runtime_error("cannot write dump in " + dump_dir);
            std::fwrite(kv.second.data.data(), sizeof(float), kv.second.data.size(), f);
            std::fclose(f);
            index += std::string(first ? "" : ",") + "\"" + kv.first + "\":[";
            for (size_t d = 0; d < kv.second.shape.size(); d++)
                index += (d ? "," : "") + std::to_string(kv.second.shape[d]);
            index += "]";
            first = false;
        }
        FILE * f = std::fopen((dump_dir + "/index.json").c_str(), "wb");
        if (f) {
            index += "}";
            std::fwrite(index.data(), 1, index.size(), f);
            std::fclose(f);
        }
    }
    if (!ok) throw std::runtime_error(std::string("native inference failed: ") + layout_detect::last_error(ctx));

    std::vector<double> total_ms, pre_ms, enc_ms, dec_ms, post_ms;
    for (int r = 0; r < repeat; r++) {
        std::vector<layout_detect::candidate> again;
        if (!layout_detect::detect_candidates(ctx, rgb.data(), w, h, 3, again))
            throw std::runtime_error("timing run failed");
        const auto t = layout_detect::last_timings(ctx);
        total_ms.push_back(t.total_ms);
        pre_ms.push_back(t.preprocess_ms);
        enc_ms.push_back(t.encoder_ms);
        dec_ms.push_back(t.decoder_ms);
        post_ms.push_back(t.postprocess_ms);
    }

    std::string out = "{\"schema\":\"crispembed.oxr-dfine.diff.v1\",\"image\":\"" + image +
                      "\",\"width\":" + std::to_string(w) + ",\"height\":" + std::to_string(h) +
                      ",\"inject_input\":" + (inject ? "true" : "false") + ",\"threads\":" + std::to_string(threads) +
                      ",\"budgets\":{\"max_rel\":" + fmt(max_rel) + ",\"box_tol_px\":" + fmt(box_tol) +
                      ",\"score_tol\":" + fmt(score_tol) + "}";
    out += ",\"timing_ms\":{\"repeat\":" + std::to_string(repeat) + ",\"total_median\":" + fmt(median(total_ms)) +
           ",\"preprocess_median\":" + fmt(median(pre_ms)) + ",\"encoder_median\":" + fmt(median(enc_ms)) +
           ",\"decoder_median\":" + fmt(median(dec_ms)) + ",\"postprocess_median\":" + fmt(median(post_ms)) + "}";
    bool stage_pass = true, selection_pass = true;
    if (have_ref) {
        std::vector<std::string> order = {
            "input",          "bb.stem",        "bb.stage0",   "bb.stage1",       "bb.stage2",
            "bb.stage3",      "enc.proj0",      "enc.proj1",   "enc.proj2",       "enc.aifi",
            "enc.lat0",       "enc.fpn0",       "enc.lat1",    "enc.fpn1",        "enc.down0",
            "enc.pan0",       "enc.down1",      "enc.pan1",    "dec.memory",      "dec.output_memory",
            "dec.enc_logits", "dec.topk_index", "dec.content", "dec.topk_anchor", "dec.enc_bbox_delta",
            "dec.ref_unact"
        };
        for (int li = 0; li < 4; li++) {
            const std::string p = "dec." + std::to_string(li) + ".";
            for (const char * s : { "qpos", "self", "cross", "gate", "out", "corners" }) order.push_back(p + s);
            if (li == 0) {
                order.push_back("dec.pre_bbox_delta");
                order.push_back("dec.ref_initial");
            }
            order.push_back(p + "bbox");
        }
        for (const char * s : { "dec.logits", "post.labels", "post.scores", "post.boxes" }) order.push_back(s);
        // Query alignment: decoder queries are identified by their selected
        // encoder token. When native and reference select the same token set
        // but order near-tied scores differently, compare per-query stages
        // in reference order (attention is permutation-equivariant).
        bool aligned = false;
        {
            auto ri = ref.get_f32("dec.topk_index");
            auto ni = stages.find("dec.topk_index");
            if (ri.first && ni != stages.end() && ri.second == ni->second.data.size()) {
                const size_t nq = ri.second;
                std::vector<int> perm(nq, -1); // reference position -> native position
                std::map<int, int> pos;
                for (size_t q = 0; q < nq; q++) pos[(int)ni->second.data[q]] = (int)q;
                bool same = pos.size() == nq, permuted = false;
                for (size_t q = 0; same && q < nq; q++) {
                    auto it = pos.find((int)ri.first[q]);
                    if (it == pos.end())
                        same = false;
                    else {
                        perm[q] = it->second;
                        permuted |= it->second != (int)q;
                    }
                }
                if (same && permuted) {
                    for (auto & kv : stages) {
                        const std::string & n = kv.first;
                        const bool per_query = n == "dec.topk_index" || n == "dec.content" || n == "dec.topk_anchor" ||
                                               n == "dec.enc_bbox_delta" || n == "dec.ref_unact" ||
                                               n == "dec.pre_bbox_delta" || n == "dec.ref_initial" ||
                                               n == "dec.logits" || (n.size() > 6 && n[4] >= '0' && n[4] <= '9');
                        if (!per_query || kv.second.data.size() % nq) continue;
                        const size_t row = kv.second.data.size() / nq;
                        std::vector<float> re(kv.second.data.size());
                        for (size_t q = 0; q < nq; q++)
                            std::copy_n(kv.second.data.data() + (size_t)perm[q] * row, row, re.data() + q * row);
                        kv.second.data.swap(re);
                    }
                    aligned = true;
                }
            }
        }
        std::string earliest;
        out += ",\"query_alignment_applied\":" + std::string(aligned ? "true" : "false");
        out += ",\"stages\":[";
        bool topk_equal = true;
        for (size_t i = 0; i < order.size(); i++) {
            const bool nonfinite_ok = order[i] == "dec.topk_anchor" || order[i] == "dec.ref_unact";
            const auto r = compare(order[i], stages, ref, nonfinite_ok);
            if (order[i] == "dec.topk_index") topk_equal = r.found && r.max_abs == 0;
            // post.* are top-k orderings over (query, class): near-tied scores
            // may legitimately swap positions, so they are reported, not
            // budgeted; final detections are gated by oxr_selection below.
            const bool budgeted = order[i].rfind("post.", 0) != 0;
            const bool bad = !r.found || !r.finite || (budgeted && r.rel > max_rel);
            if (bad && earliest.empty()) earliest = order[i];
            out += std::string(i ? "," : "") + "{\"name\":\"" + r.name +
                   "\",\"found\":" + (r.found ? "true" : "false") + ",\"finite\":" + (r.finite ? "true" : "false") +
                   ",\"count\":" + std::to_string(r.count) + ",\"max_abs\":" + fmt(r.max_abs) +
                   ",\"mean_abs\":" + fmt(r.mean_abs) + ",\"ref_max_abs\":" + fmt(r.ref_max) +
                   ",\"rel\":" + fmt(r.rel) + ",\"cosine\":" + fmt(r.cosine) +
                   ",\"worst_index\":" + std::to_string(r.worst) + ",\"budgeted\":" + (budgeted ? "true" : "false") +
                   ",\"within_budget\":" + (bad ? "false" : "true") + "}";
        }
        out += "],\"earliest_divergence\":" + (earliest.empty() ? std::string("null") : "\"" + earliest + "\"") +
               ",\"topk_query_selection_equal\":" + (topk_equal ? "true" : "false");
        stage_pass = earliest.empty();

        // Final candidates, position by position (service order).
        auto labels = ref.get_f32("post.labels"), scores = ref.get_f32("post.scores"),
             boxes = ref.get_f32("post.boxes");
        if (!labels.first || labels.second != cands.size() || !boxes.first || boxes.second != 4 * cands.size())
            throw std::runtime_error("reference post.* missing or sized differently");
        int label_mismatch = 0;
        double box_max = 0, score_max = 0;
        for (size_t k = 0; k < cands.size(); k++) {
            if (cands[k].raw_class != (int)labels.first[k]) label_mismatch++;
            score_max = std::max(score_max, std::fabs((double)cands[k].score - scores.first[k]));
            for (int j = 0; j < 4; j++)
                box_max = std::max(box_max, std::fabs((double)cands[k].box[j] - boxes.first[k * 4 + j]));
        }
        // OXR client selection (score >= 0.5, clip, class-wise NMS 0.6) on both sides.
        std::vector<layout_detect::candidate> rc(cands.size());
        for (size_t k = 0; k < cands.size(); k++) {
            rc[k].raw_class = (int)labels.first[k];
            rc[k].score = scores.first[k];
            for (int j = 0; j < 4; j++) rc[k].box[j] = boxes.first[k * 4 + j];
        }
        const auto mine = layout_detect::oxr_select(cands, w, h, 0.5f, 0.6f);
        const auto theirs = layout_detect::oxr_select(rc, w, h, 0.5f, 0.6f);
        bool same_regions = mine.size() == theirs.size();
        double sel_box = 0, sel_score = 0;
        for (size_t k = 0; same_regions && k < mine.size(); k++) {
            if (mine[k].raw_class != theirs[k].raw_class) same_regions = false;
            const float a[4] = { mine[k].x1, mine[k].y1, mine[k].x2, mine[k].y2 };
            const float b[4] = { theirs[k].x1, theirs[k].y1, theirs[k].x2, theirs[k].y2 };
            for (int j = 0; j < 4; j++) sel_box = std::max(sel_box, std::fabs((double)a[j] - b[j]));
            sel_score = std::max(sel_score, std::fabs((double)mine[k].score - theirs[k].score));
        }
        const bool sel_pass = same_regions && sel_box <= box_tol && sel_score <= score_tol;
        // Identity-matched candidates: for every native candidate scoring >= 0.05,
        // the reference candidate of the same class with the highest IoU.
        int considered = 0, matched = 0;
        double m_box = 0, m_score = 0;
        for (size_t k = 0; k < cands.size(); k++) {
            if (cands[k].score < 0.05f) continue;
            considered++;
            double best_iou = 0;
            size_t best = 0;
            for (size_t j = 0; j < cands.size(); j++) {
                if ((int)labels.first[j] != cands[k].raw_class) continue;
                const float * b = boxes.first + j * 4;
                const float * a = cands[k].box;
                const double iw = std::max(0.0, (double)std::min(a[2], b[2]) - std::max(a[0], b[0]));
                const double ih = std::max(0.0, (double)std::min(a[3], b[3]) - std::max(a[1], b[1]));
                const double inter = iw * ih;
                const double uni =
                    (double)(a[2] - a[0]) * (a[3] - a[1]) + (double)(b[2] - b[0]) * (b[3] - b[1]) - inter;
                const double iou = uni > 0 ? inter / uni : 0;
                if (iou > best_iou) {
                    best_iou = iou;
                    best = j;
                }
            }
            if (best_iou < 0.99) continue;
            matched++;
            m_score = std::max(m_score, std::fabs((double)cands[k].score - scores.first[best]));
            for (int j = 0; j < 4; j++)
                m_box = std::max(m_box, std::fabs((double)cands[k].box[j] - boxes.first[best * 4 + j]));
        }
        out += ",\"matched_candidates_score_ge_0.05\":{\"native\":" + std::to_string(considered) +
               ",\"matched_iou_ge_0.99\":" + std::to_string(matched) + ",\"box_max_abs_px\":" + fmt(m_box) +
               ",\"score_max_abs\":" + fmt(m_score) + "}";
        out += ",\"candidates\":{\"count\":" + std::to_string(cands.size()) +
               ",\"label_mismatches_by_position\":" + std::to_string(label_mismatch) +
               ",\"box_max_abs_px\":" + fmt(box_max) + ",\"score_max_abs\":" + fmt(score_max) + "}";
        out += ",\"oxr_selection\":{\"score_threshold\":0.5,\"nms_iou\":0.6,\"native_count\":" +
               std::to_string(mine.size()) + ",\"reference_count\":" + std::to_string(theirs.size()) +
               ",\"same_labels_in_order\":" + (same_regions ? "true" : "false") +
               ",\"box_max_abs_px\":" + fmt(sel_box) + ",\"score_max_abs\":" + fmt(sel_score) +
               ",\"pass\":" + (sel_pass ? "true" : "false") + ",\"regions\":[";
        for (size_t k = 0; k < mine.size(); k++)
            out += std::string(k ? "," : "") + "{\"label\":\"" + mine[k].raw_label + "\",\"canonical\":\"" +
                   mine[k].label_name + "\",\"score\":" + fmt(mine[k].score) + ",\"bbox\":[" + fmt(mine[k].x1) + "," +
                   fmt(mine[k].y1) + "," + fmt(mine[k].x2) + "," + fmt(mine[k].y2) + "]}";
        out += "],\"reference_regions\":[";
        for (size_t k = 0; k < theirs.size(); k++)
            out += std::string(k ? "," : "") + "{\"label\":\"" + theirs[k].raw_label +
                   "\",\"score\":" + fmt(theirs[k].score) + ",\"bbox\":[" + fmt(theirs[k].x1) + "," +
                   fmt(theirs[k].y1) + "," + fmt(theirs[k].x2) + "," + fmt(theirs[k].y2) + "]}";
        out += "]}";
        selection_pass = sel_pass;
    }
    const bool pass = stage_pass && selection_pass;
    out += ",\"stage_budget_pass\":" + std::string(stage_pass ? "true" : "false") +
           ",\"selection_pass\":" + (selection_pass ? "true" : "false") + ",\"pass\":" + (pass ? "true" : "false") +
           "}";
    std::printf("%s\n", out.c_str());
    layout_detect::free(ctx);
    return pass ? 0 : 2;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        core_util::clean_exit(run(argc, argv));
    } catch (const std::exception & e) {
        std::fprintf(stderr, "oxr_dfine_diff: %s\n", e.what());
        core_util::clean_exit(3);
    }
}
