// xycut_plus_plus.cpp — see xycut_plus_plus.h for provenance and changes.
// Copyright (c) 2024 PaddlePaddle Authors. Licensed under the Apache License,
// Version 2.0. Translated to C++ for CrispEmbed (modified: see header).
#include "xycut_plus_plus.h"

#include "core/py_str.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace xycut_plus_plus {
namespace {

using bbox_t = std::array<long long, 4>;
const double INF = std::numeric_limits<double>::infinity();

bool in(const std::string & v, std::initializer_list<const char *> set) {
    for (const char * s : set)
        if (v == s) return true;
    return false;
}
bool in(const std::string & v, const std::vector<std::string> & set) {
    return std::find(set.begin(), set.end(), v) != set.end();
}

// ---- numpy generic aquicksort / aheapsort (npysort/quicksort.cpp, heapsort.cpp)
void aheapsort(const long long * v, long long * tosort, long n) {
    long long * a = tosort - 1;
    long i, j, l;
    long long tmp;
    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && v[a[j]] < v[a[j + 1]]) j += 1;
            if (v[tmp] < v[a[j]]) {
                a[i] = a[j];
                i = j;
                j += j;
            } else
                break;
        }
        a[i] = tmp;
    }
    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && v[a[j]] < v[a[j + 1]]) j++;
            if (v[tmp] < v[a[j]]) {
                a[i] = a[j];
                i = j;
                j += j;
            } else
                break;
        }
        a[i] = tmp;
    }
}

void aquicksort(const long long * v, long long * tosort, long num) {
    const long SMALL_QUICKSORT = 15;
    long long * pl = tosort;
    long long * pr = tosort + num - 1;
    long long * stack[128];
    long long ** sptr = stack;
    long long *pm, *pi, *pj, *pk, vi, tmp;
    int depth[128];
    int * psdepth = depth;
    int cdepth = 0;
    for (unsigned long u = (unsigned long)num; u >>= 1;) cdepth++;
    cdepth *= 2;
    long long vp;
    for (;;) {
        if (cdepth < 0) {
            aheapsort(v, pl, long(pr - pl + 1));
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            pm = pl + ((pr - pl) >> 1);
            if (v[*pm] < v[*pl]) std::swap(*pm, *pl);
            if (v[*pr] < v[*pm]) std::swap(*pr, *pm);
            if (v[*pm] < v[*pl]) std::swap(*pm, *pl);
            vp = v[*pm];
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do ++pi;
                while (v[*pi] < vp);
                do --pj;
                while (vp < v[*pj]);
                if (pi >= pj) break;
                std::swap(*pi, *pj);
            }
            pk = pr - 1;
            std::swap(*pi, *pk);
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            } else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl && vp < v[*pk]) *pj-- = *pk--;
            *pj = vi;
        }
    stack_pop:
        if (sptr == stack) break;
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }
    (void)tmp;
}

std::vector<long long> argsort_column(const std::vector<bbox_t> & boxes, int column) {
    std::vector<long long> values(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) values[i] = boxes[i][size_t(column)];
    return numpy_argsort(values);
}

// ---- projection helpers
std::vector<long long> projection_by_bboxes(const std::vector<bbox_t> & boxes, int axis) {
    long long max_length = std::numeric_limits<long long>::min();
    for (const auto & b : boxes) max_length = std::max({ max_length, b[size_t(axis)], b[size_t(axis + 2)] });
    if (max_length < 0) max_length = 0; // np.zeros(negative) would raise; boxes are nonnegative
    std::vector<long long> projection(size_t(max_length), 0);
    const long long n = max_length;
    for (const auto & b : boxes) {
        long long start = b[size_t(axis)], end = b[size_t(axis + 2)];
        if (start < 0) start = std::max(0LL, start + n); // numpy slice normalization
        if (end < 0) end = std::max(0LL, end + n);
        start = std::min(start, n);
        end = std::min(end, n);
        for (long long t = start; t < end; ++t) projection[size_t(t)] += 1;
    }
    return projection;
}

struct intervals {
    std::vector<long long> starts, ends;
};
std::optional<intervals> split_projection_profile(const std::vector<long long> & arr, double min_value,
                                                  double min_gap) {
    std::vector<long long> sig;
    for (size_t i = 0; i < arr.size(); ++i)
        if (double(arr[i]) > min_value) sig.push_back((long long)i);
    if (sig.empty()) return std::nullopt;
    intervals out;
    out.starts.push_back(sig[0]);
    for (size_t i = 0; i + 1 < sig.size(); ++i) {
        if (double(sig[i + 1] - sig[i]) > min_gap) {
            out.ends.push_back(sig[i]);
            out.starts.push_back(sig[i + 1]);
        }
    }
    out.ends.push_back(sig.back() + 1);
    return out;
}

template <typename T> std::vector<T> take(const std::vector<T> & v, const std::vector<long long> & idx) {
    std::vector<T> out;
    out.reserve(idx.size());
    for (long long i : idx) out.push_back(v[size_t(i)]);
    return out;
}

void recursive_yx_cut(const std::vector<bbox_t> & boxes, const std::vector<long long> & indices,
                      std::vector<long long> & res, std::vector<long long> & segment_ids, long long current_id,
                      double min_gap = 1);
void recursive_xy_cut(const std::vector<bbox_t> & boxes, const std::vector<long long> & indices,
                      std::vector<long long> & res, std::vector<long long> & segment_ids, long long current_id,
                      double min_gap = 1);

void recursive_cut(bool y_first, const std::vector<bbox_t> & boxes, const std::vector<long long> & indices,
                   std::vector<long long> & res, std::vector<long long> & segment_ids, long long current_id,
                   double min_gap) {
    const int a1 = y_first ? 1 : 0, a2 = y_first ? 0 : 1;
    const auto order = argsort_column(boxes, a1);
    const auto sorted_boxes = take(boxes, order);
    const auto sorted_indices = take(indices, order);
    const auto first = split_projection_profile(projection_by_bboxes(sorted_boxes, a1), 0, 1);
    if (!first) {
        for (size_t i = 0; i < indices.size(); ++i) segment_ids.push_back(current_id);
        res.insert(res.end(), indices.begin(), indices.end());
        return;
    }
    for (size_t s = 0; s < first->starts.size(); ++s) {
        const long long start = first->starts[s], end = first->ends[s];
        std::vector<bbox_t> chunk;
        std::vector<long long> chunk_indices;
        for (size_t i = 0; i < sorted_boxes.size(); ++i) {
            const long long v = sorted_boxes[i][size_t(a1)];
            if (start <= v && v < end) {
                chunk.push_back(sorted_boxes[i]);
                chunk_indices.push_back(sorted_indices[i]);
            }
        }
        const auto order2 = argsort_column(chunk, a2);
        const auto sorted_chunk = take(chunk, order2);
        const auto sorted_chunk_indices = take(chunk_indices, order2);
        const auto second = split_projection_profile(projection_by_bboxes(sorted_chunk, a2), 0, min_gap);
        if (!second || second->starts.size() == 1) {
            for (size_t i = 0; i < sorted_chunk_indices.size(); ++i) segment_ids.push_back(current_id);
            current_id += 1;
            res.insert(res.end(), sorted_chunk_indices.begin(), sorted_chunk_indices.end());
            continue;
        }
        for (size_t t = 0; t < second->starts.size(); ++t) {
            std::vector<bbox_t> sub;
            std::vector<long long> sub_indices;
            for (size_t i = 0; i < sorted_chunk.size(); ++i) {
                const long long v = sorted_chunk[i][size_t(a2)];
                if (second->starts[t] <= v && v < second->ends[t]) {
                    sub.push_back(sorted_chunk[i]);
                    sub_indices.push_back(sorted_chunk_indices[i]);
                }
            }
            if (y_first)
                recursive_yx_cut(sub, sub_indices, res, segment_ids, current_id);
            else
                recursive_xy_cut(sub, sub_indices, res, segment_ids, current_id);
            current_id += 1;
        }
    }
}
void recursive_yx_cut(const std::vector<bbox_t> & boxes, const std::vector<long long> & indices,
                      std::vector<long long> & res, std::vector<long long> & segment_ids, long long current_id,
                      double min_gap) {
    recursive_cut(true, boxes, indices, res, segment_ids, current_id, min_gap);
}
void recursive_xy_cut(const std::vector<bbox_t> & boxes, const std::vector<long long> & indices,
                      std::vector<long long> & res, std::vector<long long> & segment_ids, long long current_id,
                      double min_gap) {
    recursive_cut(false, boxes, indices, res, segment_ids, current_id, min_gap);
}

std::pair<std::vector<long long>, std::vector<long long>> sort_by_xycut(const std::vector<bbox_t> & boxes,
                                                                        int direction, double min_gap) {
    std::vector<long long> res, seg, idx(boxes.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (long long)i;
    if (direction == 1)
        recursive_yx_cut(boxes, idx, res, seg, 0, min_gap);
    else
        recursive_xy_cut(boxes, idx, res, seg, 0, min_gap);
    return { res, seg };
}

// ---- geometry helpers
double overlap_div_minbox(const bbox_t & b1, const bbox_t & b2) {
    const long long xl = std::max(b1[0], b2[0]), yt = std::max(b1[1], b2[1]);
    const long long xr = std::min(b1[2], b2[2]), yb = std::min(b1[3], b2[3]);
    if (xr <= xl || yb <= yt) return 0.0;
    const long long inter = (xr - xl) * (yb - yt);
    const long long a1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);
    const long long a2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
    const long long m = std::min(a1, a2);
    if (m <= 0) return 0.0;
    return double(inter) / double(m);
}

bool bbox_direction(const bbox_t & b, double ratio = 1.0) {
    return double(b[2] - b[0]) * ratio >= double(b[3] - b[1]);
}

double projection_iou(const bbox_t & a, const bbox_t & m, bool horizontal) {
    long long overlap, width;
    if (horizontal) {
        overlap = std::max(0LL, std::min(a[2], m[2]) - std::max(a[0], m[0]));
        width = std::min(a[2] - a[0], m[2] - m[0]);
    } else {
        overlap = std::max(0LL, std::min(a[3], m[3]) - std::max(a[1], m[1]));
        width = std::min(a[3] - a[1], m[3] - m[1]);
    }
    return width > 0 ? double(overlap) / double(width) : 0.0;
}

double np_median(std::vector<long long> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n % 2) return double(v[n / 2]);
    return (double(v[n / 2 - 1]) + double(v[n / 2])) / 2.0;
}

struct node {
    bbox_t bbox{};
    std::string label, sub_label, layout;
    std::vector<std::pair<int, bbox_t>> title_text;
    std::vector<bbox_t> sub_title, vision_footnote;
    std::optional<double> index, sub_index;
};

struct median_result {
    double median;
    long long count;
    long long page_length;
};
median_result text_median_width(const std::vector<node> & N, const std::vector<int> & list, bool horizontal) {
    std::vector<long long> widths;
    long long page_max = 0, page_min = 0;
    for (int id : list) {
        const auto & b = N[size_t(id)].bbox;
        if (horizontal) {
            if (N[size_t(id)].label == "text") widths.push_back(b[2] - b[0]);
            page_max = std::max(page_max, b[2]);
            page_min = std::min(page_min, b[0]);
        } else {
            if (in(N[size_t(id)].label, { "text", "formula", "algorithm", "reference", "content", "abstract" }))
                widths.push_back(b[3] - b[1]);
            page_max = std::max(page_max, b[3]);
            page_min = std::min(page_min, b[1]);
        }
    }
    if (widths.empty()) return { INF, 0, 0 };
    const double k = np_median(widths);
    long long count = 0;
    const double k01 = k * 0.4;
    for (long long w : widths)
        if (std::fabs(double(w) - k) < k01) ++count;
    return { k, count, page_max - page_min };
}

bool layout_property(std::vector<node> & N, std::vector<int> & blocks, double median_width,
                     const std::vector<std::string> & no_mask_labels, double threshold) {
    std::stable_sort(blocks.begin(), blocks.end(), [&](int a, int b) {
        const auto & A = N[size_t(a)].bbox;
        const auto & B = N[size_t(b)].bbox;
        return std::make_pair(A[0], A[2] - A[0]) < std::make_pair(B[0], B[2] - B[0]);
    });
    std::vector<std::pair<int, std::vector<std::pair<int, double>>>> check_single;
    double page_min_x = INF, page_max_x = 0;
    long long double_area = 0, single_area = 0;
    for (int id : blocks) {
        page_min_x = std::min(page_min_x, double(N[size_t(id)].bbox[0]));
        page_max_x = std::max(page_max_x, double(N[size_t(id)].bbox[2]));
    }
    const double page_width = page_max_x - page_min_x;
    for (size_t i = 0; i < blocks.size(); ++i) {
        node & block = N[size_t(blocks[i])];
        if (!in(block.label, no_mask_labels)) continue;
        long long x_min_i = block.bbox[0];
        const long long x_max_i = block.bbox[2];
        const long long layout_length = x_max_i - x_min_i;
        long long cover_count = 0;
        std::vector<std::pair<int, double>> matches;
        for (size_t j = 0; j < blocks.size(); ++j) {
            const node & other = N[size_t(blocks[j])];
            if (i == j || !in(other.label, no_mask_labels)) continue;
            const long long x_min_j = other.bbox[0], x_max_j = other.bbox[2];
            const long long mmin = std::max(x_min_i, x_min_j), mmax = std::min(x_max_i, x_max_j);
            const double iou = double(mmax - mmin) / double(x_max_j - x_min_j);
            if (iou > 0) {
                cover_count += 1;
                if (iou > threshold) matches.push_back({ int(j), iou });
                x_min_i = mmax;
                if (x_min_i >= x_max_i) break;
            }
        }
        const long long area = (block.bbox[2] - block.bbox[0]) * (block.bbox[3] - block.bbox[1]);
        if ((double(layout_length) > median_width * 1.3 && cover_count >= 2) ||
            double(layout_length) > 0.6 * page_width) {
            block.layout = "double";
            double_area += area;
        } else {
            block.layout = "single";
            check_single.push_back({ int(i), matches });
        }
    }
    for (const auto & item : check_single) {
        if (item.second.empty()) continue;
        const auto [index, iou] = item.second.back();
        node & bi = N[size_t(blocks[size_t(item.first)])];
        const long long area = (bi.bbox[2] - bi.bbox[0]) * (bi.bbox[3] - bi.bbox[1]);
        if (iou > 0.9 && N[size_t(blocks[size_t(index)])].layout == "double") {
            bi.layout = "double";
            double_area += area;
        } else {
            single_area += area;
        }
    }
    return double_area > single_area;
}

struct pre_cut_result {
    bool empty = true;
    std::vector<long long> cuts;
    double median = 0.0;
};
pre_cut_result get_pre_cut(const std::vector<bbox_t> & boxes, int direction, double min_gap) {
    const auto order = argsort_column(boxes, direction);
    const auto sorted = take(boxes, order);
    const auto iv = split_projection_profile(projection_by_bboxes(sorted, direction), 0, 1);
    pre_cut_result out;
    if (!iv) return out;
    out.empty = false;
    long long last_end = -1;
    std::vector<long long> gaps;
    for (size_t s = 0; s < iv->starts.size(); ++s) {
        const long long start = iv->starts[s], end = iv->ends[s];
        if (last_end != -1 && double(start - last_end) >= min_gap) {
            out.cuts.push_back(py_str::ifloordiv(last_end + start, 2));
            gaps.push_back(start - last_end);
        }
        last_end = end;
    }
    out.median = gaps.empty() ? 0.0 : np_median(gaps);
    return out;
}

std::pair<double, std::array<double, 2>> nearest_edge_distance(const bbox_t & input, const bbox_t & match,
                                                               const std::array<double, 4> & weight,
                                                               const std::string & label,
                                                               const std::vector<std::string> & no_mask_labels,
                                                               std::array<double, 2> min_cfg, double tolerance) {
    const double iou = overlap_div_minbox(input, match);
    if (iou > 0 && !in(label, no_mask_labels)) return { 0.0, { 0.0, 0.0 } };
    const double mx = min_cfg[0], my = min_cfg[1];
    const double x1 = double(input[0]), y1 = double(input[1]), x2 = double(input[2]), y2 = double(input[3]);
    const double x1p = double(match[0]), y1p = double(match[1]), x2p = double(match[2]), y2p = double(match[3]);
    int direction_num = 0;
    double dx = INF, dy = INF;
    double d[4] = { INF, INF, INF, INF };
    if (x2 <= x1p) {
        direction_num += 1;
        d[0] = x1p - x2;
        dx = std::fabs(d[0] - mx) <= tolerance ? mx * weight[0] : d[0] * weight[0];
    } else if (x1 >= x2p) {
        direction_num += 1;
        d[1] = x1 - x2p;
        dx = std::fabs(d[1] - mx) <= tolerance ? mx * weight[1] : d[1] * weight[1];
    } else if (iou > 0) {
        d[0] = 0;
        dx = 0;
    }
    if (y2 <= y1p) {
        direction_num += 1;
        d[2] = y1p - y2;
        dy = std::fabs(d[2] - my) <= tolerance ? my * weight[2] : d[2] * weight[2];
        if (in(label, no_mask_labels)) dy = std::max(0.1, dy) * 10;
    } else if (y1 >= y2p) {
        direction_num += 1;
        d[3] = y1 - y2p;
        dy = std::fabs(d[3] - my) <= tolerance ? my * weight[3] : d[3] * weight[3];
    } else if (iou > 0) {
        d[2] = 0;
        dy = 0;
    }
    const std::array<double, 2> cfg = { std::min(d[0], d[1]), std::min(d[2], d[3]) };
    if (direction_num == 2) return { dx + dy, cfg };
    return { std::min(dx, dy), cfg };
}

std::array<double, 4> get_weights(const std::string & label, bool horizontal) {
    if (label == "doc_title")
        return horizontal ? std::array<double, 4>{ 1, 0.1, 0.1, 1 } : std::array<double, 4>{ 0.2, 0.1, 1, 1 };
    if (in(label, { "paragraph_title", "table_title", "abstract", "image", "seal", "chart", "figure" }))
        return { 1, 1, 0.1, 1 };
    return { 1, 1, 1, 0.1 };
}

struct distance_config {
    std::array<double, 2> edge{ INF, INF };
    double up = INF, total = INF;
};

const std::vector<std::string> OUTER_VISION = { "image", "table", "seal", "chart", "figure" };
const std::vector<std::string> OUTER_TITLES = { "doc_title", "paragraph_title" };
const std::vector<std::string> OUTER_VISION_TITLES = { "table_title", "chart_title", "figure_title" };

double nearest_iou_edge_distance(const bbox_t & input_bbox, const bbox_t & match_bbox, const std::string & label,
                                 const std::vector<std::string> & no_mask_labels, double median_width,
                                 const std::vector<std::pair<int, bbox_t>> & title_text,
                                 const std::vector<bbox_t> & sub_title, distance_config & cfg, double tolerance_len) {
    long long x1 = input_bbox[0], y1 = input_bbox[1], x2 = input_bbox[2], y2 = input_bbox[3];
    const long long x1p = match_bbox[0], y2p = match_bbox[3], x2p = match_bbox[2], y1p = match_bbox[1];
    (void)x2p;
    double iou_distance = 0;
    bool h1, h2;
    if (in(label, OUTER_VISION)) {
        h1 = h2 = true;
    } else {
        h1 = bbox_direction(input_bbox);
        h2 = bbox_direction(match_bbox, 3);
    }
    if (h1 != h2 || projection_iou(input_bbox, match_bbox, h1) < 0.01) iou_distance = 1;
    if (label == "doc_title") tolerance_len = std::max(tolerance_len, std::max(1.0, median_width));
    bbox_t adjusted = input_bbox;
    if (!sub_title.empty()) {
        for (const auto & s : sub_title) {
            const long long nx1 = std::min(x1, s[0]), ny1 = std::min(y1, s[1]), nx2 = std::min(x2, s[2]),
                            ny2 = std::max(y2, s[3]);
            x1 = nx1, y1 = ny1, x2 = nx2, y2 = ny2;
        }
        adjusted = { x1, y1, x2, y2 };
    }
    if (!title_text.empty()) {
        for (const auto & t : title_text) {
            const auto & s = t.second;
            long long nx1, ny1, nx2, ny2;
            if (h1) {
                nx1 = std::min(x1, s[0]), ny1 = std::min(y1, s[1]), nx2 = std::min(x2, s[2]), ny2 = std::max(y2, s[3]);
            } else {
                nx1 = std::min(x1, s[0]), ny1 = std::min(y1, s[1]), nx2 = std::max(x2, s[2]), ny2 = std::min(y2, s[3]);
            }
            x1 = nx1, y1 = ny1, x2 = nx2, y2 = ny2;
        }
        adjusted = { x1, y1, x2, y2 };
    }
    const auto weight = get_weights(label, h1);
    if (label == "abstract") tolerance_len *= 2;
    const auto [edge_distance, edge_cfg] =
        nearest_edge_distance(adjusted, match_bbox, weight, label, no_mask_labels, cfg.edge, tolerance_len);
    double up = double(y1p), left = double(x1p);
    if ((in(label, no_mask_labels) || in(label, OUTER_TITLES) || in(label, OUTER_VISION)) && y1 > y2p) {
        up = -double(y2p);
        left = -double(match_bbox[2]);
    }
    const double min_up = cfg.up;
    if (std::fabs(min_up - up) <= tolerance_len) up = min_up;
    const double distance = iou_distance * 1e8 + edge_distance * 1e4 + up * 1 + left * 0.0001;
    if (cfg.total > distance) {
        cfg.edge = edge_cfg;
        cfg.up = up;
        cfg.total = distance;
    }
    return distance;
}

double manhattan(const bbox_t & a, const bbox_t & b) {
    return 1.0 * double(std::llabs(a[0] - b[0])) + 1.0 * double(std::llabs(a[1] - b[1]));
}

void get_sub_category(std::vector<node> & N, std::vector<int> & blocks, std::vector<std::string> & pre_keys,
                      bool & tran, std::vector<long long> & pre_x, bool & has_x, std::vector<long long> & pre_y,
                      bool & has_y) {
    // title_labels = ["doc_title"] (title_text_labels), min_gap_x = min_gap_y = -1, is_only_x = False.
    const std::vector<std::string> title_labels = { "doc_title" };
    const std::vector<std::string> sub_title_labels = { "paragraph_title" };
    const std::vector<std::string> vision_labels = { "image", "table", "chart", "figure" };
    const std::vector<std::string> vision_title_labels = { "figure_title", "chart_title", "table_title" };
    std::vector<std::string> all_labels = title_labels;
    all_labels.insert(all_labels.end(), sub_title_labels.begin(), sub_title_labels.end());
    all_labels.insert(all_labels.end(), vision_labels.begin(), vision_labels.end());
    all_labels.insert(all_labels.end(), vision_title_labels.begin(), vision_title_labels.end());
    const std::vector<std::string> & special_pre_cut_labels = sub_title_labels;
    tran = false;
    has_x = has_y = false;
    if (blocks.empty()) return;

    long long min_x = N[size_t(blocks[0])].bbox[0], min_y = N[size_t(blocks[0])].bbox[1];
    long long max_x = N[size_t(blocks[0])].bbox[2], max_y = N[size_t(blocks[0])].bbox[3];
    for (int id : blocks) {
        const auto & b = N[size_t(id)].bbox;
        min_x = std::min(min_x, b[0]);
        min_y = std::min(min_y, b[1]);
        max_x = std::max(max_x, b[2]);
        max_y = std::max(max_y, b[3]);
    }
    const double region_x_center = double(min_x + max_x) / 2, region_y_center = double(min_y + max_y) / 2;
    const double region_width = double(max_x - min_x), region_height = double(max_y - min_y);

    std::vector<bbox_t> boxes;
    for (int id : blocks) boxes.push_back(N[size_t(id)].bbox);
    const auto xr = get_pre_cut(boxes, 0, -1);
    const auto yr = get_pre_cut(boxes, 1, 70);
    if (xr.empty || yr.empty) throw std::invalid_argument("XY-Cut++: empty projection (unpack of [] in upstream)");
    if (xr.median * 2 < yr.median) {
        tran = true;
        has_y = true;
        pre_y = yr.cuts;
    }
    if (!xr.cuts.empty() && (long long)blocks.size() > -1) {
        has_x = true;
        pre_x = xr.cuts;
    }
    (void)pre_keys;

    for (size_t i = 0; i < blocks.size(); ++i) {
        node & block1 = N[size_t(blocks[i])];
        if (block1.sub_label.empty()) block1.sub_label = block1.label; // setdefault
        if (!in(block1.label, all_labels)) continue;
        const bbox_t bbox1 = block1.bbox;
        const long long x1 = bbox1[0], y1 = bbox1[1], x2 = bbox1[2], y2 = bbox1[3];
        const bool h1 = bbox_direction(bbox1);
        double lu_dist = INF, rd_dist = INF;
        int lu_index = -1, rd_index = -1;
        int lu_dir = -1, rd_dir = -1; // -1 = None
        long long block_length;
        double required_length;
        if (h1) {
            block_length = x2 - x1;
            required_length = region_width / 2;
        } else {
            block_length = y2 - y1;
            required_length = region_height / 2;
        }
        const bool length_condition =
            in(block1.label, special_pre_cut_labels) ? true : double(block_length) > required_length;
        const double bxc = double(x1 + x2) / 2, byc = double(y1 + y2) / 2;
        long long tolerance = py_str::ifloordiv(block_length, 5);
        if (in(block1.label, special_pre_cut_labels)) tolerance = py_str::ifloordiv(block_length, 10);
        const bool centered = h1 ? std::fabs(bxc - region_x_center) <= double(tolerance)
                                 : std::fabs(byc - region_y_center) <= double(tolerance);
        bool left = false, right = false, above = false, below = false;
        for (int id2 : blocks) {
            const node & b2 = N[size_t(id2)];
            if (!(in(b2.label, { "text", "formula", "content", "abstract", "reference" }) ||
                  in(b2.label, title_labels) || in(b2.label, sub_title_labels) || in(b2.label, vision_title_labels)))
                continue;
            const auto & q = b2.bbox;
            if (h1) {
                if (q[2] <= x1 && !(q[3] <= y1 || q[1] >= y2)) left = true;
                if (q[0] >= x2 && !(q[3] <= y1 || q[1] >= y2)) right = true;
            } else {
                if (q[3] <= y1 && !(q[2] <= x1 || q[0] >= x2)) above = true;
                if (q[1] >= y2 && !(q[2] <= x1 || q[0] >= x2)) below = true;
            }
            if ((h1 && left && right) || (!h1 && above && below)) break;
        }
        const bool no_text_on_sides = h1 ? !(left || right) : !(above || below);
        if (centered && length_condition && no_text_on_sides) {
            if (h1) {
                has_y = true;
                pre_y.push_back(y1);
            } else {
                has_x = true;
                pre_x.push_back(x1);
            }
        }
        double block_iou_threshold = 0.1;
        if (in(block1.label, sub_title_labels)) block_iou_threshold = 0.5;
        bool lu_flag = false, rd_flag = false;
        for (size_t j = 0; j < blocks.size(); ++j) {
            if (i == j) continue;
            const bbox_t bbox2 = N[size_t(blocks[j])].bbox;
            const long long x1p = bbox2[0], y1p = bbox2[1], x2p = bbox2[2], y2p = bbox2[3];
            const bool h2 = bbox_direction(bbox2);
            const double match_iou = projection_iou(bbox2, bbox1, h1);
            auto distance_ = [&](bool horizontal, bool left_up) -> double {
                if (match_iou > 0) {
                    if (horizontal) {
                        if (left_up) return double(py_str::ifloordiv(y1 - y2p + 2, 5)) + double(x1p) / 5000;
                        return double(py_str::ifloordiv(y1p - y2 + 2, 5)) + double(x1p) / 5000;
                    }
                    if (left_up) return double(py_str::ifloordiv(x1 - x2p + 2, 5)) + double(y1p) / 5000;
                    return double(py_str::ifloordiv(x1p - x2 + 2, 5)) + double(y1p) / 5000;
                }
                return INF;
            };
            if (h1) {
                const double lu = distance_(true, true), rd = distance_(true, false);
                if (y2p <= y1 && lu <= lu_dist) {
                    lu_index = int(j);
                    lu_dist = lu;
                    lu_dir = h2;
                    lu_flag = !(match_iou < block_iou_threshold);
                } else if (y1p > y2 && rd < rd_dist) {
                    rd_dist = rd;
                    rd_index = int(j);
                    rd_dir = h2;
                    rd_flag = !(match_iou < block_iou_threshold);
                }
            } else {
                const double lu = distance_(false, true), rd = distance_(false, false);
                if (x2p <= x1 && lu <= lu_dist) {
                    lu_dist = lu;
                    lu_index = int(j);
                    lu_dir = h2;
                    lu_flag = !(match_iou < block_iou_threshold);
                } else if (x1p > x2 && rd < rd_dist) {
                    rd_dist = rd;
                    rd_index = int(j);
                    rd_dir = h2;
                    rd_flag = !(match_iou < block_iou_threshold);
                }
            }
        }
        const double height = double(bbox1[3] - bbox1[1]), width = double(bbox1[2] - bbox1[0]);
        const double ttw[2] = { 0.8, 0.8 };
        std::vector<std::pair<int, bbox_t>> title_text;
        std::vector<bbox_t> sub_title, vision_footnote;
        auto py_at = [&](int index) -> node & {
            return N[size_t(blocks[size_t(index < 0 ? int(blocks.size()) + index : index)])];
        };
        auto sub_category_ = [&](int direction, int index, const std::string & label, bool left_up) {
            const int d0 = left_up ? 1 : 3, d1 = left_up ? 2 : 4;
            const int dir_[2] = { left_up ? 1 : 2, left_up ? 3 : 4 };
            (void)d0, (void)d1;
            if (direction == int(h1) && index != -1 && (label == "text" || label == "paragraph_title")) {
                node & target = py_at(index);
                const bbox_t bbox2 = target.bbox;
                const double height1 = double(bbox2[3] - bbox2[1]), width1 = double(bbox2[2] - bbox2[0]);
                const std::array<double, 4> unit = { 1.0, 1.0, 1.0, 1.0 };
                const std::vector<std::string> none;
                if (h1) {
                    if (label == "text") {
                        if (nearest_edge_distance(bbox1, bbox2, unit, "text", none, { INF, INF }, 10.0).first <= 15 &&
                            in(block1.label, vision_labels) && width1 < width && height1 < 0.5 * height) {
                            target.sub_label = "vision_footnote";
                            vision_footnote.push_back(bbox2);
                        } else if (height1 < height * ttw[0] && (width1 < width || width1 > 1.5 * width) &&
                                   in(block1.label, title_labels)) {
                            target.sub_label = "title_text";
                            title_text.push_back({ dir_[0], bbox2 });
                        }
                    } else if (label == "paragraph_title" && in(block1.label, sub_title_labels)) {
                        sub_title.push_back(bbox2);
                    }
                } else {
                    if (label == "text") {
                        if (nearest_edge_distance(bbox1, bbox2, unit, "text", none, { INF, INF }, 10.0).first <= 15 &&
                            in(block1.label, vision_labels) && height1 < height && width1 < 0.5 * width) {
                            target.sub_label = "vision_footnote";
                            vision_footnote.push_back(bbox2);
                        } else if (width1 < width * ttw[1] && in(block1.label, title_labels)) {
                            target.sub_label = "title_text";
                            title_text.push_back({ dir_[1], bbox2 });
                        }
                    } else if (label == "paragraph_title" && in(block1.label, sub_title_labels)) {
                        sub_title.push_back(bbox2);
                    }
                }
            }
        };
        if (!rd_flag) rd_index = -1;
        if (!lu_flag) lu_index = -1;
        const double diff5 = std::fabs(lu_dist - rd_dist) * 5;
        if ((h1 && diff5 > height) || (!h1 && diff5 > width)) {
            if (lu_dist < rd_dist)
                sub_category_(lu_dir, lu_index, py_at(lu_index).label, true);
            else
                sub_category_(rd_dir, rd_index, py_at(rd_index).label, false);
        } else {
            sub_category_(lu_dir, lu_index, py_at(lu_index).label, true);
            sub_category_(rd_dir, rd_index, py_at(rd_index).label, false);
        }
        node & self = N[size_t(blocks[i])];
        if (in(block1.label, title_labels) && self.title_text.empty()) self.title_text = title_text;
        if (in(block1.label, sub_title_labels) && self.sub_title.empty()) self.sub_title = sub_title;
        if (in(block1.label, vision_labels) && self.vision_footnote.empty()) self.vision_footnote = vision_footnote;
    }
}

std::vector<int> projection_index_xy(std::vector<node> & N, std::vector<int> blocks) {
    std::stable_sort(blocks.begin(), blocks.end(), [&](int a, int b) {
        const node &A = N[size_t(a)], &B = N[size_t(b)];
        return std::make_tuple(A.index.value(), A.bbox[1], A.bbox[0]) <
               std::make_tuple(B.index.value(), B.bbox[1], B.bbox[0]);
    });
    std::vector<int> sorted;
    size_t i = 0;
    while (i < blocks.size()) {
        const node & b1 = N[size_t(blocks[i])];
        std::vector<int> mid = { blocks[i] };
        for (size_t j = i + 1; j < blocks.size(); ++j) {
            const node & b2 = N[size_t(blocks[j])];
            if (b1.index.value() != b2.index.value()) break;
            if (projection_iou(b1.bbox, b2.bbox, false) > 0.5)
                mid.push_back(blocks[j]);
            else
                break;
        }
        std::stable_sort(mid.begin(), mid.end(),
                         [&](int a, int b) { return N[size_t(a)].bbox[0] < N[size_t(b)].bbox[0]; });
        sorted.insert(sorted.end(), mid.begin(), mid.end());
        i += mid.size();
    }
    return sorted;
}

std::vector<bbox_t> projection_xy(const std::vector<bbox_t> & bboxes, const std::vector<long long> & segment_ids) {
    // is_horizontal=False: axis=0.
    std::vector<std::pair<bbox_t, long long>> items;
    for (size_t i = 0; i < bboxes.size() && i < segment_ids.size(); ++i) items.push_back({ bboxes[i], segment_ids[i] });
    std::stable_sort(items.begin(), items.end(), [](const auto & a, const auto & b) {
        return std::make_tuple(a.second, a.first[1], a.first[0]) < std::make_tuple(b.second, b.first[1], b.first[0]);
    });
    std::vector<bbox_t> out;
    std::vector<bool> visited(items.size(), false);
    for (size_t i = 0; i < items.size(); ++i) {
        if (visited[i]) continue;
        const long long current = items[i].second;
        std::vector<std::pair<bbox_t, long long>> mid = { items[i] };
        visited[i] = true;
        for (size_t j = i + 1; j < items.size(); ++j) {
            if (visited[j] || items[j].second != current) continue;
            if (projection_iou(mid.back().first, items[j].first, false) > 0.5) {
                mid.push_back(items[j]);
                visited[j] = true;
            }
        }
        std::stable_sort(mid.begin(), mid.end(), [](const auto & a, const auto & b) {
            return std::make_pair(a.first[0], a.first[1]) < std::make_pair(b.first[0], b.first[1]);
        });
        for (const auto & m : mid) out.push_back(m.first);
    }
    return out;
}

long long list_index(const std::vector<bbox_t> & list, const bbox_t & value) {
    for (size_t i = 0; i < list.size(); ++i)
        if (list[i] == value) return (long long)i;
    throw std::invalid_argument("XY-Cut++: bbox is not in list");
}

} // namespace

std::vector<long long> numpy_argsort(const std::vector<long long> & values) {
    std::vector<long long> idx(values.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (long long)i;
    if (values.size() > 1) aquicksort(values.data(), idx.data(), long(values.size()));
    return idx;
}

std::vector<int> get_layout_ordering(const std::vector<entry> & entries,
                                     const std::vector<std::string> & no_mask_labels, bool remove_overlaps) {
    std::vector<node> N;
    for (const auto & e : entries) {
        if (e.label == "doc_title")
            throw std::invalid_argument("XY-Cut++ doc_title branches are not translated (OXR never emits doc_title)");
        node n;
        n.bbox = e.bbox;
        n.label = e.label;
        N.push_back(n);
    }
    std::vector<int> list(N.size());
    for (size_t i = 0; i < list.size(); ++i) list[i] = int(i);

    if (remove_overlaps) {
        std::vector<bool> dropped(list.size(), false);
        for (size_t i = 0; i < list.size(); ++i)
            for (size_t j = i + 1; j < list.size(); ++j) {
                if (dropped[i] || dropped[j]) continue;
                const auto & b1 = N[size_t(list[i])].bbox;
                const auto & b2 = N[size_t(list[j])].bbox;
                const long long a1 = (b1[2] - b1[0]) * (b1[3] - b1[1]), a2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
                if (overlap_div_minbox(b1, b2) > 0.5) dropped[a1 <= a2 ? i : j] = true;
            }
        std::vector<int> kept;
        for (size_t i = 0; i < list.size(); ++i)
            if (!dropped[i]) kept.push_back(list[i]);
        list = kept;
    }

    std::vector<std::string> unused;
    bool tran = false, has_x = false, has_y = false;
    std::vector<long long> pre_x, pre_y;
    get_sub_category(N, list, unused, tran, pre_x, has_x, pre_y, has_y);

    bool projection_direction_page = false;
    const auto mh = text_median_width(N, list, false);
    if (mh.median < 0.05 * double(mh.page_length) && mh.count > 45) {
        has_x = true;
        pre_x.clear();
        has_y = false;
        tran = false;
        projection_direction_page = true;
        // pre_cuts = {"x": []}: len(pre_cuts) == 1, so no pre-cut split happens.
    }
    std::vector<std::vector<int>> groups = { list };
    const bool reset_to_x_only = projection_direction_page;
    if (!reset_to_x_only && (has_x || has_y)) {
        std::vector<const char *> axes =
            tran ? std::vector<const char *>{ "y", "x" } : std::vector<const char *>{ "x", "y" };
        for (const char * axis : axes) {
            const bool is_y = axis[0] == 'y';
            if (is_y ? !has_y : !has_x) continue;
            const int ai = is_y ? 1 : 0;
            long long max_val = std::numeric_limits<long long>::min();
            for (int id : list) max_val = std::max(max_val, N[size_t(id)].bbox[size_t(ai + 2)]);
            std::vector<long long> cuts = is_y ? pre_y : pre_x;
            std::stable_sort(cuts.begin(), cuts.end());
            std::vector<std::pair<long long, long long>> iv;
            long long prev = 0;
            for (long long c : cuts) {
                iv.push_back({ prev, c });
                prev = c;
            }
            iv.push_back({ prev, max_val });
            std::vector<std::vector<int>> next;
            for (const auto & g : groups)
                for (const auto & [start, end] : iv) {
                    std::vector<int> cur;
                    for (int id : g) {
                        const long long v = N[size_t(id)].bbox[size_t(ai)];
                        if (v >= start && v < end) cur.push_back(id);
                    }
                    next.push_back(cur);
                }
            groups = next;
        }
    }

    std::vector<int> final_list;
    double num_index = 0, num_sub_index = 0;
    const std::vector<std::string> & vision_labels = OUTER_VISION;
    const std::vector<std::string> & title_labels = OUTER_TITLES;
    const std::vector<std::string> & vision_title_labels = OUTER_VISION_TITLES;
    for (auto & group : groups) {
        const double median_width = text_median_width(N, group, true).median;
        bool projection_direction = layout_property(N, group, median_width, no_mask_labels, 0.3);
        projection_direction = projection_direction || projection_direction_page;

        std::vector<int> double_text, title_text_blocks, title_blocks, vision_blocks, vision_title_blocks,
            vision_footnote_blocks, other_blocks;
        std::stable_sort(group.begin(), group.end(),
                         [&](int a, int b) { return N[size_t(a)].bbox[1] < N[size_t(b)].bbox[1]; });
        std::vector<int> keep;
        for (int id : group) {
            node & block = N[size_t(id)];
            const std::string & label = block.sub_label;
            if (in(label, no_mask_labels)) {
                if (block.layout == "double")
                    double_text.push_back(id);
                else
                    keep.push_back(id);
            } else if (label == "title_text") {
                title_text_blocks.push_back(id);
            } else if (label == "vision_footnote") {
                vision_footnote_blocks.push_back(id);
            } else if (in(label, vision_title_labels)) {
                vision_title_blocks.push_back(id);
            } else if (in(label, title_labels)) {
                title_blocks.push_back(id);
            } else if (in(label, vision_labels)) {
                vision_blocks.push_back(id);
            } else {
                other_blocks.push_back(id);
            }
        }
        group = keep;

        if (!group.empty()) {
            std::pair<std::vector<long long>, std::vector<long long>> cut;
            std::vector<bbox_t> boxes;
            if (double_text.size() > group.size() || projection_direction) {
                group.insert(group.end(), title_blocks.begin(), title_blocks.end());
                group.insert(group.end(), double_text.begin(), double_text.end());
                title_blocks.clear();
                double_text.clear();
                for (int id : group) boxes.push_back(N[size_t(id)].bbox);
                cut = sort_by_xycut(boxes, 1, 1);
            } else {
                for (int id : group) boxes.push_back(N[size_t(id)].bbox);
                cut = sort_by_xycut(boxes, 0, 1);
            }
            std::vector<bbox_t> sorted_boxes;
            for (long long i : cut.first) sorted_boxes.push_back(boxes[size_t(i)]);
            sorted_boxes = projection_xy(sorted_boxes, cut.second);
            for (int id : group) {
                const long long pos = list_index(sorted_boxes, N[size_t(id)].bbox);
                N[size_t(id)].index = num_index + double(pos) + 1;
                N[size_t(id)].sub_index = num_sub_index + double(pos) + 1;
            }
        }

        auto nearest_match_ = [&](const std::vector<int> & input, const char * type, bool add_index) {
            const std::string distance_type = type;
            for (int id : input) {
                node & block = N[size_t(id)];
                const bbox_t bbox = block.bbox;
                double min_distance = INF;
                distance_config cfg;
                double nearest = 0;
                for (size_t m = 0; m < group.size(); ++m) {
                    const node & match = N[size_t(group[m])];
                    const bbox_t & mb = match.bbox;
                    double distance;
                    if (distance_type == "nearest_iou_edge_distance") {
                        distance = nearest_iou_edge_distance(bbox, mb, block.sub_label, no_mask_labels, median_width,
                                                             block.title_text, block.sub_title, cfg, 10);
                    } else if (distance_type == "title_text") {
                        if ((in(match.label, title_labels) || match.label == "abstract") && !match.title_text.empty()) {
                            const double l = overlap_div_minbox(bbox, match.title_text.front().second);
                            const double r = overlap_div_minbox(bbox, match.title_text.back().second);
                            distance = manhattan(bbox, mb) * (1 - std::max(l, r));
                        } else
                            distance = INF;
                    } else if (distance_type == "manhattan") {
                        distance = manhattan(bbox, mb);
                    } else if (distance_type == "vision_footnote") {
                        if (in(match.label, vision_labels) && !match.vision_footnote.empty()) {
                            const double l = overlap_div_minbox(bbox, match.vision_footnote.front());
                            const double r = overlap_div_minbox(bbox, match.vision_footnote.back());
                            distance = manhattan(bbox, mb) * (1 - std::max(l, r));
                        } else
                            distance = INF;
                    } else {
                        throw std::logic_error("unknown distance type");
                    }
                    if (distance < min_distance) {
                        min_distance = distance;
                        if (add_index)
                            nearest = match.index ? *match.index : 999;
                        else
                            nearest = match.sub_index ? *match.sub_index : 999;
                    }
                }
                if (add_index)
                    block.index = nearest;
                else
                    block.sub_index = nearest;
                group.push_back(id);
            }
        };
        auto renumber = [&](bool both) {
            for (size_t i = 0; i < group.size(); ++i) {
                if (both) N[size_t(group[i])].index = num_index + double(i) + 1;
                N[size_t(group[i])].sub_index = num_sub_index + double(i) + 1;
            }
        };
        auto key_index = [&](int id) { return N[size_t(id)].index.value(); };
        auto key_sub = [&](int id) { return N[size_t(id)].sub_index.value(); };

        std::stable_sort(double_text.begin(), double_text.end(), [&](int a, int b) {
            const auto &A = N[size_t(a)].bbox, &B = N[size_t(b)].bbox;
            return std::make_tuple(py_str::ifloordiv(A[1], 10), py_str::floordiv(double(A[0]), median_width),
                                   A[1] * A[1] + A[0] * A[0]) <
                   std::make_tuple(py_str::ifloordiv(B[1], 10), py_str::floordiv(double(B[0]), median_width),
                                   B[1] * B[1] + B[0] * B[0]);
        });
        nearest_match_(double_text, "nearest_iou_edge_distance", true);
        group = projection_index_xy(N, group);
        renumber(true);

        std::stable_sort(title_blocks.begin(), title_blocks.end(), [&](int a, int b) {
            const auto &A = N[size_t(a)].bbox, &B = N[size_t(b)].bbox;
            return std::make_tuple(A[1], py_str::floordiv(double(A[0]), median_width), A[1] * A[1] + A[0] * A[0]) <
                   std::make_tuple(B[1], py_str::floordiv(double(B[0]), median_width), B[1] * B[1] + B[0] * B[0]);
        });
        nearest_match_(title_blocks, "nearest_iou_edge_distance", true);
        group = projection_index_xy(N, group); // doc_flag is always False here
        renumber(true);

        nearest_match_(title_text_blocks, "title_text", true);
        std::stable_sort(group.begin(), group.end(), [&](int a, int b) {
            auto hv = [&](int id) {
                const auto & bb = N[size_t(id)].bbox;
                return bbox_direction(bb) ? bb[1] : (N[size_t(id)].label == "doc_title" ? -1 : bb[0]);
            };
            return std::make_pair(key_index(a), hv(a)) < std::make_pair(key_index(b), hv(b));
        });
        renumber(true);

        auto sort_sub_yx = [&]() {
            std::stable_sort(group.begin(), group.end(), [&](int a, int b) {
                const auto &A = N[size_t(a)].bbox, &B = N[size_t(b)].bbox;
                return std::make_tuple(key_sub(a), A[1], A[0]) < std::make_tuple(key_sub(b), B[1], B[0]);
            });
        };
        nearest_match_(vision_blocks, "nearest_iou_edge_distance", false);
        sort_sub_yx();
        renumber(false);

        nearest_match_(vision_title_blocks, "nearest_iou_edge_distance", false);
        sort_sub_yx();
        renumber(false);

        nearest_match_(vision_footnote_blocks, "vision_footnote", false);
        std::stable_sort(group.begin(), group.end(), [&](int a, int b) {
            const auto &A = N[size_t(a)].bbox, &B = N[size_t(b)].bbox;
            const int pa = N[size_t(a)].sub_label == "vision_footnote" ? 9999 : 0;
            const int pb = N[size_t(b)].sub_label == "vision_footnote" ? 9999 : 0;
            return std::make_tuple(key_sub(a), pa, A[1], A[0]) < std::make_tuple(key_sub(b), pb, B[1], B[0]);
        });
        renumber(false);

        nearest_match_(other_blocks, "manhattan", false);

        final_list.insert(final_list.end(), group.begin(), group.end());
        num_sub_index += double(group.size());
        for (int id : group)
            if (N[size_t(id)].index && *N[size_t(id)].index != 0) num_index += 1;
    }
    std::stable_sort(final_list.begin(), final_list.end(),
                     [&](int a, int b) { return N[size_t(a)].sub_index.value() < N[size_t(b)].sub_index.value(); });
    return final_list;
}

} // namespace xycut_plus_plus
