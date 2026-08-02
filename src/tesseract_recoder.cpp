#include "tesseract_recoder.h"

#include <algorithm>
#include <cstdint>

namespace tesseract_recoder {

bool prefix_legal(const std::vector<int> & prefix, const std::vector<std::vector<int>> & codes, bool allow_partial) {
    if (codes.empty()) return true;
    const int n = (int)prefix.size();
    std::vector<uint8_t> reachable(n + 1, 0);
    reachable[0] = 1;
    for (int pos = 0; pos <= n; ++pos) {
        if (!reachable[pos]) continue;
        for (const auto & code : codes) {
            if (code.empty() || pos + (int)code.size() > n) {
                if (allow_partial && pos < n && !code.empty() && pos + (int)code.size() > n &&
                    std::equal(prefix.begin() + pos, prefix.end(), code.begin())) {
                    return true;
                }
                continue;
            }
            if (std::equal(code.begin(), code.end(), prefix.begin() + pos)) reachable[pos + code.size()] = 1;
        }
    }
    return reachable[n] != 0;
}

bool compose_classes(const std::vector<int> & labels, const std::vector<std::vector<int>> & codes,
                     std::vector<int> & unichars, std::vector<int> & starts) {
    // These are result vectors, not accumulators. Clear them so a failed
    // composition cannot leave stale tokens from a previous recognition.
    unichars.clear();
    starts.clear();
    const int n = (int)labels.size();
    if (codes.empty()) return false;
    std::vector<int> previous(n + 1, -1), previous_uid(n + 1, -1);
    previous[0] = 0;
    for (int pos = 0; pos < n; ++pos) {
        if (previous[pos] < 0) continue;
        for (int uid = 0; uid < (int)codes.size(); ++uid) {
            const auto & code = codes[uid];
            if (code.empty() || pos + (int)code.size() > n) continue;
            if (!std::equal(code.begin(), code.end(), labels.begin() + pos)) continue;
            const int end = pos + (int)code.size();
            if (previous[end] < 0) {
                previous[end] = pos;
                previous_uid[end] = uid;
            }
        }
    }
    if (previous[n] < 0) return false;
    for (int end = n; end > 0;) {
        const int uid = previous_uid[end];
        const int begin = previous[end];
        if (uid < 0 || begin < 0) return false;
        unichars.push_back(uid);
        starts.push_back(begin);
        end = begin;
    }
    std::reverse(unichars.begin(), unichars.end());
    std::reverse(starts.begin(), starts.end());
    return true;
}

bool compose_classes_partial(const std::vector<int> & labels, const std::vector<std::vector<int>> & codes,
                             std::vector<int> & unichars, std::vector<int> & starts) {
    unichars.clear();
    starts.clear();
    if (codes.empty()) return false;
    for (int pos = 0; pos < (int)labels.size();) {
        int best_uid = -1;
        int best_len = 0;
        for (int uid = 0; uid < (int)codes.size(); ++uid) {
            const auto & code = codes[uid];
            if (code.empty() || pos + (int)code.size() > (int)labels.size()) continue;
            if (!std::equal(code.begin(), code.end(), labels.begin() + pos)) continue;
            if ((int)code.size() > best_len) {
                best_uid = uid;
                best_len = (int)code.size();
            }
        }
        starts.push_back(pos);
        if (best_uid < 0) {
            unichars.push_back(-1);
            ++pos;
        } else {
            unichars.push_back(best_uid);
            pos += best_len;
        }
    }
    return !unichars.empty();
}

} // namespace tesseract_recoder
