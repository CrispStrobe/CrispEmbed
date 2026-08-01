#include "tesseract_recoder.h"

#include <algorithm>

namespace tesseract_recoder {

bool compose_classes(const std::vector<int> & labels, const std::vector<std::vector<int>> & codes,
                     std::vector<int> & unichars, std::vector<int> & starts) {
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

} // namespace tesseract_recoder
