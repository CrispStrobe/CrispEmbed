#include "ocr_region_router.h"

#include <cstdio>
#include <vector>

static int failures = 0;
#define CHECK(expr)                                                                                                    \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            std::fprintf(stderr, "FAIL: %s\n", #expr);                                                                 \
            ++failures;                                                                                                \
        }                                                                                                              \
    } while (0)

int main() {
    using namespace ocr_region_router;
    using layout_detect::label_id;

    std::vector<layout_detect::region> layout = {
        { 0, 0, 100, 100, 0.9f, label_id::table, "table" },
        { 100, 0, 200, 100, 0.9f, label_id::formula, "formula" },
        { 200, 0, 300, 100, 0.9f, label_id::text, "text" },
    };
    std::vector<ocr_detect::text_box> boxes = {
        { 10, 10, 20, 10, 0.9f, 0.0f, {}, {} },
        { 110, 10, 20, 10, 0.9f, 0.0f, {}, {} },
        { 210, 10, 20, 10, 0.9f, 0.0f, {}, {} },
        { 400, 10, 20, 10, 0.9f, 0.0f, {}, {} },
    };

    routing_plan plan;
    request_policy policy;
    policy.want_tables = true;
    policy.want_formulas = true;
    build(layout, boxes, policy, plan);
    CHECK(plan.table_layout_indices.size() == 1);
    CHECK(plan.formula_layout_indices.size() == 1);
    CHECK(plan.text_indices.size() == 2);
    CHECK(plan.suppress_text[0] == 1 && plan.suppress_text[1] == 1);
    CHECK(plan.suppress_text[2] == 0 && plan.suppress_text[3] == 0);
    CHECK(plan.text_to_layout[3] == -1);

    routing_plan no_layout;
    build({}, boxes, policy, no_layout);
    CHECK(no_layout.text_indices.size() == boxes.size());
    CHECK(no_layout.table_layout_indices.empty());

    return failures == 0 ? 0 : 1;
}
