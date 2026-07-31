#pragma once

#include <cstddef>

struct easyocr_craft_context;

easyocr_craft_context * easyocr_craft_init(const char * model_path, int width, int height);
void easyocr_craft_free(easyocr_craft_context * ctx);
bool easyocr_craft_forward(easyocr_craft_context * ctx, const float * chw_input, size_t n_elem);
int easyocr_craft_diff(easyocr_craft_context * ctx, const char * ref_path);
