// Experimental fixed-256 MI-GAN CPU/F32 prototype. No public C ABI/registry.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace migan {

constexpr int image_size = 256;
constexpr size_t rgb_elements = size_t(image_size) * image_size * 3;
constexpr size_t mask_elements = size_t(image_size) * image_size;
constexpr size_t input_elements = size_t(image_size) * image_size * 4;

struct context;

// Both init and run require core_env::on("CRISPEMBED_MIGAN"). The expected
// checkpoint digest must come from a reviewed original capture, not the file.
context * init(const char * path, const char * expected_checkpoint_sha256, int n_threads, std::string & error);
void free(context * ctx);
bool run(context * ctx, const float * input_nchw, size_t input_count, float * raw_nchw, size_t output_count,
         std::string & error);

// Fixed-size binary white=fill boundary. These pure contract helpers do not
// load or execute a model. No resize, mask threshold or implicit channel cast.
bool preprocess_rgb8(const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes, int width,
                     int height, float * input_nchw, size_t input_count, std::string & error);
bool composite_rgb8(const float * raw_nchw, size_t raw_count, const uint8_t * rgb, size_t rgb_bytes,
                    const uint8_t * fill_mask, size_t mask_bytes, uint8_t * output_rgb, size_t output_bytes,
                    std::string & error);

} // namespace migan
