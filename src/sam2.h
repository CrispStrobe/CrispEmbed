// sam2.h — SAM 2.1 image segmentation (Hiera encoder, FPN neck, prompt encoder, two-way mask decoder).
//
// Image mode only: one image, then any number of point/box prompts for it, like SAM 2's
// SAM2ImagePredictor. The video-memory parts of SAM 2 are not implemented.
//
// Weights: models/convert-sam2-to-gguf.py. Parity: tools/dump_sam2_reference.py + tests/test_sam2_diff.cpp.
// Gates: SAM2_FORCE_CPU=1 keeps everything on the CPU backend; CRISPEMBED_SAM2_BENCH=1 prints timings.
#pragma once

#include <cstdint>

struct sam2_context;

// Returns nullptr on failure. n_threads <= 0: 4.
sam2_context * sam2_init(const char * model_path, int n_threads);
void sam2_free(sam2_context * ctx);

// Side of the square network input (1024) and of the low-resolution mask logits (256).
int sam2_image_size(const sam2_context * ctx);
int sam2_mask_size(const sam2_context * ctx);
// Name of the backend the encoder runs on ("CPU", "MTL0", ...).
const char * sam2_backend_name(const sam2_context * ctx);

// The image as the network takes it: 3 x size x size floats, planar RGB, already resized
// (antialiased bilinear, aspect ratio not kept) and normalised with the ImageNet mean/std. 0 on success.
int sam2_set_image_f32(sam2_context * ctx, const float * chw);
// 8-bit RGB (row by row, 3 bytes per pixel): resized and normalised here as SAM 2 does. 0 on success.
int sam2_set_image_rgb(sam2_context * ctx, const uint8_t * rgb, int width, int height);

// Mask logits for prompts on the current image. points_xy: n pairs in pixels of the network input
// (x * size / width); box corners come as two points labelled 2 and 3, before the clicks.
// labels: 1 object, 0 background, 2/3 box corners. Writes 4 x mask_size x mask_size logits
// (token 0: the single mask; 1..3: the three alternatives) and 4 predicted scores. 0 on success.
int sam2_predict(sam2_context * ctx, const float * points_xy, const int * labels, int n_points, float * mask_logits,
                 float * scores);

#ifdef __cplusplus
#include <map>
#include <string>
#include <vector>
// Test hook (tests/test_sam2_diff.cpp): with probing on, the encoder keeps its intermediate
// tensors (hiera.patch, hiera.block.NN, neck.N, image_embed, high_res_0, high_res_1),
// channel-last, as float vectors.
void sam2_set_probing(sam2_context * ctx, bool on);
const std::map<std::string, std::vector<float>> & sam2_probes(const sam2_context * ctx);
#endif
