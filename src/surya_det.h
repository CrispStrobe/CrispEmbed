// surya_det.h — surya-ocr-2 text detector (EfficientViT segformer)
//
// Segmentation-based text line detection. Produces a 2-channel heatmap
// (text lines, separators) that is post-processed into polygon bounding boxes.
//
// Architecture: EfficientViT-Large encoder + SegFormer decode head
//   38M params, 147 MB F32, 73 MB F16
//
// Port of: datalab-to/surya-ocr-2 detector
//   (surya.detection.model.encoderdecoder.EfficientViTForSemanticSegmentation)

#ifndef SURYA_DET_H
#define SURYA_DET_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct surya_det_context surya_det_context;

typedef struct {
    int input_h, input_w;      // expected input (1200x1200)
    int num_classes;           // 2 (text, separator)
    int stem_ch;               // 32
    int stage_ch[4];           // {64, 128, 256, 512}
    int head_dim;              // 32
    int dec_hidden;            // 512
    int dec_layer_hidden;      // 128
} surya_det_hparams;

// Bounding box result
typedef struct {
    float x0, y0, x1, y1;     // pixel coordinates in original image
    float confidence;
} surya_det_bbox;

// Initialize from GGUF
surya_det_context * surya_det_init(const char * model_path, int n_threads);
void surya_det_free(surya_det_context * ctx);
const surya_det_hparams * surya_det_get_hparams(const surya_det_context * ctx);

// Run detection on grayscale or RGB image
// Returns heatmap as [2, out_h, out_w] float array (caller does NOT free)
// out_h = input_h / 4, out_w = input_w / 4
const float * surya_det_detect(surya_det_context * ctx,
                                const uint8_t * pixels, int width, int height, int channels,
                                int * out_h, int * out_w);

// Run detection from pre-processed float tensor [3, H, W] (ImageNet-normalized)
// Bypasses image loading/resize — used for parity testing with Python reference.
const float * surya_det_detect_raw(surya_det_context * ctx,
                                    const float * preprocessed, int H, int W,
                                    int * out_h, int * out_w);

// Get raw heatmap after last detection (2 channels: text, separator)
const float * surya_det_get_heatmap(surya_det_context * ctx, int * out_h, int * out_w);

// Extract text bounding boxes from the heatmap (call after surya_det_detect)
// Returns array of surya_det_bbox. Caller does NOT free.
// text_threshold: confidence for text region (default 0.6)
// low_threshold:  binary threshold for connected components (default 0.35)
const surya_det_bbox * surya_det_get_boxes(surya_det_context * ctx,
                                            int orig_w, int orig_h,
                                            float text_threshold, float low_threshold,
                                            int * n_boxes);

// Get intermediate activation for parity debugging (env SURYA_DET_DUMP)
const float * surya_det_get_debug(surya_det_context * ctx, const char * name,
                                   int * n_elements);

#ifdef __cplusplus
}
#endif

#endif // SURYA_DET_H
