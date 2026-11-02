// cv_compat.h — exact OpenCV 4.x uint8 image operations used by Python
// document helpers (OXR crops/recovery and ink-coverage checks).
//
// Translated from OpenCV imgproc resize.cpp (resizeArea_, computeResizeAreaTab)
// and color_rgb.cpp (RGB2Gray fixed point), Apache-2.0. Compile without FMA
// contraction (CMake sets -ffp-contract=off for cv_compat.cpp) so float
// accumulation matches the x86-64 SSE baseline of the opencv-python wheels.
#pragma once

#include <cstdint>
#include <vector>

namespace cv_compat {

// cv2.resize(src, (dst_w, dst_h), interpolation=cv2.INTER_AREA) for a tightly
// packed uint8 image with cn channels. Equal sizes copy. Downscaling with a
// non-integer ratio uses the generic area-weight path. Upscaling (which
// OpenCV routes to bilinear) and integer-ratio fast decimation are rejected.
std::vector<uint8_t> resize_area_u8(const uint8_t * src, int width, int height, int cn, int dst_w, int dst_h);

// cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) applied to the same pixels stored
// as RGB: Y = (9798 R + 19235 G + 3735 B + 16384) >> 15 (OpenCV 4.11
// RGB2Gray 8u fixed point; verified against cv2 4.11.0 on 2M random colours).
std::vector<uint8_t> rgb_to_gray_u8(const uint8_t * rgb, int width, int height);

} // namespace cv_compat
