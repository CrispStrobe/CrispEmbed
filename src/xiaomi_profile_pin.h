#pragma once
// Pins for the experimental Xiaomi-OCR-0 profile (format 4).
//
// Every value here is copied from an actually observed original capture
// (tools/kaggle/xiaomi-ocr-0-blueprint). Empty values fail closed, so a build
// without the audited evidence cannot load a Xiaomi GGUF even with the
// environment gates enabled.
namespace qwen35_ocr {
// Byte-exact output of
// Qwen3VLProcessor.apply_chat_template(document prompt, enable_thinking=False).
inline constexpr const char * xiaomi_rendered_template = nullptr;
// LFS SHA-256 of the pinned checkpoint revision.
inline constexpr const char * xiaomi_checkpoint_sha256 = nullptr;
// SHA-256 of the audited original-capture evidence that produced the values above.
inline constexpr const char * xiaomi_evidence_sha256 = nullptr;
} // namespace qwen35_ocr
