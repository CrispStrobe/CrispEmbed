// crispembed server — HTTP API for embeddings, OCR, face, NER, layout.
//
// Usage: crispembed-server -m model.gguf [--port 8080] [--host 0.0.0.0]
//
// Endpoints:
//   POST /embed           — {"texts": ["hello"]} → {"embeddings": [[...]]}
//   POST /v1/embeddings   — OpenAI-compatible embedding API
//   POST /api/embed       — Ollama-compatible (batch)
//   POST /api/embeddings  — Ollama-compatible (single, legacy)
//   POST /math/ocr        — {"image": "formula.png"} → {"text": "...", "ms": M}
//   POST /ocr             — {"image": "doc.png"} → {"results": [...], "ms": M}
//   POST /ocr/pipeline    — {"image": "doc.png"} → full orchestrator (routing+cleanup+gate)
//   POST /layout/detect   — {"image": "page.png"} → {"regions": [...]}
//   POST /text/detect     — {"image": "page.png"} → {"regions": [...]}
//   POST /detect          — {"image": "face.jpg"} → face detection
//   POST /face            — {"image": "face.jpg"} → face embedding
//   POST /vit/encode      — {"image": "img.jpg"} → ViT embedding
//   POST /pix2struct/generate — {"image": "doc.png", "max_tokens": 256} → {"text": "...", "ms": M}
//   POST /clip/text       — {"text": "query"} → CLIP text embedding
//   POST /colbert/score   — {"query": "...", "documents": [...]} → ColBERT scoring
//   POST /ner/extract     — {"text": "...", "labels": [...]} → NER entities
//   POST /scan/cleanup    — {"image": "scan.png"} → cleaned image
//   POST /pdf/dpi              — {"file": "path.pdf"} → per-page DPI info
//   POST /preprocess/skew      — {"image": "..."} → {"angle": F, "confidence": F}
//   POST /preprocess/dewarp    — {"image": "...", "output": "..."} → PGM image or JSON
//   POST /preprocess/cc-detect — {"image": "..."} → {"regions": [...]}
//   POST /render/ocr           — {"results": [...], "format": "hocr"} → rendered document
//   POST /ocr/document         — multi-page OCR: images → searchable PDF / hOCR / text
//   GET  /health          — server status + loaded capabilities

#include "crispembed.h"
#include "ocr_render.h"
#include "scan_cleanup.h"
#include "model_mgr.h"
#include "pdf_info.h"
#if __has_include("text_lid_dispatch.h")
#include "text_lid_dispatch.h"
#define SERVER_HAS_LID 1
#else
#define SERVER_HAS_LID 0
#endif
#include "httplib.h"

// stb_image for /math/ocr image loading
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "../../ggml/examples/stb_image.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

static std::string json_escape(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string host = "127.0.0.1";
    std::string det_model_path;  // face detection model
    std::string rec_model_path;  // face recognition model
    std::string vit_model_path;  // standalone ViT model (SigLIP/CLIP)
    std::string clip_text_model_path;  // CLIP text encoder
    std::string ocr_model_path;   // math OCR model (PP-FormulaNet, HMER, BTTR, PosFormer, etc.)
    std::string ocr_det_model_path;   // general OCR: text detection model (DBNet)
    std::string ocr_rec_model_path;   // general OCR: text recognition model (TrOCR)
    std::string layout_model_path;    // layout detection model (RT-DETRv2)
    std::string text_det_model_path;  // surya text detection model
    std::string ner_model_path;       // NER model (GLiNER)
    std::string lid_model_path;       // text LID model
    bool enable_ocr_orch = false;     // --ocr-pipeline: enable orchestrator endpoint
    std::string vlm_model_path;       // VLM escalation model for orchestrator
    int vlm_engine = 0;               // 0=GOT, 1=GLM, 2=Qwen2-VL(+PaddleOCR-VL), 3=InternVL2
    std::string punct_model_path;     // punct restoration model for orchestrator
    std::string sr_model_path;        // text super-resolution model (--sr-model)
    std::string pan_model_path;       // PAN super-resolution model (--pan-model)
    std::string hat_model_path;       // HAT super-resolution model (--hat-model)
    std::string dat_model_path;       // DAT super-resolution model (--dat-model)
    std::string safmn_model_path;     // SAFMN super-resolution model (--safmn-model)
    std::string esrgan_model_path;    // Real-ESRGAN super-resolution model (--esrgan-model)
    std::string swinir_model_path;    // SwinIR super-resolution model (--swinir-model)
    std::string tbsrn_model_path;     // TBSRN text-line SR model (--tbsrn-model)
    std::string restormer_model_path; // Restormer restoration model (--restormer-model)
    std::string scunet_model_path;   // SCUNet denoising model (--scunet-model)
    std::string instructir_model_path; // InstructIR restoration model (--instructir-model)
    std::string adair_model_path;     // AdaIR restoration model (--adair-model)
    std::string pix2struct_model_path; // Pix2Struct document understanding model (--pix2struct)
    int port = 8080;
    int n_threads = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) host = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--det") == 0 && i + 1 < argc) det_model_path = argv[++i];
        else if (strcmp(argv[i], "--rec") == 0 && i + 1 < argc) rec_model_path = argv[++i];
        else if (strcmp(argv[i], "--vit") == 0 && i + 1 < argc) vit_model_path = argv[++i];
        else if (strcmp(argv[i], "--pix2struct") == 0 && i + 1 < argc) pix2struct_model_path = argv[++i];
        else if (strcmp(argv[i], "--clip-text") == 0 && i + 1 < argc) clip_text_model_path = argv[++i];
        else if (strcmp(argv[i], "--ocr") == 0 && i + 1 < argc) ocr_model_path = argv[++i];
        else if (strcmp(argv[i], "--ocr-det") == 0 && i + 1 < argc) ocr_det_model_path = argv[++i];
        else if (strcmp(argv[i], "--ocr-rec") == 0 && i + 1 < argc) ocr_rec_model_path = argv[++i];
        else if (strcmp(argv[i], "--layout") == 0 && i + 1 < argc) layout_model_path = argv[++i];
        else if (strcmp(argv[i], "--text-det") == 0 && i + 1 < argc) text_det_model_path = argv[++i];
        else if (strcmp(argv[i], "--ner") == 0 && i + 1 < argc) ner_model_path = argv[++i];
        else if (strcmp(argv[i], "--lid") == 0 && i + 1 < argc) lid_model_path = argv[++i];
        else if (strcmp(argv[i], "--ocr-pipeline") == 0) enable_ocr_orch = true;
        else if (strcmp(argv[i], "--vlm-model") == 0 && i + 1 < argc) vlm_model_path = argv[++i];
        else if (strcmp(argv[i], "--vlm-engine") == 0 && i + 1 < argc) vlm_engine = atoi(argv[++i]);
        else if (strcmp(argv[i], "--punct-model") == 0 && i + 1 < argc) punct_model_path = argv[++i];
        else if (strcmp(argv[i], "--sr-model") == 0 && i + 1 < argc) sr_model_path = argv[++i];
        else if (strcmp(argv[i], "--pan-model") == 0 && i + 1 < argc) pan_model_path = argv[++i];
        else if (strcmp(argv[i], "--hat-model") == 0 && i + 1 < argc) hat_model_path = argv[++i];
        else if (strcmp(argv[i], "--dat-model") == 0 && i + 1 < argc) dat_model_path = argv[++i];
        else if (strcmp(argv[i], "--safmn-model") == 0 && i + 1 < argc) safmn_model_path = argv[++i];
        else if (strcmp(argv[i], "--esrgan-model") == 0 && i + 1 < argc) esrgan_model_path = argv[++i];
        else if (strcmp(argv[i], "--swinir-model") == 0 && i + 1 < argc) swinir_model_path = argv[++i];
        else if (strcmp(argv[i], "--tbsrn-model") == 0 && i + 1 < argc) tbsrn_model_path = argv[++i];
        else if (strcmp(argv[i], "--restormer-model") == 0 && i + 1 < argc) restormer_model_path = argv[++i];
        else if (strcmp(argv[i], "--scunet-model") == 0 && i + 1 < argc) scunet_model_path = argv[++i];
        else if (strcmp(argv[i], "--instructir-model") == 0 && i + 1 < argc) instructir_model_path = argv[++i];
        else if (strcmp(argv[i], "--adair-model") == 0 && i + 1 < argc) adair_model_path = argv[++i];
    }

    if (model_path.empty() && det_model_path.empty() && vit_model_path.empty() && ocr_model_path.empty() && layout_model_path.empty() && ner_model_path.empty() && sr_model_path.empty() && pan_model_path.empty() && hat_model_path.empty() && dat_model_path.empty() && safmn_model_path.empty() && esrgan_model_path.empty() && swinir_model_path.empty() && tbsrn_model_path.empty() && restormer_model_path.empty() && scunet_model_path.empty() && instructir_model_path.empty() && adair_model_path.empty()) {
        fprintf(stderr, "Usage: crispembed-server -m MODEL [--port 8080] [--host 127.0.0.1]\n");
        fprintf(stderr, "  MODEL can be a .gguf path or a model name (auto-downloads from HuggingFace)\n");
        fprintf(stderr, "  Examples: -m all-MiniLM-L6-v2   -m octen-0.6b   -m model.gguf\n");
        fprintf(stderr, "\nFace pipeline:\n");
        fprintf(stderr, "  --det MODEL   face detection model (SCRFD GGUF)\n");
        fprintf(stderr, "  --rec MODEL   face recognition model (ArcFace/SFace GGUF)\n");
        fprintf(stderr, "\nStandalone ViT (SigLIP/CLIP):\n");
        fprintf(stderr, "  --vit MODEL   ViT image embedding model (SigLIP/CLIP GGUF)\n");
        fprintf(stderr, "  --pix2struct MODEL  Pix2Struct document understanding model GGUF\n");
        fprintf(stderr, "  --clip-text MODEL  CLIP text encoder GGUF\n");
        fprintf(stderr, "\nMath OCR (formula recognition):\n");
        fprintf(stderr, "  --ocr MODEL   math OCR model (PP-FormulaNet, HMER, BTTR GGUF)\n");
        fprintf(stderr, "\nLayout detection (document structure):\n");
        fprintf(stderr, "  --layout MODEL   RT-DETRv2 layout detection model GGUF\n");
        fprintf(stderr, "\nText detection:\n");
        fprintf(stderr, "  --text-det MODEL  Surya text line detection model GGUF\n");
        fprintf(stderr, "\nNamed Entity Recognition:\n");
        fprintf(stderr, "  --ner MODEL       GLiNER zero-shot NER model GGUF\n");
        fprintf(stderr, "  --lid MODEL       text LID model GGUF (CLD3 or GlotLID)\n");
        fprintf(stderr, "\nOCR orchestrator (full pipeline with routing + cleanup + accept-gate):\n");
        fprintf(stderr, "  --ocr-pipeline    enable POST /ocr/pipeline endpoint\n");
        fprintf(stderr, "  --ocr-det MODEL   detection model (required with --ocr-pipeline)\n");
        fprintf(stderr, "  --ocr-rec MODEL   recognition model (required with --ocr-pipeline)\n");
        fprintf(stderr, "  --vlm-model MODEL VLM escalation fallback GGUF (optional)\n");
        fprintf(stderr, "  --vlm-engine N    VLM backend: 0=GOT 1=GLM 2=Qwen2-VL 3=InternVL2\n");
        fprintf(stderr, "  --punct-model M   post-OCR punctuation/spacing GGUF (optional)\n");
        fprintf(stderr, "\nText super-resolution (low-DPI upscaling before OCR):\n");
        fprintf(stderr, "  --sr-model MODEL  text SR GGUF (NAFNet+PixelShuffle, 2x or 4x); enables POST /text/sr\n");
        fprintf(stderr, "\nPAN super-resolution (whole-image upscaling):\n");
        fprintf(stderr, "  --pan-model MODEL PAN SR GGUF (Pixel Attention Network, 2x or 4x); enables POST /pan/sr\n");
        fprintf(stderr, "\nHAT super-resolution (Hybrid Attention Transformer, 4x):\n");
        fprintf(stderr, "  --hat-model MODEL   HAT GGUF (21M params, CVPR 2023 SOTA); enables POST /hat/sr\n");
        fprintf(stderr, "\nDAT super-resolution (Dual Aggregation Transformer, 2x):\n");
        fprintf(stderr, "  --dat-model MODEL   DAT-light GGUF (830K params, ICCV 2023); enables POST /dat/sr\n");
        fprintf(stderr, "\nSAFMN super-resolution (4x):\n");
        fprintf(stderr, "  --safmn-model MODEL SAFMN GGUF (228K params, ICCV 2023); enables POST /safmn/sr\n");
        fprintf(stderr, "\nReal-ESRGAN super-resolution (4x):\n");
        fprintf(stderr, "  --esrgan-model MODEL Real-ESRGAN GGUF (620K params); enables POST /esrgan/sr\n");
        fprintf(stderr, "\nSwinIR super-resolution (Swin Transformer, 2x/3x/4x):\n");
        fprintf(stderr, "  --swinir-model MODEL SwinIR GGUF (lightweight, ~0.9M-4.2M params); enables POST /swinir/sr\n");
        fprintf(stderr, "\nTBSRN text-line super-resolution:\n");
        fprintf(stderr, "  --tbsrn-model MODEL TBSRN GGUF (Telescope, 1.1M params, fixed 4x); enables POST /tbsrn/sr\n");
        fprintf(stderr, "\nRestormer image restoration:\n");
        fprintf(stderr, "  --restormer-model MODEL Restormer GGUF (26M params, CVPR 2022); enables POST /restormer\n");
        fprintf(stderr, "\nSCUNet image denoising:\n");
        fprintf(stderr, "  --scunet-model MODEL  SCUNet GGUF (18M params, CVPR 2022); enables POST /scunet/denoise\n");
        fprintf(stderr, "\nInstructIR all-in-one image restoration:\n");
        fprintf(stderr, "  --instructir-model MODEL  InstructIR GGUF (16M params, ECCV 2024); enables POST /instructir/restore\n");
        fprintf(stderr, "\nAdaIR all-in-one image restoration:\n");
        fprintf(stderr, "  --adair-model MODEL  AdaIR GGUF (28.8M params, ICLR 2025); enables POST /adair/restore\n");
        return 1;
    }

    // Text embedding model (optional when only face models are loaded)
    crispembed_context * ctx = nullptr;
    const crispembed_hparams * hp = nullptr;
    int dim = 0;
    std::mutex model_mutex;
    std::string model_name;

    if (!model_path.empty()) {
        std::string resolved = crispembed_mgr::resolve_model(model_path, true);
        if (resolved.empty()) {
            fprintf(stderr, "Failed to resolve model '%s'\n", model_path.c_str());
            return 1;
        }
        model_path = resolved;
        ctx = crispembed_init(model_path.c_str(), n_threads);
        if (!ctx) {
            fprintf(stderr, "Failed to load model '%s'\n", model_path.c_str());
            return 1;
        }
        hp = crispembed_get_hparams(ctx);
        dim = hp->n_output > 0 ? hp->n_output : hp->n_embd;
        model_name = model_path;
        auto slash = model_name.find_last_of("/\\");
        if (slash != std::string::npos) model_name = model_name.substr(slash + 1);
        auto dot = model_name.rfind(".gguf");
        if (dot != std::string::npos) model_name = model_name.substr(0, dot);
    }

    httplib::Server svr;

    // CORS: allow browser access from any origin
    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "POST, GET, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type, Authorization"},
    });
    svr.Options("/(.*)", [](const httplib::Request &, httplib::Response & res) {
        res.status = 204;
    });

    // POST /embed — simple API
    svr.Post("/embed", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no text model loaded\"}", "application/json");
            return;
        }
        std::vector<std::string> texts;
        auto body = req.body;

        // Quick parse: find "texts" array or "text" string
        auto pos = body.find("\"texts\"");
        if (pos != std::string::npos) {
            auto arr_start = body.find('[', pos);
            auto arr_end = body.find(']', arr_start);
            if (arr_start != std::string::npos && arr_end != std::string::npos) {
                std::string arr = body.substr(arr_start + 1, arr_end - arr_start - 1);
                size_t i = 0;
                while (i < arr.size()) {
                    auto q1 = arr.find('"', i);
                    if (q1 == std::string::npos) break;
                    auto q2 = arr.find('"', q1 + 1);
                    if (q2 == std::string::npos) break;
                    texts.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
                    i = q2 + 1;
                }
            }
        } else {
            pos = body.find("\"text\"");
            if (pos != std::string::npos) {
                auto q1 = body.find('"', pos + 6);
                auto q2 = body.find('"', q1 + 1);
                if (q1 != std::string::npos && q2 != std::string::npos) {
                    texts.push_back(body.substr(q1 + 1, q2 - q1 - 1));
                }
            }
        }

        if (texts.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"no texts provided\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(model_mutex);
        auto t0 = std::chrono::steady_clock::now();

        std::ostringstream js;
        js << "{\"embeddings\": [";

        if (texts.size() == 1) {
            // Single text: use single encode
            int d = 0;
            const float * vec = crispembed_encode(ctx, texts[0].c_str(), &d);
            if (!vec || d <= 0) {
                res.status = 500;
                res.set_content("{\"error\": \"encoding failed\"}", "application/json");
                return;
            }
            js << "[";
            for (int j = 0; j < d; j++) {
                if (j > 0) js << ", ";
                js << vec[j];
            }
            js << "]";
        } else {
            // Multiple texts: use batched encode (single graph on GPU)
            std::vector<const char *> ptrs(texts.size());
            for (size_t i = 0; i < texts.size(); i++) ptrs[i] = texts[i].c_str();
            int d = 0;
            const float * vecs = crispembed_encode_batch(ctx, ptrs.data(), (int)texts.size(), &d);
            if (!vecs || d <= 0) {
                res.status = 500;
                res.set_content("{\"error\": \"batch encoding failed\"}", "application/json");
                return;
            }
            for (size_t i = 0; i < texts.size(); i++) {
                if (i > 0) js << ", ";
                js << "[";
                for (int j = 0; j < d; j++) {
                    if (j > 0) js << ", ";
                    js << vecs[i * d + j];
                }
                js << "]";
            }
        }

        js << "], \"dim\": " << dim << "}";

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "crispembed-server: encoded %zu text(s) in %.1f ms\n", texts.size(), ms);

        res.set_content(js.str(), "application/json");
    });

    // POST /v1/embeddings — OpenAI-compatible
    svr.Post("/v1/embeddings", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no text model loaded\"}", "application/json");
            return;
        }
        std::vector<std::string> texts;
        auto body = req.body;
        auto pos = body.find("\"input\"");
        if (pos != std::string::npos) {
            auto arr_start = body.find('[', pos);
            if (arr_start != std::string::npos) {
                auto arr_end = body.find(']', arr_start);
                std::string arr = body.substr(arr_start + 1, arr_end - arr_start - 1);
                size_t i = 0;
                while (i < arr.size()) {
                    auto q1 = arr.find('"', i);
                    if (q1 == std::string::npos) break;
                    auto q2 = arr.find('"', q1 + 1);
                    if (q2 == std::string::npos) break;
                    texts.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
                    i = q2 + 1;
                }
            } else {
                auto q1 = body.find('"', pos + 7);
                auto q2 = body.find('"', q1 + 1);
                if (q1 != std::string::npos && q2 != std::string::npos)
                    texts.push_back(body.substr(q1 + 1, q2 - q1 - 1));
            }
        }

        if (texts.empty()) {
            res.status = 400;
            res.set_content("{\"error\": {\"message\": \"no input\"}}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(model_mutex);

        std::ostringstream js;
        // Batch encode all texts at once
        std::vector<const char *> ptrs(texts.size());
        for (size_t i = 0; i < texts.size(); i++) ptrs[i] = texts[i].c_str();
        int d = 0;
        const float * vecs = crispembed_encode_batch(ctx, ptrs.data(), (int)texts.size(), &d);
        if (!vecs || d <= 0) {
            res.status = 500;
            res.set_content("{\"error\": {\"message\": \"encoding failed\"}}", "application/json");
            return;
        }

        js << "{\"object\": \"list\", \"data\": [";
        for (size_t i = 0; i < texts.size(); i++) {
            if (i > 0) js << ", ";
            js << "{\"object\": \"embedding\", \"index\": " << i << ", \"embedding\": [";
            for (int j = 0; j < d; j++) {
                if (j > 0) js << ", ";
                js << vecs[i * d + j];
            }
            js << "]}";
        }
        js << "], \"model\": \"" << json_escape(model_name) << "\", \"usage\": {\"prompt_tokens\": 0, \"total_tokens\": 0}}";
        res.set_content(js.str(), "application/json");
    });

    // POST /api/embed — Ollama-compatible (batch)
    // Request:  {"model": "...", "input": ["text1", "text2"]} or {"model": "...", "input": "text"}
    // Response: {"model": "...", "embeddings": [[...], [...]], "total_duration": ns, "load_duration": 0, "prompt_eval_count": n}
    svr.Post("/api/embed", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no text model loaded\"}", "application/json");
            return;
        }
        std::vector<std::string> texts;
        auto body = req.body;

        // Parse "input" — can be array or string
        auto pos = body.find("\"input\"");
        if (pos != std::string::npos) {
            auto arr_start = body.find('[', pos);
            auto str_start = body.find('"', pos + 7);
            if (arr_start != std::string::npos &&
                (str_start == std::string::npos || arr_start < str_start)) {
                // Array of strings
                auto arr_end = body.find(']', arr_start);
                if (arr_end != std::string::npos) {
                    std::string arr = body.substr(arr_start + 1, arr_end - arr_start - 1);
                    size_t i = 0;
                    while (i < arr.size()) {
                        auto q1 = arr.find('"', i);
                        if (q1 == std::string::npos) break;
                        auto q2 = arr.find('"', q1 + 1);
                        if (q2 == std::string::npos) break;
                        texts.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
                        i = q2 + 1;
                    }
                }
            } else if (str_start != std::string::npos) {
                // Single string
                auto q2 = body.find('"', str_start + 1);
                if (q2 != std::string::npos)
                    texts.push_back(body.substr(str_start + 1, q2 - str_start - 1));
            }
        }

        if (texts.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"no input provided\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(model_mutex);
        auto t0 = std::chrono::steady_clock::now();

        std::vector<const char *> ptrs(texts.size());
        for (size_t i = 0; i < texts.size(); i++) ptrs[i] = texts[i].c_str();
        int d = 0;
        const float * vecs = crispembed_encode_batch(ctx, ptrs.data(), (int)texts.size(), &d);
        if (!vecs || d <= 0) {
            res.status = 500;
            res.set_content("{\"error\": \"encoding failed\"}", "application/json");
            return;
        }

        auto t1 = std::chrono::steady_clock::now();
        int64_t total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        std::ostringstream js;
        js << "{\"model\": \"" << json_escape(model_name) << "\", \"embeddings\": [";
        for (size_t i = 0; i < texts.size(); i++) {
            if (i > 0) js << ", ";
            js << "[";
            for (int j = 0; j < d; j++) {
                if (j > 0) js << ", ";
                js << vecs[i * d + j];
            }
            js << "]";
        }
        js << "], \"total_duration\": " << total_ns
           << ", \"load_duration\": 0"
           << ", \"prompt_eval_count\": " << texts.size() << "}";

        fprintf(stderr, "crispembed-server: /api/embed %zu text(s) in %.1f ms\n",
                texts.size(), total_ns / 1e6);
        res.set_content(js.str(), "application/json");
    });

    // POST /api/embeddings — Ollama-compatible (single text, legacy)
    // Request:  {"model": "...", "prompt": "text"}
    // Response: {"embedding": [...]}
    svr.Post("/api/embeddings", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no text model loaded\"}", "application/json");
            return;
        }
        std::string text;
        auto body = req.body;

        auto pos = body.find("\"prompt\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 8);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                text = body.substr(q1 + 1, q2 - q1 - 1);
        }

        if (text.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"no prompt provided\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(model_mutex);

        int d = 0;
        const float * vec = crispembed_encode(ctx, text.c_str(), &d);
        if (!vec || d <= 0) {
            res.status = 500;
            res.set_content("{\"error\": \"encoding failed\"}", "application/json");
            return;
        }

        std::ostringstream js;
        js << "{\"embedding\": [";
        for (int j = 0; j < d; j++) {
            if (j > 0) js << ", ";
            js << vec[j];
        }
        js << "]}";

        res.set_content(js.str(), "application/json");
    });

    // ── Face pipeline endpoints ───────────────────────────────────────
    crispembed_face_context * face_det = nullptr;
    crispembed_face_context * face_rec = nullptr;
    std::mutex face_mutex;

    if (!det_model_path.empty()) {
        face_det = crispembed_face_init(det_model_path.c_str(), n_threads);
        if (!face_det)
            fprintf(stderr, "Warning: failed to load detection model '%s'\n", det_model_path.c_str());
    }
    if (!rec_model_path.empty()) {
        face_rec = crispembed_face_init(rec_model_path.c_str(), n_threads);
        if (!face_rec)
            fprintf(stderr, "Warning: failed to load recognition model '%s'\n", rec_model_path.c_str());
    }

    // POST /detect — face detection
    // Request:  {"image": "/path/to/image.jpg", "conf": 0.5}
    // Response: {"faces": [{"bbox":[x,y,w,h], "conf":0.9, "landmarks":[...]}]}
    svr.Post("/detect", [&](const httplib::Request & req, httplib::Response & res) {
        if (!face_det) {
            res.status = 503;
            res.set_content("{\"error\": \"no detection model loaded (use --det)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        float conf = 0.5f;
        int det_size = 0;

        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        auto cpos = body.find("\"conf\"");
        if (cpos != std::string::npos) {
            auto start = body.find_first_of("0123456789.", cpos + 6);
            if (start != std::string::npos) conf = (float)atof(body.c_str() + start);
        }
        auto dspos = body.find("\"det_size\"");
        if (dspos != std::string::npos) {
            auto start = body.find_first_of("0123456789", dspos + 10);
            if (start != std::string::npos) det_size = atoi(body.c_str() + start);
        }

        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"no image path\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(face_mutex);
        int n = 0;
        auto * dets = crispembed_detect_faces(face_det, image_path.c_str(), conf, det_size, &n);

        std::ostringstream js;
        js << "{\"faces\": [";
        for (int i = 0; i < n; i++) {
            if (i > 0) js << ", ";
            js << "{\"bbox\":[" << dets[i].x << "," << dets[i].y
               << "," << dets[i].w << "," << dets[i].h
               << "], \"conf\":" << dets[i].confidence
               << ", \"landmarks\":[";
            for (int k = 0; k < 10; k++) {
                if (k > 0) js << ",";
                js << dets[i].landmarks[k];
            }
            js << "]}";
        }
        js << "]}";
        res.set_content(js.str(), "application/json");
    });

    // POST /face — full pipeline: detect + align + encode
    // Request:  {"image": "/path/to/image.jpg", "conf": 0.5}
    // Response: {"faces": [{"bbox":[...], "conf":..., "landmarks":[...], "embedding":[...]}]}
    svr.Post("/face", [&](const httplib::Request & req, httplib::Response & res) {
        if (!face_det || !face_rec) {
            res.status = 503;
            res.set_content("{\"error\": \"need both --det and --rec models\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        float conf = 0.5f;
        int det_size = 0;

        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        auto cpos = body.find("\"conf\"");
        if (cpos != std::string::npos) {
            auto start = body.find_first_of("0123456789.", cpos + 6);
            if (start != std::string::npos) conf = (float)atof(body.c_str() + start);
        }
        auto dspos = body.find("\"det_size\"");
        if (dspos != std::string::npos) {
            auto start = body.find_first_of("0123456789", dspos + 10);
            if (start != std::string::npos) det_size = atoi(body.c_str() + start);
        }

        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"no image path\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(face_mutex);
        auto t0 = std::chrono::steady_clock::now();
        int n = 0;
        auto * results = crispembed_face_pipeline(face_det, face_rec, image_path.c_str(), conf, det_size, &n);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::ostringstream js;
        js << "{\"faces\": [";
        for (int i = 0; i < n; i++) {
            if (i > 0) js << ", ";
            js << "{\"bbox\":[" << results[i].det.x << "," << results[i].det.y
               << "," << results[i].det.w << "," << results[i].det.h
               << "], \"conf\":" << results[i].det.confidence
               << ", \"landmarks\":[";
            for (int k = 0; k < 10; k++) {
                if (k > 0) js << ",";
                js << results[i].det.landmarks[k];
            }
            js << "], \"embedding\":[";
            for (int j = 0; j < results[i].embedding_dim; j++) {
                if (j > 0) js << ",";
                js << results[i].embedding[j];
            }
            js << "]}";
        }
        js << "], \"duration_ms\": " << ms << "}";

        fprintf(stderr, "crispembed-server: /face %d faces in %.1f ms\n", n, ms);
        res.set_content(js.str(), "application/json");
    });

    // ── Standalone ViT (SigLIP/CLIP) endpoints ─────────────────────────
    crispembed_vit_context * vit_ctx = nullptr;
    std::mutex vit_mutex;

    if (!vit_model_path.empty()) {
        vit_ctx = crispembed_vit_init(vit_model_path.c_str(), n_threads);
        if (!vit_ctx)
            fprintf(stderr, "Warning: failed to load ViT model '%s'\n", vit_model_path.c_str());
    }

    // POST /vit/encode — standalone ViT image embedding
    // Request:  {"image": "/path/to/image.jpg"}
    // Response: {"embedding": [...], "dim": N}
    svr.Post("/vit/encode", [&](const httplib::Request & req, httplib::Response & res) {
        if (!vit_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no ViT model loaded (use --vit)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;

        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }

        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"no image path\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(vit_mutex);
        auto t0 = std::chrono::steady_clock::now();

        int d = 0;
        const float * vec = crispembed_vit_encode_file(vit_ctx, image_path.c_str(), &d);
        if (!vec || d <= 0) {
            res.status = 500;
            res.set_content("{\"error\": \"ViT encoding failed\"}", "application/json");
            return;
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::ostringstream js;
        js << "{\"embedding\": [";
        for (int j = 0; j < d; j++) {
            if (j > 0) js << ", ";
            js << vec[j];
        }
        js << "], \"dim\": " << d << "}";

        fprintf(stderr, "crispembed-server: /vit/encode in %.1f ms (dim=%d)\n", ms, d);
        res.set_content(js.str(), "application/json");
    });

    // ── Pix2Struct (document understanding) ──
    crispembed_pix2struct_context * pix2struct_ctx = nullptr;
    std::mutex pix2struct_mutex;

    if (!pix2struct_model_path.empty()) {
        pix2struct_ctx = crispembed_pix2struct_init(pix2struct_model_path.c_str(), n_threads);
        if (!pix2struct_ctx)
            fprintf(stderr, "Warning: failed to load Pix2Struct model '%s'\n", pix2struct_model_path.c_str());
    }

    // POST /pix2struct/generate — document understanding image-to-text
    // Request:  {"image": "/path/to/doc.png", "max_tokens": 256}
    // Response: {"text": "...", "ms": M}
    svr.Post("/pix2struct/generate", [&](const httplib::Request & req, httplib::Response & res) {
        if (!pix2struct_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no Pix2Struct model loaded (use --pix2struct)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        int max_tokens = 256;

        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        auto mpos = body.find("\"max_tokens\"");
        if (mpos != std::string::npos) {
            auto colon = body.find(':', mpos + 12);
            if (colon != std::string::npos)
                max_tokens = atoi(body.c_str() + colon + 1);
        }

        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"no image path\"}", "application/json");
            return;
        }

        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 0);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(pix2struct_mutex);
        auto t0 = std::chrono::steady_clock::now();

        const char * text = crispembed_pix2struct_generate(pix2struct_ctx, data, w, h, max_tokens);
        stbi_image_free(data);

        if (!text) {
            res.status = 500;
            res.set_content("{\"error\": \"Pix2Struct generation failed\"}", "application/json");
            return;
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::ostringstream js;
        js << "{\"text\": \"" << json_escape(text)
           << "\", \"ms\": " << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /pix2struct/generate in %.1f ms\n", ms);
        crispembed_pix2struct_free_text(text);
        res.set_content(js.str(), "application/json");
    });

    // ── CLIP text encoder ──
    crispembed_clip_text_context * clip_text_ctx = nullptr;
    std::mutex clip_text_mutex;

    if (!clip_text_model_path.empty()) {
        clip_text_ctx = crispembed_clip_text_init(clip_text_model_path.c_str(), n_threads);
        if (!clip_text_ctx)
            fprintf(stderr, "Warning: failed to load CLIP text model '%s'\n", clip_text_model_path.c_str());
    }

    // ── Math OCR ──
    void * ocr_model_ctx = nullptr;
    std::mutex ocr_model_mutex;

    if (!ocr_model_path.empty()) {
        std::string resolved = crispembed_mgr::resolve_model(ocr_model_path, true);
        if (!resolved.empty()) ocr_model_path = resolved;
        ocr_model_ctx = crispembed_ocr_model_init(ocr_model_path.c_str(), n_threads);
        if (!ocr_model_ctx)
            fprintf(stderr, "Warning: failed to load math OCR model '%s'\n", ocr_model_path.c_str());
    }

    // ── General OCR Pipeline (text detection + recognition) ──
    void * ocr_pipeline_ctx = nullptr;
    std::mutex ocr_pipeline_mutex;

    // ── OCR Orchestrator (source-type routing + cleanup + accept-gate) ──
    void * ocr_orch_ctx = nullptr;
    std::mutex ocr_orch_mutex;

    if (enable_ocr_orch && !ocr_det_model_path.empty()) {
        crispembed_ocr_pipeline_params pp = crispembed_ocr_pipeline_defaults();
        std::string det_r = crispembed_mgr::resolve_model(ocr_det_model_path, true);
        if (!det_r.empty()) ocr_det_model_path = det_r;
        std::string rec_r = crispembed_mgr::resolve_model(ocr_rec_model_path, true);
        if (!rec_r.empty()) ocr_rec_model_path = rec_r;
        pp.det_model = ocr_det_model_path.c_str();
        pp.rec_model = ocr_rec_model_path.c_str();
        if (!vlm_model_path.empty()) {
            std::string vlm_r = crispembed_mgr::resolve_model(vlm_model_path, true);
            if (!vlm_r.empty()) vlm_model_path = vlm_r;
            pp.vlm_model = vlm_model_path.c_str();
            pp.vlm_engine = vlm_engine;
        }
        if (!punct_model_path.empty()) {
            std::string punct_r = crispembed_mgr::resolve_model(punct_model_path, true);
            if (!punct_r.empty()) punct_model_path = punct_r;
            pp.punct_model = punct_model_path.c_str();
        }
        if (!sr_model_path.empty()) {
            pp.sr_model = sr_model_path.c_str();
        }
        ocr_orch_ctx = crispembed_ocr_pipeline_init(&pp, n_threads);
        if (!ocr_orch_ctx)
            fprintf(stderr, "Warning: failed to init OCR orchestrator\n");
    }

    if (!ocr_det_model_path.empty() && !ocr_rec_model_path.empty()) {
        std::string det_resolved = crispembed_mgr::resolve_model(ocr_det_model_path, true);
        if (!det_resolved.empty()) ocr_det_model_path = det_resolved;
        std::string rec_resolved = crispembed_mgr::resolve_model(ocr_rec_model_path, true);
        if (!rec_resolved.empty()) ocr_rec_model_path = rec_resolved;
        ocr_pipeline_ctx = crispembed_ocr_init(ocr_det_model_path.c_str(),
                                                ocr_rec_model_path.c_str(), n_threads);
        if (!ocr_pipeline_ctx)
            fprintf(stderr, "Warning: failed to load OCR pipeline models\n");
    }

    // ── Layout Detection ──
    void * layout_ctx = nullptr;
    std::mutex layout_mutex;

    if (!layout_model_path.empty()) {
        std::string resolved = crispembed_mgr::resolve_model(layout_model_path, true);
        if (!resolved.empty()) layout_model_path = resolved;
        layout_ctx = crispembed_layout_init(layout_model_path.c_str(), n_threads);
        if (!layout_ctx)
            fprintf(stderr, "Warning: failed to load layout model '%s'\n", layout_model_path.c_str());
    }

    // Surya text detection
    void * text_det_ctx = nullptr;
    std::mutex text_det_mutex;

    if (!text_det_model_path.empty()) {
        std::string resolved = crispembed_mgr::resolve_model(text_det_model_path, true);
        if (!resolved.empty()) text_det_model_path = resolved;
        text_det_ctx = crispembed_text_det_init(text_det_model_path.c_str(), n_threads);
        if (!text_det_ctx)
            fprintf(stderr, "Warning: failed to load text detection model '%s'\n", text_det_model_path.c_str());
    }

    // NER (GLiNER)
    void * ner_ctx = nullptr;
    std::mutex ner_mutex;

    if (!ner_model_path.empty()) {
        std::string resolved = crispembed_mgr::resolve_model(ner_model_path, true);
        if (!resolved.empty()) ner_model_path = resolved;
        ner_ctx = crispembed_ner_init(ner_model_path.c_str(), n_threads);
        if (!ner_ctx)
            fprintf(stderr, "Warning: failed to load NER model '%s'\n", ner_model_path.c_str());
    }

    // Text LID
#if SERVER_HAS_LID
    text_lid_context * lid_ctx = nullptr;
    std::mutex lid_mutex;

    if (!lid_model_path.empty()) {
        std::string resolved = crispembed_mgr::resolve_model(lid_model_path, true);
        if (!resolved.empty()) lid_model_path = resolved;
        lid_ctx = text_lid_init_from_file(lid_model_path.c_str(), n_threads);
        if (!lid_ctx)
            fprintf(stderr, "Warning: failed to load LID model '%s'\n", lid_model_path.c_str());
    }
#endif

    // KIE (OCR + NER pipeline) — auto-enabled when NER + OCR det/rec are all loaded.
    void * kie_ctx = nullptr;
    std::mutex kie_mutex;

    if (ner_ctx && !ocr_det_model_path.empty() && !ocr_rec_model_path.empty()) {
        kie_ctx = crispembed_kie_init(
            ocr_det_model_path.c_str(), ocr_rec_model_path.c_str(),
            ner_model_path.c_str(), n_threads);
        if (!kie_ctx)
            fprintf(stderr, "Warning: failed to init KIE pipeline\n");
    }

    // ── Text Super-Resolution ──
    void * text_sr_ctx = nullptr;
    std::mutex text_sr_mutex;

    if (!sr_model_path.empty()) {
        text_sr_ctx = crispembed_text_sr_init(sr_model_path.c_str(), n_threads);
        if (!text_sr_ctx)
            fprintf(stderr, "Warning: failed to load text SR model '%s'\n", sr_model_path.c_str());
    }

    // ── PAN Super-Resolution ──
    void * pan_sr_ctx = nullptr;
    std::mutex pan_sr_mutex;

    if (!pan_model_path.empty()) {
        pan_sr_ctx = crispembed_pan_sr_init(pan_model_path.c_str(), n_threads);
        if (!pan_sr_ctx)
            fprintf(stderr, "Warning: failed to load PAN SR model '%s'\n", pan_model_path.c_str());
    }

    // ── HAT Super-Resolution ──
    void * hat_sr_ctx = nullptr;
    std::mutex hat_sr_mutex;

    if (!hat_model_path.empty()) {
        hat_sr_ctx = crispembed_hat_sr_init(hat_model_path.c_str(), n_threads);
        if (!hat_sr_ctx)
            fprintf(stderr, "Warning: failed to load HAT SR model '%s'\n", hat_model_path.c_str());
    }

    // ── DAT Super-Resolution ──
    void * dat_sr_ctx = nullptr;
    std::mutex dat_sr_mutex;

    if (!dat_model_path.empty()) {
        dat_sr_ctx = crispembed_dat_sr_init(dat_model_path.c_str(), n_threads);
        if (!dat_sr_ctx)
            fprintf(stderr, "Warning: failed to load DAT SR model '%s'\n", dat_model_path.c_str());
    }

    // ── SAFMN Super-Resolution ──
    void * safmn_sr_ctx = nullptr;
    std::mutex safmn_sr_mutex;

    if (!safmn_model_path.empty()) {
        safmn_sr_ctx = crispembed_safmn_sr_init(safmn_model_path.c_str(), n_threads);
        if (!safmn_sr_ctx)
            fprintf(stderr, "Warning: failed to load SAFMN SR model '%s'\n", safmn_model_path.c_str());
    }

    // ── Real-ESRGAN Super-Resolution ──
    void * esrgan_sr_ctx = nullptr;
    std::mutex esrgan_sr_mutex;

    if (!esrgan_model_path.empty()) {
        esrgan_sr_ctx = crispembed_esrgan_sr_init(esrgan_model_path.c_str(), n_threads);
        if (!esrgan_sr_ctx)
            fprintf(stderr, "Warning: failed to load Real-ESRGAN SR model '%s'\n", esrgan_model_path.c_str());
    }

    // ── SwinIR Super-Resolution ──
    void * swinir_sr_ctx = nullptr;
    std::mutex swinir_sr_mutex;

    if (!swinir_model_path.empty()) {
        swinir_sr_ctx = crispembed_swinir_sr_init(swinir_model_path.c_str(), n_threads);
        if (!swinir_sr_ctx)
            fprintf(stderr, "Warning: failed to load SwinIR SR model '%s'\n", swinir_model_path.c_str());
    }

    // ── TBSRN text-line Super-Resolution ──
    void * tbsrn_sr_ctx = nullptr;
    std::mutex tbsrn_sr_mutex;

    if (!tbsrn_model_path.empty()) {
        tbsrn_sr_ctx = crispembed_tbsrn_sr_init(tbsrn_model_path.c_str(), n_threads);
        if (!tbsrn_sr_ctx)
            fprintf(stderr, "Warning: failed to load TBSRN SR model '%s'\n", tbsrn_model_path.c_str());
    }

    // ── Restormer image restoration ──
    void * restormer_ctx = nullptr;
    std::mutex restormer_mutex;

    if (!restormer_model_path.empty()) {
        restormer_ctx = crispembed_restormer_init(restormer_model_path.c_str(), n_threads);
        if (!restormer_ctx)
            fprintf(stderr, "Warning: failed to load Restormer model '%s'\n", restormer_model_path.c_str());
    }

    // ── SCUNet image denoising ──
    void * scunet_ctx = nullptr;
    std::mutex scunet_mutex;

    if (!scunet_model_path.empty()) {
        scunet_ctx = crispembed_scunet_init(scunet_model_path.c_str(), n_threads);
        if (!scunet_ctx)
            fprintf(stderr, "Warning: failed to load SCUNet model '%s'\n", scunet_model_path.c_str());
    }

    // ── InstructIR all-in-one restoration ──
    void * instructir_ctx = nullptr;
    std::mutex instructir_mutex;

    if (!instructir_model_path.empty()) {
        instructir_ctx = crispembed_instructir_init(instructir_model_path.c_str(), n_threads);
        if (!instructir_ctx)
            fprintf(stderr, "Warning: failed to load InstructIR model '%s'\n", instructir_model_path.c_str());
    }

    // ── AdaIR all-in-one restoration ──
    void * adair_ctx = nullptr;
    std::mutex adair_mutex;

    if (!adair_model_path.empty()) {
        adair_ctx = crispembed_adair_init(adair_model_path.c_str(), n_threads);
        if (!adair_ctx)
            fprintf(stderr, "Warning: failed to load AdaIR model '%s'\n", adair_model_path.c_str());
    }

    // POST /clip/text — CLIP text encoding
    // Request:  {"text": "a photo of a cat"}
    // Response: {"embedding": [...], "dim": N}
    svr.Post("/clip/text", [&](const httplib::Request & req, httplib::Response & res) {
        if (!clip_text_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no CLIP text model loaded (use --clip-text)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string text;
        auto pos = body.find("\"text\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 6);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                text = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (text.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"no text\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(clip_text_mutex);
        auto t0 = std::chrono::steady_clock::now();

        int d = 0;
        const float * vec = crispembed_clip_text_encode(clip_text_ctx, text.c_str(), &d);
        if (!vec || d <= 0) {
            res.status = 500;
            res.set_content("{\"error\": \"CLIP text encoding failed\"}", "application/json");
            return;
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::ostringstream js;
        js << "{\"embedding\": [";
        for (int j = 0; j < d; j++) {
            if (j > 0) js << ", ";
            js << vec[j];
        }
        js << "], \"dim\": " << d << "}";

        fprintf(stderr, "crispembed-server: /clip/text in %.1f ms (dim=%d)\n", ms, d);
        res.set_content(js.str(), "application/json");
    });

    // POST /colbert/score — ColBERT late interaction scoring
    // Request:  {"query": "search text", "documents": ["doc1", "doc2", ...]}
    // Response: {"scores": [{"index": 0, "score": 12.5, "text": "doc1"}, ...], "ms": 5.2}
    svr.Post("/colbert/score", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no embedding model loaded\"}", "application/json");
            return;
        }
        if (!crispembed_has_colbert(ctx)) {
            res.status = 400;
            res.set_content("{\"error\": \"loaded model does not support ColBERT\"}", "application/json");
            return;
        }

        // Parse query text
        std::string query_text;
        {
            auto qp = req.body.find("\"query\"");
            if (qp != std::string::npos) {
                auto q1 = req.body.find('"', qp + 7);
                auto q2 = req.body.find('"', q1 + 1);
                if (q1 != std::string::npos && q2 != std::string::npos)
                    query_text = req.body.substr(q1 + 1, q2 - q1 - 1);
            }
        }
        if (query_text.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing query field\"}", "application/json");
            return;
        }

        // Parse documents array
        std::vector<std::string> doc_texts;
        {
            auto dp = req.body.find("\"documents\"");
            if (dp != std::string::npos) {
                auto arr_start = req.body.find('[', dp);
                if (arr_start != std::string::npos) {
                    auto arr_end = req.body.find(']', arr_start);
                    std::string arr = req.body.substr(arr_start + 1, arr_end - arr_start - 1);
                    size_t i = 0;
                    while (i < arr.size()) {
                        auto q1 = arr.find('"', i);
                        if (q1 == std::string::npos) break;
                        auto q2 = arr.find('"', q1 + 1);
                        if (q2 == std::string::npos) break;
                        doc_texts.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
                        i = q2 + 1;
                    }
                }
            }
        }
        if (doc_texts.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"no documents provided\"}", "application/json");
            return;
        }

        // Check if client wants SSE streaming
        bool stream = req.has_header("Accept") &&
                      req.get_header_value("Accept").find("text/event-stream") != std::string::npos;

        std::lock_guard<std::mutex> lock(model_mutex);
        auto t0 = std::chrono::high_resolution_clock::now();

        // Encode query
        int n_query = 0, qdim = 0;
        const float * query_vecs = crispembed_encode_multivec(ctx, query_text.c_str(), &n_query, &qdim);
        if (!query_vecs || n_query == 0) {
            res.status = 500;
            res.set_content("{\"error\": \"failed to encode query\"}", "application/json");
            return;
        }

        // Copy query vecs (encode_multivec invalidates previous pointer)
        std::vector<float> query_copy(query_vecs, query_vecs + n_query * qdim);

        if (stream) {
            // SSE streaming: emit each document score as it's computed
            auto shared_docs = std::make_shared<std::vector<std::string>>(std::move(doc_texts));
            auto shared_query = std::make_shared<std::vector<float>>(std::move(query_copy));
            auto shared_n_query = n_query;
            auto shared_qdim = qdim;
            auto shared_t0 = t0;

            res.set_chunked_content_provider("text/event-stream",
                [shared_docs, shared_query, shared_n_query, shared_qdim, shared_t0, &ctx, &model_mutex]
                (size_t /*offset*/, httplib::DataSink & sink) -> bool {
                    for (int i = 0; i < (int)shared_docs->size(); i++) {
                        int n_doc = 0, ddim = 0;
                        const float * doc_vecs = crispembed_encode_multivec(
                            ctx, (*shared_docs)[i].c_str(), &n_doc, &ddim);
                        float score = 0;
                        if (doc_vecs && n_doc > 0 && ddim == shared_qdim)
                            score = crispembed_colbert_score(
                                shared_query->data(), shared_n_query, doc_vecs, n_doc, shared_qdim);

                        // Escape text
                        std::string escaped;
                        for (char c : (*shared_docs)[i]) {
                            if (c == '"') escaped += "\\\"";
                            else if (c == '\\') escaped += "\\\\";
                            else if (c == '\n') escaped += "\\n";
                            else escaped += c;
                        }

                        std::ostringstream ev;
                        ev << "data: {\"index\": " << i
                           << ", \"score\": " << score
                           << ", \"text\": \"" << escaped << "\"}\n\n";
                        sink.write(ev.str().c_str(), ev.str().size());
                    }

                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ms = std::chrono::duration<double, std::milli>(t1 - shared_t0).count();
                    std::ostringstream ev;
                    ev << "event: done\ndata: {\"ms\": " << ms << "}\n\n";
                    sink.write(ev.str().c_str(), ev.str().size());
                    sink.done();
                    return true;
                });

            fprintf(stderr, "crispembed-server: /colbert/score SSE %d docs\n",
                    (int)shared_docs->size());
        } else {
            // Non-streaming: score all, sort, return JSON
            struct scored_doc { int index; float score; };
            std::vector<scored_doc> results;
            results.reserve(doc_texts.size());

            for (int i = 0; i < (int)doc_texts.size(); i++) {
                int n_doc = 0, ddim = 0;
                const float * doc_vecs = crispembed_encode_multivec(
                    ctx, doc_texts[i].c_str(), &n_doc, &ddim);
                if (!doc_vecs || n_doc == 0 || ddim != qdim) {
                    results.push_back({i, 0.0f});
                    continue;
                }
                float score = crispembed_colbert_score(
                    query_copy.data(), n_query, doc_vecs, n_doc, qdim);
                results.push_back({i, score});
            }

            std::sort(results.begin(), results.end(),
                      [](const scored_doc & a, const scored_doc & b) { return a.score > b.score; });

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::ostringstream js;
            js << "{\"scores\": [";
            for (size_t i = 0; i < results.size(); i++) {
                if (i > 0) js << ", ";
                std::string escaped;
                for (char c : doc_texts[results[i].index]) {
                    if (c == '"') escaped += "\\\"";
                    else if (c == '\\') escaped += "\\\\";
                    else if (c == '\n') escaped += "\\n";
                    else escaped += c;
                }
                js << "{\"index\": " << results[i].index
                   << ", \"score\": " << results[i].score
                   << ", \"text\": \"" << escaped << "\""
                   << "}";
            }
            js << "], \"ms\": " << ms << "}";

            fprintf(stderr, "crispembed-server: /colbert/score %d docs in %.1f ms\n",
                    (int)doc_texts.size(), ms);
            res.set_content(js.str(), "application/json");
        }
    });

    // POST /ocr/model — single-model OCR recognition (math or text, auto-dispatched).
    // /math/ocr is kept as a deprecated alias for the same handler.
    // Request:  {"image": "/path/to/formula.png"}
    // Response: {"latex": "\\frac{a}{b}", "len": 12, "ms": 450.2}
    auto ocr_model_handler = [&](const httplib::Request & req, httplib::Response & res) {
        if (!ocr_model_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no OCR model loaded (use --ocr)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"no image path\"}", "application/json");
            return;
        }

        int req_max_tokens = 0;
        auto mt_pos = body.find("\"max_tokens\"");
        if (mt_pos != std::string::npos) {
            auto colon = body.find(':', mt_pos + 12);
            if (colon != std::string::npos)
                req_max_tokens = std::atoi(body.c_str() + colon + 1);
        }

        std::lock_guard<std::mutex> lock(ocr_model_mutex);
        if (req_max_tokens > 0) crispembed_ocr_model_set_max_tokens(ocr_model_ctx, req_max_tokens);
        auto t0 = std::chrono::steady_clock::now();

        int w = 0, h = 0, ch = 0;
        unsigned char * pixels = stbi_load(image_path.c_str(), &w, &h, &ch, 0);
        if (!pixels) {
            res.status = 400;
            res.set_content("{\"error\": \"failed to load image\"}", "application/json");
            return;
        }

        int out_len = 0;
        const char * latex = crispembed_ocr_model_recognize(ocr_model_ctx, pixels, w, h, ch, &out_len);
        stbi_image_free(pixels);

        // Fetch per-token and mean confidence scores after recognition.
        float mean_conf = crispembed_ocr_model_mean_confidence(ocr_model_ctx);
        int n_tok_conf = 0;
        const float * tok_conf = crispembed_ocr_model_confidences(ocr_model_ctx, &n_tok_conf);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::ostringstream js;
        js << "{\"latex\": \"" << json_escape(latex ? latex : "") << "\", \"len\": " << out_len
           << ", \"confidence\": " << std::fixed << std::setprecision(4) << mean_conf
           << ", \"token_confidences\": [";
        for (int i = 0; i < n_tok_conf; i++) {
            if (i > 0) js << ",";
            js << std::fixed << std::setprecision(4) << tok_conf[i];
        }
        js << "]"
           << ", \"ms\": " << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /ocr/model in %.1f ms (%d chars, conf=%.2f)\n", ms, out_len, mean_conf);
        res.set_content(js.str(), "application/json");
    };
    svr.Post("/ocr/model", ocr_model_handler);
    svr.Post("/math/ocr", ocr_model_handler);  // deprecated alias

    // POST /ocr — general text detection + recognition pipeline
    // Request:  {"image": "/path/to/document.png"}
    // Response: {"results": [{"text": "Hello", "bbox": [x,y,x2,y2], "confidence": 0.99}], "ms": M}
    svr.Post("/ocr", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ocr_pipeline_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"OCR pipeline not loaded (use --ocr-det + --ocr-rec)\"}", "application/json");
            return;
        }
        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        std::lock_guard<std::mutex> lock(ocr_pipeline_mutex);
        int n_results = 0;
        const crispembed_ocr_result* results = crispembed_ocr(ocr_pipeline_ctx, image_path.c_str(), &n_results);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::ostringstream js;
        js << "{\"results\": [";
        for (int i = 0; i < n_results; i++) {
            if (i > 0) js << ",";
            js << "{\"text\":\"" << json_escape(results[i].text) << "\""
               << ",\"bbox\":[" << results[i].x << "," << results[i].y
               << "," << (results[i].x + results[i].w) << "," << (results[i].y + results[i].h) << "]"
               << ",\"confidence\":" << results[i].confidence
               << ",\"rec_confidence\":" << results[i].confidence << "}";
        }
        js << "],\"n\":" << n_results << ",\"ms\":" << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /ocr in %.1f ms (%d regions)\n", ms, n_results);
        res.set_content(js.str(), "application/json");
    });

    // POST /ocr/pipeline — full orchestrator (routing + cleanup + accept-gate)
    // Request:  {"image": "/path/to/doc.png", "min_chars": 8, "min_confidence": 0.5}
    // Response: {"text": "...", "n_regions": N, "mean_confidence": F, "ms": M}
    // Note: per-request params override the context defaults when a fresh context
    // is built per call. Currently uses the shared ocr_orch_ctx (init-time config).
    svr.Post("/ocr/pipeline", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ocr_orch_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"OCR orchestrator not loaded (use --ocr-pipeline --ocr-det MODEL --ocr-rec MODEL)\"}", "application/json");
            return;
        }
        auto body = req.body;
        // Parse "image" (string)
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        std::lock_guard<std::mutex> lock(ocr_orch_mutex);
        int n_results = 0;
        const char * full_text = nullptr;
        float mean_conf = 0.0f;
        const crispembed_ocr_result * results =
            crispembed_ocr_pipeline_run(ocr_orch_ctx, image_path.c_str(),
                                        &n_results, &full_text, &mean_conf);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::ostringstream js;
        js << "{\"text\":\"" << json_escape(full_text ? full_text : "") << "\""
           << ",\"n_regions\":" << n_results
           << ",\"mean_confidence\":" << std::fixed << std::setprecision(4) << mean_conf
           << ",\"results\":[";
        for (int i = 0; i < n_results; i++) {
            if (i > 0) js << ",";
            float rec_conf = crispembed_ocr_pipeline_region_rec_confidence(ocr_orch_ctx, i);
            js << "{\"text\":\"" << json_escape(results[i].text) << "\""
               << ",\"bbox\":[" << results[i].x << "," << results[i].y
               << "," << (results[i].x + results[i].w) << "," << (results[i].y + results[i].h) << "]"
               << ",\"confidence\":" << results[i].confidence
               << ",\"rec_confidence\":" << std::fixed << std::setprecision(4) << rec_conf << "}";
        }
        js << "],\"ms\":" << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /ocr/pipeline in %.1f ms (%d regions, conf=%.2f)\n",
                ms, n_results, mean_conf);
        res.set_content(js.str(), "application/json");
    });

    // POST /layout/detect — document layout analysis
    // Request:  {"image": "/path/to/page.png", "threshold": 0.3}
    // Response: {"regions": [{"label": "text", "score": 0.95, "bbox": [x1, y1, x2, y2]}, ...]}
    svr.Post("/layout/detect", [&](const httplib::Request & req, httplib::Response & res) {
        if (!layout_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no layout model loaded (use --layout)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        float threshold = 0.3f;

        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        auto tpos = body.find("\"threshold\"");
        if (tpos != std::string::npos) {
            auto colon = body.find(':', tpos);
            if (colon != std::string::npos)
                threshold = (float)atof(body.c_str() + colon + 1);
        }

        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"no image path\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(layout_mutex);
        auto t0 = std::chrono::steady_clock::now();

        int n_regions = 0;
        const crispembed_layout_region * regions = crispembed_layout_detect(
            layout_ctx, image_path.c_str(), threshold, &n_regions);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::ostringstream js;
        js << "{\"regions\": [";
        for (int i = 0; i < n_regions; i++) {
            if (i > 0) js << ", ";
            js << "{\"label\": \"" << regions[i].label_name
               << "\", \"score\": " << std::fixed << std::setprecision(4) << regions[i].score
               << ", \"bbox\": [" << std::setprecision(1)
               << regions[i].x1 << ", " << regions[i].y1 << ", "
               << regions[i].x2 << ", " << regions[i].y2 << "]}";
        }
        js << "], \"ms\": " << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /layout/detect in %.1f ms (%d regions)\n", ms, n_regions);
        res.set_content(js.str(), "application/json");
    });

    // POST /table/parse — rule-based table structure recognition
    // Request:  {"image": "/path/to/table.png"}
    // Response: {"html": "<table>...</table>", "ms": T}
    svr.Post("/table/parse", [&](const httplib::Request & req, httplib::Response & res) {
        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"no image path\"}", "application/json");
            return;
        }

        auto t0 = std::chrono::steady_clock::now();

        // Load image as grayscale
        int w = 0, h = 0, ch = 0;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 1);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        void * tctx = crispembed_table_parse_init(nullptr, n_threads);
        char * html = tctx ? crispembed_table_parse_to_html(tctx, data, w, h) : nullptr;
        stbi_image_free(data);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (html) {
            std::ostringstream js;
            js << "{\"html\": \"" << json_escape(html)
               << "\", \"ms\": " << std::fixed << std::setprecision(1) << ms << "}";
            res.set_content(js.str(), "application/json");
            crispembed_table_parse_free_string(html);
        } else {
            res.set_content("{\"error\": \"table parsing failed\"}", "application/json");
        }
        if (tctx) crispembed_table_parse_free(tctx);

        fprintf(stderr, "crispembed-server: /table/parse in %.1f ms\n", ms);
    });

    // POST /text/detect — surya text line detection
    // Request:  {"image": "/path/to/page.png", "threshold": 0.6}
    // Response: {"boxes": [{"bbox": [x0, y0, x1, y1], "confidence": 0.95}, ...]}
    svr.Post("/text/detect", [&](const httplib::Request & req, httplib::Response & res) {
        if (!text_det_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no text detection model loaded (use --text-det)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        float threshold = 0.6f;

        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        auto tpos = body.find("\"threshold\"");
        if (tpos != std::string::npos) {
            auto colon = body.find(':', tpos);
            if (colon != std::string::npos)
                threshold = (float)atof(body.c_str() + colon + 1);
        }

        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"no image path\"}", "application/json");
            return;
        }

        // Load image via stb_image
        int w, h, ch;
        unsigned char * px = stbi_load(image_path.c_str(), &w, &h, &ch, 3);
        if (!px) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(text_det_mutex);
        auto t0 = std::chrono::steady_clock::now();

        int n_boxes = 0;
        const crispembed_text_det_result * boxes = crispembed_text_det(
            text_det_ctx, px, w, h, 3, threshold, 0.35f, &n_boxes);
        stbi_image_free(px);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::ostringstream js;
        js << "{\"boxes\": [";
        for (int i = 0; i < n_boxes; i++) {
            if (i > 0) js << ", ";
            js << "{\"bbox\": [" << std::fixed << std::setprecision(1)
               << boxes[i].x0 << ", " << boxes[i].y0 << ", "
               << boxes[i].x1 << ", " << boxes[i].y1
               << "], \"confidence\": " << std::setprecision(4) << boxes[i].confidence << "}";
        }
        js << "], \"ms\": " << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /text/detect in %.1f ms (%d boxes)\n", ms, n_boxes);
        res.set_content(js.str(), "application/json");
    });

    // POST /ner/extract — zero-shot named entity recognition
    // Request:  {"text": "Barack Obama was born in Hawaii", "labels": ["person", "location"], "threshold": 0.5}
    // Response: {"entities": [{"text": "Barack Obama", "label": "person", "start": 0, "end": 12, "score": 0.56}]}
    svr.Post("/ner/extract", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ner_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no NER model loaded (use --ner)\"}", "application/json");
            return;
        }

        auto body = req.body;

        // Parse "text"
        std::string text;
        {
            auto pos = body.find("\"text\"");
            if (pos != std::string::npos) {
                auto q1 = body.find('"', pos + 6);
                auto q2 = body.find('"', q1 + 1);
                if (q1 != std::string::npos && q2 != std::string::npos)
                    text = body.substr(q1 + 1, q2 - q1 - 1);
            }
        }

        // Parse "labels" array
        std::vector<std::string> labels;
        {
            auto pos = body.find("\"labels\"");
            if (pos != std::string::npos) {
                auto arr_start = body.find('[', pos);
                auto arr_end = body.find(']', arr_start);
                if (arr_start != std::string::npos && arr_end != std::string::npos) {
                    std::string arr = body.substr(arr_start + 1, arr_end - arr_start - 1);
                    size_t i = 0;
                    while (i < arr.size()) {
                        auto q1 = arr.find('"', i);
                        if (q1 == std::string::npos) break;
                        auto q2 = arr.find('"', q1 + 1);
                        if (q2 == std::string::npos) break;
                        labels.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
                        i = q2 + 1;
                    }
                }
            }
        }

        // Parse "threshold"
        float threshold = 0.5f;
        {
            auto pos = body.find("\"threshold\"");
            if (pos != std::string::npos) {
                auto colon = body.find(':', pos);
                if (colon != std::string::npos)
                    threshold = (float)atof(body.c_str() + colon + 1);
            }
        }

        if (text.empty() || labels.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"provide \\\"text\\\" and \\\"labels\\\" fields\"}", "application/json");
            return;
        }

        std::vector<const char *> label_ptrs(labels.size());
        for (size_t i = 0; i < labels.size(); i++)
            label_ptrs[i] = labels[i].c_str();

        std::lock_guard<std::mutex> lock(ner_mutex);
        auto t0 = std::chrono::steady_clock::now();

        crispembed_ner_entity * entities = nullptr;
        int n = crispembed_ner_extract(ner_ctx, text.c_str(),
                                       label_ptrs.data(), (int)labels.size(),
                                       threshold, &entities);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::ostringstream js;
        js << "{\"entities\": [";
        for (int i = 0; i < n; i++) {
            if (i > 0) js << ", ";
            js << "{\"text\": \"" << json_escape(entities[i].text ? entities[i].text : "")
               << "\", \"label\": \"" << json_escape(entities[i].label ? entities[i].label : "")
               << "\", \"start\": " << entities[i].start_char
               << ", \"end\": " << entities[i].end_char
               << ", \"score\": " << std::fixed << std::setprecision(4) << entities[i].score
               << "}";
        }
        js << "], \"ms\": " << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /ner/extract in %.1f ms (%d entities)\n", ms, n);
        res.set_content(js.str(), "application/json");
    });

    // POST /lid/detect — text language identification
    // Request:  {"text": "Hallo Welt, wie geht es Ihnen?"}
    // Response: {"lang": "de", "confidence": 0.99, "ms": T}
#if SERVER_HAS_LID
    svr.Post("/lid/detect", [&](const httplib::Request & req, httplib::Response & res) {
        if (!lid_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no LID model loaded (use --lid)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string text;
        auto pos = body.find("\"text\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 6);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                text = body.substr(q1 + 1, q2 - q1 - 1);
        }

        if (text.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"provide \\\"text\\\" field\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(lid_mutex);
        auto t0 = std::chrono::steady_clock::now();

        float conf = 0.0f;
        const char * lang = text_lid_predict(lid_ctx, text.c_str(), &conf);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::ostringstream js;
        js << "{\"lang\": \"" << (lang ? lang : "") << "\""
           << ", \"confidence\": " << std::fixed << std::setprecision(4) << conf
           << ", \"ms\": " << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /lid/detect → %s (%.2f) in %.1f ms\n",
                lang ? lang : "?", conf, ms);
        res.set_content(js.str(), "application/json");
    });
#endif

    // POST /kie/extract — key information extraction (OCR + NER)
    // Request:  {"image": "/path/to/doc.png", "labels": ["total", "date", "vendor"], "threshold": 0.5}
    // Response: {"fields": [{label, value, score, bbox}], "ocr_text": "...", "ocr_confidence": 0.85, "n_ocr_regions": 12}
    svr.Post("/kie/extract", [&](const httplib::Request & req, httplib::Response & res) {
        if (!kie_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"KIE not available (need --ner + --ocr-det + --ocr-rec)\"}", "application/json");
            return;
        }

        auto body = req.body;

        // Parse "image"
        std::string image_path;
        {
            auto pos = body.find("\"image\"");
            if (pos != std::string::npos) {
                auto q1 = body.find('"', pos + 7);
                auto q2 = body.find('"', q1 + 1);
                if (q1 != std::string::npos && q2 != std::string::npos)
                    image_path = body.substr(q1 + 1, q2 - q1 - 1);
            }
        }

        // Parse "labels" array
        std::vector<std::string> labels;
        {
            auto pos = body.find("\"labels\"");
            if (pos != std::string::npos) {
                auto arr_start = body.find('[', pos);
                auto arr_end = body.find(']', arr_start);
                if (arr_start != std::string::npos && arr_end != std::string::npos) {
                    std::string arr = body.substr(arr_start + 1, arr_end - arr_start - 1);
                    size_t i = 0;
                    while (i < arr.size()) {
                        auto q1 = arr.find('"', i);
                        if (q1 == std::string::npos) break;
                        auto q2 = arr.find('"', q1 + 1);
                        if (q2 == std::string::npos) break;
                        labels.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
                        i = q2 + 1;
                    }
                }
            }
        }

        // Parse "threshold"
        float threshold = 0.5f;
        {
            auto pos = body.find("\"threshold\"");
            if (pos != std::string::npos) {
                auto colon = body.find(':', pos);
                if (colon != std::string::npos)
                    threshold = (float)atof(body.c_str() + colon + 1);
            }
        }

        if (image_path.empty() || labels.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"provide \\\"image\\\" and \\\"labels\\\" fields\"}", "application/json");
            return;
        }

        std::vector<const char *> label_ptrs(labels.size());
        for (size_t i = 0; i < labels.size(); i++)
            label_ptrs[i] = labels[i].c_str();

        std::lock_guard<std::mutex> lock(kie_mutex);
        auto t0 = std::chrono::steady_clock::now();

        crispembed_kie_result kr = crispembed_kie_extract(
            kie_ctx, image_path.c_str(),
            label_ptrs.data(), (int)labels.size(), threshold);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::ostringstream js;
        js << "{\"fields\": [";
        for (int i = 0; i < kr.n_fields; i++) {
            if (i > 0) js << ", ";
            js << "{\"label\": \"" << json_escape(kr.fields[i].label ? kr.fields[i].label : "")
               << "\", \"value\": \"" << json_escape(kr.fields[i].value ? kr.fields[i].value : "")
               << "\", \"score\": " << std::fixed << std::setprecision(4) << kr.fields[i].score
               << ", \"bbox\": [" << std::setprecision(1) << kr.fields[i].x << ", "
               << kr.fields[i].y << ", " << kr.fields[i].w << ", " << kr.fields[i].h << "]}";
        }
        js << "], \"ocr_text\": \"" << json_escape(kr.ocr_text ? kr.ocr_text : "")
           << "\", \"ocr_confidence\": " << std::setprecision(3) << kr.ocr_confidence
           << ", \"n_ocr_regions\": " << kr.n_ocr_regions
           << ", \"ms\": " << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /kie/extract in %.1f ms (%d fields)\n", ms, kr.n_fields);
        res.set_content(js.str(), "application/json");
    });

    // POST /scan/cleanup — document scan preprocessing (no model needed)
    // Request:  {"image": "/path/to/scan.png", "deskew": true, "binarize": false}
    // Response: {"width": W, "height": H, "original_width": OW, "original_height": OH, "ms": T}
    svr.Post("/scan/cleanup", [&](const httplib::Request & req, httplib::Response & res) {
        auto body = req.body;
        std::string image_path;

        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }

        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }

        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 0);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        auto t0 = std::chrono::steady_clock::now();

        auto params = scan_cleanup_defaults();
        // Parse optional params from JSON
        auto parse_bool = [&](const char * key) -> int {
            auto p = body.find(key);
            if (p == std::string::npos) return -1;
            auto c = body.find(':', p);
            if (c == std::string::npos) return -1;
            auto v = body.find_first_not_of(" \t\n", c + 1);
            if (v == std::string::npos) return -1;
            return (body[v] == 't' || body[v] == '1') ? 1 : 0;
        };
        int v;
        if ((v = parse_bool("\"deskew\"")) >= 0) params.deskew = v;
        if ((v = parse_bool("\"crop_borders\"")) >= 0) params.crop_borders = v;
        if ((v = parse_bool("\"whiten_background\"")) >= 0) params.whiten_background = v;
        if ((v = parse_bool("\"binarize\"")) >= 0) params.binarize = v;

        auto * sctx = scan_cleanup_init(nullptr, 4);
        uint8_t * out = nullptr;
        int ow = 0, oh = 0;
        int rc = scan_cleanup_process(sctx, data, w, h, ch, params, &out, &ow, &oh);
        scan_cleanup_free(sctx);
        stbi_image_free(data);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0 || !out) {
            res.status = 500;
            res.set_content("{\"error\": \"scan cleanup failed\"}", "application/json");
            return;
        }

        scan_cleanup_free_image(out);

        std::ostringstream js;
        js << "{\"width\": " << ow << ", \"height\": " << oh
           << ", \"original_width\": " << w << ", \"original_height\": " << h
           << ", \"ms\": " << std::setprecision(1) << std::fixed << ms << "}";

        fprintf(stderr, "crispembed-server: /scan/cleanup in %.1f ms (%dx%d -> %dx%d)\n", ms, w, h, ow, oh);
        res.set_content(js.str(), "application/json");
    });

    // POST /pdf/dpi — PDF DPI profiling (no model needed)
    svr.Post("/pdf/dpi", [&](const httplib::Request & req, httplib::Response & res) {
        auto body = req.body;
        std::string file_path;
        auto pos = body.find("\"file\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 6);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                file_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (file_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'file' field\"}", "application/json");
            return;
        }
        int n_pages = 0;
        pdf_page_dpi_result * results = pdf_all_pages_dpi(file_path.c_str(), &n_pages);
        if (!results || n_pages <= 0) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot read PDF\"}", "application/json");
            return;
        }
        std::ostringstream js;
        js << "{\"pages\":[";
        for (int i = 0; i < n_pages; i++) {
            if (i > 0) js << ",";
            js << "{\"page\":" << i
               << ",\"dpi\":" << results[i].dpi
               << ",\"dpi_min\":" << results[i].dpi_min
               << ",\"dpi_max\":" << results[i].dpi_max
               << ",\"n_images\":" << results[i].n_images
               << ",\"page_width_pt\":" << results[i].page_width_pt
               << ",\"page_height_pt\":" << results[i].page_height_pt
               << "}";
        }
        js << "]}";
        pdf_dpi_free(results);
        res.set_content(js.str(), "application/json");
    });

    // POST /preprocess/skew — find skew angle (no model needed)
    svr.Post("/preprocess/skew", [&](const httplib::Request & req, httplib::Response & res) {
        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }
        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 1);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }
        float angle = 0, conf = 0;
        crispembed_find_skew(data, w, h, &angle, &conf);
        stbi_image_free(data);
        char buf[128];
        snprintf(buf, sizeof(buf), "{\"angle\":%.3f,\"confidence\":%.3f}", angle, conf);
        res.set_content(buf, "application/json");
    });

    // POST /preprocess/dewarp — straighten curved text (no model needed)
    // Request:  {"image": "/path/to/scan.png", "output": "/path/to/out.pgm"}
    //   - "output" is optional; if present, writes dewarped PGM to that path
    // Response: JSON metadata + optionally raw PGM image bytes
    //   - If Accept: image/* → returns PGM binary directly
    //   - Otherwise → JSON {"dewarped":bool, "width":W, "height":H}
    svr.Post("/preprocess/dewarp", [&](const httplib::Request & req, httplib::Response & res) {
        auto body = req.body;
        std::string image_path, output_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        auto opos = body.find("\"output\"");
        if (opos != std::string::npos) {
            auto q1 = body.find('"', opos + 8);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                output_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }
        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 1);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }
        std::vector<uint8_t> out(w * h);
        int ow = 0, oh = 0;
        int ret = crispembed_dewarp(data, w, h, out.data(), &ow, &oh);
        stbi_image_free(data);
        if (ret != 0) {
            res.set_content("{\"dewarped\":false,\"reason\":\"too few textlines\"}", "application/json");
            return;
        }
        // Write to output file if requested
        if (!output_path.empty()) {
            FILE * f = fopen(output_path.c_str(), "wb");
            if (f) {
                fprintf(f, "P5\n%d %d\n255\n", ow, oh);
                fwrite(out.data(), 1, ow * oh, f);
                fclose(f);
            }
        }
        // Return image or JSON based on Accept header
        bool want_image = false;
        if (req.has_header("Accept")) {
            auto accept = req.get_header_value("Accept");
            want_image = accept.find("image/") != std::string::npos;
        }
        if (want_image) {
            // PGM format: P5 header + raw bytes
            std::string pgm;
            char hdr[64];
            snprintf(hdr, sizeof(hdr), "P5\n%d %d\n255\n", ow, oh);
            pgm = hdr;
            pgm.append((const char *)out.data(), ow * oh);
            res.set_content(pgm, "image/x-portable-graymap");
        } else {
            char buf[256];
            snprintf(buf, sizeof(buf),
                "{\"dewarped\":true,\"width\":%d,\"height\":%d,\"output\":\"%s\"}",
                ow, oh, output_path.empty() ? "" : output_path.c_str());
            res.set_content(buf, "application/json");
        }
    });

    // POST /preprocess/tps-dewarp — TPS-based dewarp (learned localizer model)
    svr.Post("/preprocess/tps-dewarp", [&](const httplib::Request & req, httplib::Response & res) {
        auto body = req.body;
        std::string image_path, model_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        pos = body.find("\"model\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                model_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty() || model_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' or 'model' field\"}", "application/json");
            return;
        }
        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 1);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }
        std::vector<uint8_t> out(w * h);
        int ret = crispembed_tps_auto_dewarp(data, w, h, model_path.c_str(), out.data());
        stbi_image_free(data);
        if (ret != 0) {
            res.set_content("{\"dewarped\":false,\"reason\":\"tps-dewarp failed\"}", "application/json");
            return;
        }
        char buf[128];
        snprintf(buf, sizeof(buf), "{\"dewarped\":true,\"width\":%d,\"height\":%d}", w, h);
        res.set_content(buf, "application/json");
    });

    // POST /preprocess/cc-detect — model-free text line detection
    svr.Post("/preprocess/cc-detect", [&](const httplib::Request & req, httplib::Response & res) {
        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }
        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 1);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }
        int n = 0;
        crispembed_ocr_result * regions = crispembed_cc_detect(data, w, h, &n);
        stbi_image_free(data);
        std::ostringstream js;
        js << "{\"n\":" << n << ",\"regions\":[";
        for (int i = 0; i < n; i++) {
            if (i > 0) js << ",";
            js << "{\"x\":" << regions[i].x << ",\"y\":" << regions[i].y
               << ",\"w\":" << regions[i].w << ",\"h\":" << regions[i].h << "}";
        }
        js << "]}";
        if (regions) free(regions);
        res.set_content(js.str(), "application/json");
    });

    // POST /render/ocr — render OCR results to hOCR/ALTO/PDF
    // Request: {"results": [{"text":"Hello","x":10,"y":20,"w":50,"h":15,"confidence":0.99},...],
    //           "page_width": 800, "page_height": 600, "format": "hocr"}
    // Response: rendered document (hOCR XHTML, ALTO XML, or PDF binary)
    svr.Post("/render/ocr", [&](const httplib::Request & req, httplib::Response & res) {
        auto body = req.body;

        // Parse format
        std::string format = "text";
        auto fpos = body.find("\"format\"");
        if (fpos != std::string::npos) {
            auto q1 = body.find('"', fpos + 8);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                format = body.substr(q1 + 1, q2 - q1 - 1);
        }

        // Parse page dimensions
        int page_w = 800, page_h = 600;
        auto pw_pos = body.find("\"page_width\"");
        if (pw_pos != std::string::npos) {
            auto colon = body.find(':', pw_pos);
            if (colon != std::string::npos) page_w = atoi(body.c_str() + colon + 1);
        }
        auto ph_pos = body.find("\"page_height\"");
        if (ph_pos != std::string::npos) {
            auto colon = body.find(':', ph_pos);
            if (colon != std::string::npos) page_h = atoi(body.c_str() + colon + 1);
        }

        // Parse results array (minimal JSON — extract text + bbox per entry)
        // Each result: {"text":"...","x":N,"y":N,"w":N,"h":N,"confidence":F}
        std::vector<crispembed_ocr_result> results;
        std::vector<std::string> texts; // keep text alive
        auto rpos = body.find("\"results\"");
        if (rpos != std::string::npos) {
            auto arr_start = body.find('[', rpos);
            auto arr_end = body.rfind(']');
            if (arr_start != std::string::npos && arr_end != std::string::npos) {
                std::string arr = body.substr(arr_start, arr_end - arr_start + 1);
                // Parse each {...} object
                size_t p = 0;
                while ((p = arr.find('{', p)) != std::string::npos) {
                    auto e = arr.find('}', p);
                    if (e == std::string::npos) break;
                    std::string obj = arr.substr(p, e - p + 1);
                    p = e + 1;

                    crispembed_ocr_result r = {};
                    // Extract text
                    auto tp = obj.find("\"text\"");
                    if (tp != std::string::npos) {
                        auto tq1 = obj.find('"', tp + 6);
                        auto tq2 = obj.find('"', tq1 + 1);
                        if (tq1 != std::string::npos && tq2 != std::string::npos)
                            texts.push_back(obj.substr(tq1 + 1, tq2 - tq1 - 1));
                        else texts.push_back("");
                    } else texts.push_back("");
                    r.text = texts.back().c_str();
                    r.text_len = (int)texts.back().size();
                    // Extract numbers
                    auto parse_num = [&](const char * key) -> float {
                        auto kp = obj.find(key);
                        if (kp == std::string::npos) return 0;
                        auto cp = obj.find(':', kp);
                        return cp != std::string::npos ? (float)atof(obj.c_str() + cp + 1) : 0;
                    };
                    r.x = parse_num("\"x\"");
                    r.y = parse_num("\"y\"");
                    r.w = parse_num("\"w\"");
                    r.h = parse_num("\"h\"");
                    r.confidence = parse_num("\"confidence\"");
                    if (r.confidence == 0) r.confidence = 1.0f;
                    results.push_back(r);
                }
            }
        }

        if (results.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"no results to render\"}", "application/json");
            return;
        }

        char * rendered = crispembed_ocr_render(
            results.data(), (int)results.size(), page_w, page_h, format.c_str());
        if (!rendered) {
            res.status = 500;
            res.set_content("{\"error\": \"rendering failed\"}", "application/json");
            return;
        }

        // Set content type based on format
        const char * content_type = "text/plain";
        if (format == "hocr") content_type = "text/html; charset=utf-8";
        else if (format == "alto") content_type = "application/xml; charset=utf-8";
        else if (format == "pdf") content_type = "application/pdf";

        res.set_content(rendered, content_type);
        free(rendered);
    });

    // POST /text/sr — text image super-resolution (upscale low-DPI text before OCR)
    svr.Post("/text/sr", [&](const httplib::Request & req, httplib::Response & res) {
        if (!text_sr_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no text SR model loaded (use --sr-model)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }

        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 3);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(text_sr_mutex);
        auto t0 = std::chrono::steady_clock::now();

        uint8_t * out = nullptr;
        int ow = 0, oh = 0;
        int rc = crispembed_text_sr_process(
            text_sr_ctx, data, w, h,
            /*tile_size=*/0, /*tile_overlap=*/0,
            &out, &ow, &oh);
        stbi_image_free(data);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0 || !out) {
            res.status = 500;
            res.set_content("{\"error\": \"text SR processing failed\"}", "application/json");
            return;
        }

        // Base64-encode the raw RGB output
        static const char b64chars[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const size_t n_bytes = (size_t)ow * oh * 3;
        std::string b64;
        b64.reserve(((n_bytes + 2) / 3) * 4);
        for (size_t i = 0; i < n_bytes; i += 3) {
            uint32_t v = (uint32_t)out[i] << 16;
            if (i + 1 < n_bytes) v |= (uint32_t)out[i + 1] << 8;
            if (i + 2 < n_bytes) v |= (uint32_t)out[i + 2];
            b64 += b64chars[(v >> 18) & 0x3f];
            b64 += b64chars[(v >> 12) & 0x3f];
            b64 += (i + 1 < n_bytes) ? b64chars[(v >> 6) & 0x3f] : '=';
            b64 += (i + 2 < n_bytes) ? b64chars[v & 0x3f] : '=';
        }
        crispembed_text_sr_free_image(out);

        const int scale = crispembed_text_sr_upscale_factor(text_sr_ctx);

        std::ostringstream js;
        js << "{\"image\": \"" << b64 << "\""
           << ", \"width\": " << ow << ", \"height\": " << oh
           << ", \"original_width\": " << w << ", \"original_height\": " << h
           << ", \"upscale_factor\": " << scale
           << ", \"ms\": " << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /text/sr in %.1f ms (%dx%d -> %dx%d, %dx)\n",
                ms, w, h, ow, oh, scale);
        res.set_content(js.str(), "application/json");
    });

    // POST /pan/sr — PAN whole-image super-resolution (Pixel Attention Network)
    svr.Post("/pan/sr", [&](const httplib::Request & req, httplib::Response & res) {
        if (!pan_sr_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no PAN SR model loaded (use --pan-model)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }

        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 3);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(pan_sr_mutex);
        auto t0 = std::chrono::steady_clock::now();

        uint8_t * out = nullptr;
        int ow = 0, oh = 0;
        int rc = crispembed_pan_sr_process(
            pan_sr_ctx, data, w, h,
            /*tile_size=*/0, /*tile_overlap=*/0,
            &out, &ow, &oh);
        stbi_image_free(data);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0 || !out) {
            res.status = 500;
            res.set_content("{\"error\": \"PAN SR processing failed\"}", "application/json");
            return;
        }

        // Base64-encode the raw RGB output
        static const char b64chars[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const size_t n_bytes = (size_t)ow * oh * 3;
        std::string b64;
        b64.reserve(((n_bytes + 2) / 3) * 4);
        for (size_t i = 0; i < n_bytes; i += 3) {
            uint32_t v = (uint32_t)out[i] << 16;
            if (i + 1 < n_bytes) v |= (uint32_t)out[i + 1] << 8;
            if (i + 2 < n_bytes) v |= (uint32_t)out[i + 2];
            b64 += b64chars[(v >> 18) & 0x3f];
            b64 += b64chars[(v >> 12) & 0x3f];
            b64 += (i + 1 < n_bytes) ? b64chars[(v >> 6) & 0x3f] : '=';
            b64 += (i + 2 < n_bytes) ? b64chars[v & 0x3f] : '=';
        }
        crispembed_pan_sr_free_image(out);

        const int scale = crispembed_pan_sr_scale(pan_sr_ctx);

        std::ostringstream js;
        js << "{\"image\": \"" << b64 << "\""
           << ", \"width\": " << ow << ", \"height\": " << oh
           << ", \"original_width\": " << w << ", \"original_height\": " << h
           << ", \"upscale_factor\": " << scale
           << ", \"ms\": " << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /pan/sr in %.1f ms (%dx%d -> %dx%d, %dx)\n",
                ms, w, h, ow, oh, scale);
        res.set_content(js.str(), "application/json");
    });

    // POST /hat/sr — HAT whole-image super-resolution (Hybrid Attention Transformer)
    svr.Post("/hat/sr", [&](const httplib::Request & req, httplib::Response & res) {
        if (!hat_sr_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no HAT SR model loaded (use --hat-model)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }

        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 3);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(hat_sr_mutex);
        auto t0 = std::chrono::steady_clock::now();

        uint8_t * out = nullptr;
        int ow = 0, oh = 0;
        int rc = crispembed_hat_sr_process(
            hat_sr_ctx, data, w, h,
            /*tile_size=*/0, /*tile_overlap=*/0,
            &out, &ow, &oh);
        stbi_image_free(data);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0 || !out) {
            res.status = 500;
            res.set_content("{\"error\": \"HAT SR processing failed\"}", "application/json");
            return;
        }

        static const char b64chars[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const size_t n_bytes = (size_t)ow * oh * 3;
        std::string b64;
        b64.reserve(((n_bytes + 2) / 3) * 4);
        for (size_t i = 0; i < n_bytes; i += 3) {
            uint32_t v = (uint32_t)out[i] << 16;
            if (i + 1 < n_bytes) v |= (uint32_t)out[i + 1] << 8;
            if (i + 2 < n_bytes) v |= (uint32_t)out[i + 2];
            b64 += b64chars[(v >> 18) & 0x3f];
            b64 += b64chars[(v >> 12) & 0x3f];
            b64 += (i + 1 < n_bytes) ? b64chars[(v >> 6) & 0x3f] : '=';
            b64 += (i + 2 < n_bytes) ? b64chars[v & 0x3f] : '=';
        }
        crispembed_hat_sr_free_image(out);

        const int scale = crispembed_hat_sr_scale(hat_sr_ctx);

        std::ostringstream js;
        js << "{\"image\": \"" << b64 << "\""
           << ", \"width\": " << ow << ", \"height\": " << oh
           << ", \"original_width\": " << w << ", \"original_height\": " << h
           << ", \"upscale_factor\": " << scale
           << ", \"ms\": " << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /hat/sr in %.1f ms (%dx%d -> %dx%d, %dx)\n",
                ms, w, h, ow, oh, scale);
        res.set_content(js.str(), "application/json");
    });

    // POST /dat/sr — DAT whole-image super-resolution (Dual Aggregation Transformer)
    svr.Post("/dat/sr", [&](const httplib::Request & req, httplib::Response & res) {
        if (!dat_sr_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no DAT SR model loaded (use --dat-model)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }

        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 3);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(dat_sr_mutex);
        auto t0 = std::chrono::steady_clock::now();

        uint8_t * out = nullptr;
        int ow = 0, oh = 0;
        int rc = crispembed_dat_sr_process(
            dat_sr_ctx, data, w, h,
            /*tile_size=*/0, /*tile_overlap=*/0,
            &out, &ow, &oh);
        stbi_image_free(data);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0 || !out) {
            res.status = 500;
            res.set_content("{\"error\": \"DAT SR processing failed\"}", "application/json");
            return;
        }

        static const char b64chars[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const size_t n_bytes = (size_t)ow * oh * 3;
        std::string b64;
        b64.reserve(((n_bytes + 2) / 3) * 4);
        for (size_t i = 0; i < n_bytes; i += 3) {
            uint32_t v = (uint32_t)out[i] << 16;
            if (i + 1 < n_bytes) v |= (uint32_t)out[i + 1] << 8;
            if (i + 2 < n_bytes) v |= (uint32_t)out[i + 2];
            b64 += b64chars[(v >> 18) & 0x3f];
            b64 += b64chars[(v >> 12) & 0x3f];
            b64 += (i + 1 < n_bytes) ? b64chars[(v >> 6) & 0x3f] : '=';
            b64 += (i + 2 < n_bytes) ? b64chars[v & 0x3f] : '=';
        }
        crispembed_dat_sr_free_image(out);

        const int scale = (w > 0) ? ow / w : 2;

        std::ostringstream js;
        js << "{\"image\": \"" << b64 << "\""
           << ", \"width\": " << ow << ", \"height\": " << oh
           << ", \"original_width\": " << w << ", \"original_height\": " << h
           << ", \"upscale_factor\": " << scale
           << ", \"ms\": " << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /dat/sr in %.1f ms (%dx%d -> %dx%d, %dx)\n",
                ms, w, h, ow, oh, scale);
        res.set_content(js.str(), "application/json");
    });

    // POST /safmn/sr — SAFMN whole-image super-resolution
    svr.Post("/safmn/sr", [&](const httplib::Request & req, httplib::Response & res) {
        if (!safmn_sr_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no SAFMN SR model loaded (use --safmn-model)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }

        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 3);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(safmn_sr_mutex);
        auto t0 = std::chrono::steady_clock::now();

        uint8_t * out = nullptr;
        int ow = 0, oh = 0;
        int rc = crispembed_safmn_sr_process(
            safmn_sr_ctx, data, w, h,
            /*tile_size=*/0, /*tile_overlap=*/0,
            &out, &ow, &oh);
        stbi_image_free(data);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0 || !out) {
            res.status = 500;
            res.set_content("{\"error\": \"SAFMN SR processing failed\"}", "application/json");
            return;
        }

        static const char b64chars[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const size_t n_bytes = (size_t)ow * oh * 3;
        std::string b64;
        b64.reserve(((n_bytes + 2) / 3) * 4);
        for (size_t i = 0; i < n_bytes; i += 3) {
            uint32_t v = (uint32_t)out[i] << 16;
            if (i + 1 < n_bytes) v |= (uint32_t)out[i + 1] << 8;
            if (i + 2 < n_bytes) v |= (uint32_t)out[i + 2];
            b64 += b64chars[(v >> 18) & 0x3f];
            b64 += b64chars[(v >> 12) & 0x3f];
            b64 += (i + 1 < n_bytes) ? b64chars[(v >> 6) & 0x3f] : '=';
            b64 += (i + 2 < n_bytes) ? b64chars[v & 0x3f] : '=';
        }
        crispembed_safmn_sr_free_image(out);

        const int scale = crispembed_safmn_sr_scale(safmn_sr_ctx);

        std::ostringstream js;
        js << "{\"image\": \"" << b64 << "\""
           << ", \"width\": " << ow << ", \"height\": " << oh
           << ", \"original_width\": " << w << ", \"original_height\": " << h
           << ", \"upscale_factor\": " << scale
           << ", \"ms\": " << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /safmn/sr in %.1f ms (%dx%d -> %dx%d, %dx)\n",
                ms, w, h, ow, oh, scale);
        res.set_content(js.str(), "application/json");
    });

    // POST /esrgan/sr — Real-ESRGAN whole-image super-resolution
    svr.Post("/esrgan/sr", [&](const httplib::Request & req, httplib::Response & res) {
        if (!esrgan_sr_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no Real-ESRGAN SR model loaded (use --esrgan-model)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }

        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 3);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(esrgan_sr_mutex);
        auto t0 = std::chrono::steady_clock::now();

        uint8_t * out = nullptr;
        int ow = 0, oh = 0;
        int rc = crispembed_esrgan_sr_process(
            esrgan_sr_ctx, data, w, h,
            /*tile_size=*/0, /*tile_overlap=*/0,
            &out, &ow, &oh);
        stbi_image_free(data);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0 || !out) {
            res.status = 500;
            res.set_content("{\"error\": \"Real-ESRGAN SR processing failed\"}", "application/json");
            return;
        }

        static const char b64chars[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const size_t n_bytes = (size_t)ow * oh * 3;
        std::string b64;
        b64.reserve(((n_bytes + 2) / 3) * 4);
        for (size_t i = 0; i < n_bytes; i += 3) {
            uint32_t v = (uint32_t)out[i] << 16;
            if (i + 1 < n_bytes) v |= (uint32_t)out[i + 1] << 8;
            if (i + 2 < n_bytes) v |= (uint32_t)out[i + 2];
            b64 += b64chars[(v >> 18) & 0x3f];
            b64 += b64chars[(v >> 12) & 0x3f];
            b64 += (i + 1 < n_bytes) ? b64chars[(v >> 6) & 0x3f] : '=';
            b64 += (i + 2 < n_bytes) ? b64chars[v & 0x3f] : '=';
        }
        crispembed_esrgan_sr_free_image(out);

        const int scale = crispembed_esrgan_sr_scale(esrgan_sr_ctx);

        std::ostringstream js;
        js << "{\"image\": \"" << b64 << "\""
           << ", \"width\": " << ow << ", \"height\": " << oh
           << ", \"original_width\": " << w << ", \"original_height\": " << h
           << ", \"upscale_factor\": " << scale
           << ", \"ms\": " << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /esrgan/sr in %.1f ms (%dx%d -> %dx%d, %dx)\n",
                ms, w, h, ow, oh, scale);
        res.set_content(js.str(), "application/json");
    });

    // POST /swinir/sr — SwinIR-light whole-image super-resolution
    svr.Post("/swinir/sr", [&](const httplib::Request & req, httplib::Response & res) {
        if (!swinir_sr_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no SwinIR SR model loaded (use --swinir-model)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }

        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 3);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(swinir_sr_mutex);
        auto t0 = std::chrono::steady_clock::now();

        uint8_t * out = nullptr;
        int ow = 0, oh = 0;
        int rc = crispembed_swinir_sr_process(
            swinir_sr_ctx, data, w, h,
            /*tile_size=*/0, /*tile_overlap=*/0,
            &out, &ow, &oh);
        stbi_image_free(data);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0 || !out) {
            res.status = 500;
            res.set_content("{\"error\": \"SwinIR SR processing failed\"}", "application/json");
            return;
        }

        // Base64-encode the raw RGB output
        static const char b64chars[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const size_t n_bytes = (size_t)ow * oh * 3;
        std::string b64;
        b64.reserve(((n_bytes + 2) / 3) * 4);
        for (size_t i = 0; i < n_bytes; i += 3) {
            uint32_t v = (uint32_t)out[i] << 16;
            if (i + 1 < n_bytes) v |= (uint32_t)out[i + 1] << 8;
            if (i + 2 < n_bytes) v |= (uint32_t)out[i + 2];
            b64 += b64chars[(v >> 18) & 0x3f];
            b64 += b64chars[(v >> 12) & 0x3f];
            b64 += (i + 1 < n_bytes) ? b64chars[(v >> 6) & 0x3f] : '=';
            b64 += (i + 2 < n_bytes) ? b64chars[v & 0x3f] : '=';
        }
        crispembed_swinir_sr_free_image(out);

        const int scale = crispembed_swinir_sr_scale(swinir_sr_ctx);

        std::ostringstream js;
        js << "{\"image\": \"" << b64 << "\""
           << ", \"width\": " << ow << ", \"height\": " << oh
           << ", \"original_width\": " << w << ", \"original_height\": " << h
           << ", \"upscale_factor\": " << scale
           << ", \"ms\": " << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /swinir/sr in %.1f ms (%dx%d -> %dx%d, %dx)\n",
                ms, w, h, ow, oh, scale);
        res.set_content(js.str(), "application/json");
    });

    // POST /tbsrn/sr — TBSRN text-line super-resolution (Telescope)
    svr.Post("/tbsrn/sr", [&](const httplib::Request & req, httplib::Response & res) {
        if (!tbsrn_sr_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no TBSRN SR model loaded (use --tbsrn-model)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }

        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 3);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(tbsrn_sr_mutex);
        auto t0 = std::chrono::steady_clock::now();

        uint8_t * out = nullptr;
        int ow = 0, oh = 0;
        int rc = crispembed_tbsrn_sr_process(
            tbsrn_sr_ctx, data, w, h,
            &out, &ow, &oh);
        stbi_image_free(data);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0 || !out) {
            res.status = 500;
            res.set_content("{\"error\": \"TBSRN SR processing failed\"}", "application/json");
            return;
        }

        // Base64-encode the raw RGB output
        static const char b64chars[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const size_t n_bytes = (size_t)ow * oh * 3;
        std::string b64;
        b64.reserve(((n_bytes + 2) / 3) * 4);
        for (size_t i = 0; i < n_bytes; i += 3) {
            uint32_t v = (uint32_t)out[i] << 16;
            if (i + 1 < n_bytes) v |= (uint32_t)out[i + 1] << 8;
            if (i + 2 < n_bytes) v |= (uint32_t)out[i + 2];
            b64 += b64chars[(v >> 18) & 0x3f];
            b64 += b64chars[(v >> 12) & 0x3f];
            b64 += (i + 1 < n_bytes) ? b64chars[(v >> 6) & 0x3f] : '=';
            b64 += (i + 2 < n_bytes) ? b64chars[v & 0x3f] : '=';
        }
        crispembed_tbsrn_sr_free_image(out);

        std::ostringstream js;
        js << "{\"image\": \"" << b64 << "\""
           << ", \"width\": " << ow << ", \"height\": " << oh
           << ", \"original_width\": " << w << ", \"original_height\": " << h
           << ", \"upscale_factor\": 4"
           << ", \"ms\": " << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /tbsrn/sr in %.1f ms (%dx%d -> %dx%d, 4x)\n",
                ms, w, h, ow, oh);
        res.set_content(js.str(), "application/json");
    });

    // POST /restormer — Restormer image restoration (denoising, deblurring, deraining)
    svr.Post("/restormer", [&](const httplib::Request & req, httplib::Response & res) {
        if (!restormer_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no Restormer model loaded (use --restormer-model)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }

        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 3);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(restormer_mutex);
        auto t0 = std::chrono::steady_clock::now();

        uint8_t * out = nullptr;
        int rc = crispembed_restormer_process(
            restormer_ctx, data, w, h,
            /*tile_size=*/0, /*tile_overlap=*/0,
            &out);
        stbi_image_free(data);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0 || !out) {
            res.status = 500;
            res.set_content("{\"error\": \"Restormer processing failed\"}", "application/json");
            return;
        }

        // Base64-encode the raw RGB output
        static const char b64chars[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const size_t n_bytes = (size_t)w * h * 3;
        std::string b64;
        b64.reserve(((n_bytes + 2) / 3) * 4);
        for (size_t i = 0; i < n_bytes; i += 3) {
            uint32_t v = (uint32_t)out[i] << 16;
            if (i + 1 < n_bytes) v |= (uint32_t)out[i + 1] << 8;
            if (i + 2 < n_bytes) v |= (uint32_t)out[i + 2];
            b64 += b64chars[(v >> 18) & 0x3f];
            b64 += b64chars[(v >> 12) & 0x3f];
            b64 += (i + 1 < n_bytes) ? b64chars[(v >> 6) & 0x3f] : '=';
            b64 += (i + 2 < n_bytes) ? b64chars[v & 0x3f] : '=';
        }
        crispembed_restormer_free_image(out);

        std::ostringstream js;
        js << "{\"image\": \"" << b64 << "\""
           << ", \"width\": " << w << ", \"height\": " << h
           << ", \"ms\": " << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /restormer in %.1f ms (%dx%d)\n",
                ms, w, h);
        res.set_content(js.str(), "application/json");
    });

    // POST /scunet/denoise — SCUNet image denoising (Swin-Conv-UNet)
    svr.Post("/scunet/denoise", [&](const httplib::Request & req, httplib::Response & res) {
        if (!scunet_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no SCUNet model loaded (use --scunet-model)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }

        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 3);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(scunet_mutex);
        auto t0 = std::chrono::steady_clock::now();

        uint8_t * out = nullptr;
        int rc = crispembed_scunet_process(scunet_ctx, data, w, h, &out);
        stbi_image_free(data);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0 || !out) {
            res.status = 500;
            res.set_content("{\"error\": \"SCUNet denoising failed\"}", "application/json");
            return;
        }

        // Base64-encode the raw RGB output
        static const char b64chars[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const size_t n_bytes = (size_t)w * h * 3;
        std::string b64;
        b64.reserve(((n_bytes + 2) / 3) * 4);
        for (size_t i = 0; i < n_bytes; i += 3) {
            uint32_t v = (uint32_t)out[i] << 16;
            if (i + 1 < n_bytes) v |= (uint32_t)out[i + 1] << 8;
            if (i + 2 < n_bytes) v |= (uint32_t)out[i + 2];
            b64 += b64chars[(v >> 18) & 0x3f];
            b64 += b64chars[(v >> 12) & 0x3f];
            b64 += (i + 1 < n_bytes) ? b64chars[(v >> 6) & 0x3f] : '=';
            b64 += (i + 2 < n_bytes) ? b64chars[v & 0x3f] : '=';
        }
        crispembed_scunet_free_image(out);

        std::ostringstream js;
        js << "{\"image\": \"" << b64 << "\""
           << ", \"width\": " << w << ", \"height\": " << h
           << ", \"ms\": " << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /scunet/denoise in %.1f ms (%dx%d)\n",
                ms, w, h);
        res.set_content(js.str(), "application/json");
    });

    // POST /instructir/restore — InstructIR all-in-one image restoration
    svr.Post("/instructir/restore", [&](const httplib::Request & req, httplib::Response & res) {
        if (!instructir_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no InstructIR model loaded (use --instructir-model)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }

        // Parse task (default 0 = denoise)
        int task = 0;
        auto tpos = body.find("\"task\"");
        if (tpos != std::string::npos) {
            auto colon = body.find(':', tpos + 5);
            if (colon != std::string::npos)
                task = atoi(body.c_str() + colon + 1);
        }
        if (task < 0 || task > 6) {
            res.status = 400;
            res.set_content("{\"error\": \"task must be 0-6\"}", "application/json");
            return;
        }

        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 3);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(instructir_mutex);
        auto t0 = std::chrono::steady_clock::now();

        uint8_t * out = nullptr;
        int rc = crispembed_instructir_process(instructir_ctx, task, data, w, h, &out);
        stbi_image_free(data);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0 || !out) {
            res.status = 500;
            res.set_content("{\"error\": \"InstructIR processing failed\"}", "application/json");
            return;
        }

        // Base64-encode the raw RGB output
        static const char b64chars[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const size_t n_bytes = (size_t)w * h * 3;
        std::string b64;
        b64.reserve(((n_bytes + 2) / 3) * 4);
        for (size_t i = 0; i < n_bytes; i += 3) {
            uint32_t v = (uint32_t)out[i] << 16;
            if (i + 1 < n_bytes) v |= (uint32_t)out[i + 1] << 8;
            if (i + 2 < n_bytes) v |= (uint32_t)out[i + 2];
            b64 += b64chars[(v >> 18) & 0x3f];
            b64 += b64chars[(v >> 12) & 0x3f];
            b64 += (i + 1 < n_bytes) ? b64chars[(v >> 6) & 0x3f] : '=';
            b64 += (i + 2 < n_bytes) ? b64chars[v & 0x3f] : '=';
        }
        crispembed_instructir_free_image(out);

        std::ostringstream js;
        js << "{\"image\": \"" << b64 << "\""
           << ", \"width\": " << w << ", \"height\": " << h
           << ", \"task\": " << task
           << ", \"ms\": " << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /instructir/restore task=%d in %.1f ms (%dx%d)\n",
                task, ms, w, h);
        res.set_content(js.str(), "application/json");
    });

    // POST /adair/restore — AdaIR all-in-one image restoration (Restormer+AFLB+FFT)
    svr.Post("/adair/restore", [&](const httplib::Request & req, httplib::Response & res) {
        if (!adair_ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"no AdaIR model loaded (use --adair-model)\"}", "application/json");
            return;
        }

        auto body = req.body;
        std::string image_path;
        auto pos = body.find("\"image\"");
        if (pos != std::string::npos) {
            auto q1 = body.find('"', pos + 7);
            auto q2 = body.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                image_path = body.substr(q1 + 1, q2 - q1 - 1);
        }
        if (image_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"missing 'image' field\"}", "application/json");
            return;
        }

        int w, h, ch;
        unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &ch, 3);
        if (!data) {
            res.status = 400;
            res.set_content("{\"error\": \"cannot load image\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(adair_mutex);
        auto t0 = std::chrono::steady_clock::now();

        uint8_t * out = nullptr;
        int rc = crispembed_adair_process(adair_ctx, data, w, h, &out);
        stbi_image_free(data);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rc != 0 || !out) {
            res.status = 500;
            res.set_content("{\"error\": \"AdaIR restoration failed\"}", "application/json");
            return;
        }

        // Base64-encode the raw RGB output
        static const char b64chars[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const size_t n_bytes = (size_t)w * h * 3;
        std::string b64;
        b64.reserve(((n_bytes + 2) / 3) * 4);
        for (size_t i = 0; i < n_bytes; i += 3) {
            uint32_t v = (uint32_t)out[i] << 16;
            if (i + 1 < n_bytes) v |= (uint32_t)out[i + 1] << 8;
            if (i + 2 < n_bytes) v |= (uint32_t)out[i + 2];
            b64 += b64chars[(v >> 18) & 0x3f];
            b64 += b64chars[(v >> 12) & 0x3f];
            b64 += (i + 1 < n_bytes) ? b64chars[(v >> 6) & 0x3f] : '=';
            b64 += (i + 2 < n_bytes) ? b64chars[v & 0x3f] : '=';
        }
        crispembed_adair_free_image(out);

        std::ostringstream js;
        js << "{\"image\": \"" << b64 << "\""
           << ", \"width\": " << w << ", \"height\": " << h
           << ", \"ms\": " << std::fixed << std::setprecision(1) << ms << "}";

        fprintf(stderr, "crispembed-server: /adair/restore in %.1f ms (%dx%d)\n",
                ms, w, h);
        res.set_content(js.str(), "application/json");
    });

    // POST /ocr/document — end-to-end multi-page OCR
    // Accepts:
    //   multipart/form-data: files named "page" (or "page0","page1",...) → image bytes
    //   OR application/json: {"images": ["/path/page1.png", "/path/page2.png"],
    //                         "format": "text|hocr|alto|pdf",
    //                         "cleanup": true, "dewarp": false}
    // Returns: rendered document in the requested format
    svr.Post("/ocr/document", [&](const httplib::Request & req, httplib::Response & res) {
        std::string format = "text";
        bool do_cleanup = true;
        bool do_dewarp = false;
        std::vector<std::string> temp_files; // track files to clean up

        // Collect page images (either from multipart or JSON paths)
        std::vector<std::string> page_paths;

        if (req.is_multipart_form_data()) {
            // Multipart upload: save each file to temp, collect paths
            auto files = req.files;
            int page_idx = 0;
            for (auto & [name, file] : files) {
                if (name == "format") { format = file.content; continue; }
                if (name == "cleanup") { do_cleanup = file.content != "false" && file.content != "0"; continue; }
                if (name == "dewarp") { do_dewarp = file.content == "true" || file.content == "1"; continue; }
                // Must be a page image
                char tmp_path[256];
                snprintf(tmp_path, sizeof(tmp_path), "/tmp/crispembed_doc_%d_%d.img",
                         (int)getpid(), page_idx++);
                FILE * f = fopen(tmp_path, "wb");
                if (f) {
                    fwrite(file.content.data(), 1, file.content.size(), f);
                    fclose(f);
                    page_paths.push_back(tmp_path);
                    temp_files.push_back(tmp_path);
                }
            }
        } else {
            // JSON mode: parse image paths
            auto body = req.body;
            auto fpos = body.find("\"format\"");
            if (fpos != std::string::npos) {
                auto q1 = body.find('"', fpos + 8);
                auto q2 = body.find('"', q1 + 1);
                if (q1 != std::string::npos && q2 != std::string::npos)
                    format = body.substr(q1 + 1, q2 - q1 - 1);
            }
            // Parse "images" array
            auto apos = body.find("\"images\"");
            if (apos != std::string::npos) {
                auto arr_start = body.find('[', apos);
                auto arr_end = body.find(']', arr_start);
                if (arr_start != std::string::npos && arr_end != std::string::npos) {
                    size_t p = arr_start;
                    while ((p = body.find('"', p + 1)) < arr_end) {
                        auto q2 = body.find('"', p + 1);
                        if (q2 < arr_end) {
                            page_paths.push_back(body.substr(p + 1, q2 - p - 1));
                            p = q2;
                        } else break;
                    }
                }
            }
            // Single image shortcut
            if (page_paths.empty()) {
                auto ipos = body.find("\"image\"");
                if (ipos != std::string::npos) {
                    auto q1 = body.find('"', ipos + 7);
                    auto q2 = body.find('"', q1 + 1);
                    if (q1 != std::string::npos && q2 != std::string::npos)
                        page_paths.push_back(body.substr(q1 + 1, q2 - q1 - 1));
                }
            }
        }

        if (page_paths.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"no page images provided\"}", "application/json");
            return;
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        // Process each page
        ocr_renderer * renderer = ocr_render_create(
            format == "hocr" ? OCR_RENDER_HOCR :
            format == "alto" ? OCR_RENDER_ALTO :
            format == "pdf"  ? OCR_RENDER_PDF : OCR_RENDER_TEXT);
        if (format == "pdf") ocr_render_set_pdfa(renderer, 1);
        ocr_render_begin(renderer);

        int total_regions = 0;
        for (size_t pi = 0; pi < page_paths.size(); pi++) {
            int w = 0, h = 0, ch = 0;
            unsigned char * img = stbi_load(page_paths[pi].c_str(), &w, &h, &ch, 0);
            if (!img) continue;

            // Optional dewarp (grayscale)
            std::vector<uint8_t> gray_buf;
            if (do_dewarp && ch >= 1) {
                // Convert to gray if needed
                std::vector<uint8_t> gray(w * h);
                if (ch == 1) {
                    memcpy(gray.data(), img, w * h);
                } else {
                    for (int i = 0; i < w * h; i++)
                        gray[i] = (uint8_t)((img[i*ch]*77 + img[i*ch+1]*150 + img[i*ch+2]*29) >> 8);
                }
                std::vector<uint8_t> dewarped(w * h);
                int dw = 0, dh = 0;
                if (crispembed_dewarp(gray.data(), w, h, dewarped.data(), &dw, &dh) == 0) {
                    gray_buf = std::move(dewarped);
                    // Can't easily feed gray back to color OCR, so skip dewarp for color pipelines
                }
            }

            // OCR: use orchestrator if available, else single-shot OCR model
            int n_results = 0;
            const char * full_text = nullptr;
            float mean_conf = 0;
            const crispembed_ocr_result * results = nullptr;

            if (ocr_orch_ctx) {
                std::lock_guard<std::mutex> lock(ocr_orch_mutex);
                results = crispembed_ocr_pipeline_run(
                    ocr_orch_ctx, page_paths[pi].c_str(),
                    &n_results, &full_text, &mean_conf);
            } else if (ocr_model_ctx) {
                std::lock_guard<std::mutex> lock(ocr_model_mutex);
                int out_len = 0;
                const char * text = crispembed_ocr_model_recognize(
                    ocr_model_ctx, img, w, h, ch, &out_len);
                // Wrap single result
                static crispembed_ocr_result single_result;
                if (text && out_len > 0) {
                    single_result = {0, 0, (float)w, (float)h, 1.0f, text, out_len};
                    results = &single_result;
                    n_results = 1;
                    full_text = text;
                }
            }

            stbi_image_free(img);

            // Build render page from results
            if (n_results > 0 && results) {
                total_regions += n_results;
                std::vector<ocr_render_word> words(n_results);
                std::vector<ocr_render_line> lines(n_results);
                for (int i = 0; i < n_results; i++) {
                    words[i] = {results[i].text,
                                (int)results[i].x, (int)results[i].y,
                                (int)results[i].w, (int)results[i].h,
                                results[i].confidence};
                    lines[i] = {&words[i], 1,
                                (int)results[i].x, (int)results[i].y,
                                (int)results[i].w, (int)results[i].h};
                }
                ocr_render_page page = {lines.data(), n_results,
                                        w, h, page_paths[pi].c_str()};
                ocr_render_add_page(renderer, &page);
            } else {
                // Empty page
                ocr_render_page page = {nullptr, 0, w, h, page_paths[pi].c_str()};
                ocr_render_add_page(renderer, &page);
            }
        }

        ocr_render_end(renderer);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Clean up temp files
        for (auto & tf : temp_files) std::remove(tf.c_str());

        // Return rendered document
        int out_size = ocr_render_output_size(renderer);
        const char * out_data = ocr_render_output(renderer);

        const char * content_type = "text/plain";
        if (format == "hocr") content_type = "text/html; charset=utf-8";
        else if (format == "alto") content_type = "application/xml; charset=utf-8";
        else if (format == "pdf") content_type = "application/pdf";

        res.set_content(std::string(out_data, out_size), content_type);
        ocr_render_free(renderer);

        fprintf(stderr, "crispembed-server: /ocr/document %zu pages, %d regions, format=%s, %.1f ms\n",
                page_paths.size(), total_regions, format.c_str(), ms);
    });

    // GET /health
    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        std::ostringstream js;
        js << "{\"status\": \"ok\"";
        if (ctx) {
            js << ", \"dim\": " << dim
               << ", \"layers\": " << hp->n_layer
               << ", \"vocab\": " << hp->n_vocab;
        }
        if (face_det) js << ", \"face_detection\": true";
        if (face_rec) js << ", \"face_recognition\": true, \"face_dim\": " << crispembed_face_dim(face_rec);
        if (vit_ctx) js << ", \"vit\": true, \"vit_dim\": " << crispembed_vit_dim(vit_ctx);
        if (clip_text_ctx) js << ", \"clip_text\": true, \"clip_text_dim\": " << crispembed_clip_text_dim(clip_text_ctx);
        if (ocr_model_ctx) js << ", \"ocr_model\": true, \"math_ocr\": true";
        if (ocr_pipeline_ctx) js << ", \"ocr_pipeline\": true";
        if (layout_ctx) js << ", \"layout\": true";
        if (text_det_ctx) js << ", \"text_detection\": true";
        if (ner_ctx) js << ", \"ner\": true";
        if (pix2struct_ctx) js << ", \"pix2struct\": true";
        if (ocr_orch_ctx) js << ", \"ocr_orchestrator\": true";
        if (text_sr_ctx) js << ", \"text_sr\": true, \"text_sr_upscale\": " << crispembed_text_sr_upscale_factor(text_sr_ctx);
        if (pan_sr_ctx) js << ", \"pan_sr\": true, \"pan_sr_upscale\": " << crispembed_pan_sr_scale(pan_sr_ctx);
        if (hat_sr_ctx) js << ", \"hat_sr\": true, \"hat_sr_upscale\": " << crispembed_hat_sr_scale(hat_sr_ctx);
        if (dat_sr_ctx) js << ", \"dat_sr\": true";
        if (safmn_sr_ctx) js << ", \"safmn_sr\": true, \"safmn_sr_upscale\": " << crispembed_safmn_sr_scale(safmn_sr_ctx);
        if (esrgan_sr_ctx) js << ", \"esrgan_sr\": true, \"esrgan_sr_upscale\": " << crispembed_esrgan_sr_scale(esrgan_sr_ctx);
        if (swinir_sr_ctx) js << ", \"swinir_sr\": true, \"swinir_sr_upscale\": " << crispembed_swinir_sr_scale(swinir_sr_ctx);
        if (tbsrn_sr_ctx) js << ", \"tbsrn_sr\": true, \"tbsrn_sr_upscale\": 4";
        if (restormer_ctx) js << ", \"restormer\": true";
        if (scunet_ctx) js << ", \"scunet\": true";
        if (instructir_ctx) js << ", \"instructir\": true";
        if (adair_ctx) js << ", \"adair\": true";
        js << ", \"scan_cleanup\": true";  // always available (no model needed)
        js << "}";
        res.set_content(js.str(), "application/json");
    });

    fprintf(stderr, "\ncrispembed-server: listening on %s:%d\n", host.c_str(), port);
    if (ctx) {
        fprintf(stderr, "  POST /embed           — {\"texts\": [\"hello\"]}\n");
        fprintf(stderr, "  POST /v1/embeddings   — OpenAI-compatible\n");
        fprintf(stderr, "  POST /api/embed       — Ollama-compatible\n");
        fprintf(stderr, "  POST /api/embeddings  — Ollama-compatible (legacy)\n");
    }
    if (face_det) fprintf(stderr, "  POST /detect          — {\"image\": \"path.jpg\"}\n");
    if (face_det && face_rec) fprintf(stderr, "  POST /face            — {\"image\": \"path.jpg\"} (pipeline)\n");
    if (vit_ctx) fprintf(stderr, "  POST /vit/encode      — {\"image\": \"path.jpg\"}\n");
    if (clip_text_ctx) fprintf(stderr, "  POST /clip/text       — {\"text\": \"query\"}\n");
    if (pix2struct_ctx) fprintf(stderr, "  POST /pix2struct/generate — {\"image\": \"doc.png\", \"max_tokens\": 256}\n");
    if (ocr_model_ctx) fprintf(stderr, "  POST /ocr/model       — {\"image\": \"formula.png\"}  (alias: /math/ocr)\n");
    if (ocr_pipeline_ctx) fprintf(stderr, "  POST /ocr             — {\"image\": \"document.png\"} (detect+recognize)\n");
    if (layout_ctx) fprintf(stderr, "  POST /layout/detect   — {\"image\": \"page.png\"}\n");
    fprintf(stderr, "  POST /table/parse     — {\"image\": \"table.png\"} → {\"html\": \"<table>...\"}\n");
    if (text_det_ctx) fprintf(stderr, "  POST /text/detect     — {\"image\": \"page.png\"}\n");
    if (ner_ctx) fprintf(stderr, "  POST /ner/extract     — {\"text\": \"...\", \"labels\": [\"person\", ...]}\n");
#if SERVER_HAS_LID
    if (lid_ctx) fprintf(stderr, "  POST /lid/detect      — {\"text\": \"...\"} → {\"lang\": \"de\", \"confidence\": 0.99}\n");
#endif
    if (kie_ctx) fprintf(stderr, "  POST /kie/extract     — {\"image\": \"doc.png\", \"labels\": [\"total\", ...]} (OCR+NER)\n");
    if (ocr_orch_ctx) fprintf(stderr, "  POST /ocr/pipeline    — {\"image\": \"doc.png\"} (routing + cleanup + accept-gate)\n");
    if (text_sr_ctx) fprintf(stderr, "  POST /text/sr         — {\"image\": \"low_dpi.png\"} (upscale %dx)\n", crispembed_text_sr_upscale_factor(text_sr_ctx));
    if (pan_sr_ctx) fprintf(stderr, "  POST /pan/sr          — {\"image\": \"photo.png\"} (upscale %dx)\n", crispembed_pan_sr_scale(pan_sr_ctx));
    if (hat_sr_ctx) fprintf(stderr, "  POST /hat/sr          — {\"image\": \"photo.png\"} (upscale %dx)\n", crispembed_hat_sr_scale(hat_sr_ctx));
    if (dat_sr_ctx) fprintf(stderr, "  POST /dat/sr          — {\"image\": \"photo.png\"} (upscale 2x)\n");
    if (safmn_sr_ctx) fprintf(stderr, "  POST /safmn/sr        — {\"image\": \"photo.png\"} (upscale %dx)\n", crispembed_safmn_sr_scale(safmn_sr_ctx));
    if (esrgan_sr_ctx) fprintf(stderr, "  POST /esrgan/sr       — {\"image\": \"photo.png\"} (upscale %dx)\n", crispembed_esrgan_sr_scale(esrgan_sr_ctx));
    if (swinir_sr_ctx) fprintf(stderr, "  POST /swinir/sr       — {\"image\": \"photo.png\"} (upscale %dx)\n", crispembed_swinir_sr_scale(swinir_sr_ctx));
    if (tbsrn_sr_ctx) fprintf(stderr, "  POST /tbsrn/sr        — {\"image\": \"text_line.png\"} (upscale 4x)\n");
    if (restormer_ctx) fprintf(stderr, "  POST /restormer       — {\"image\": \"noisy.png\"} (denoise/restore)\n");
    if (scunet_ctx) fprintf(stderr, "  POST /scunet/denoise  — {\"image\": \"noisy.png\"} (Swin-Conv-UNet denoise)\n");
    if (instructir_ctx) fprintf(stderr, "  POST /instructir/restore — {\"image\": \"...\", \"task\": 0} (all-in-one restoration)\n");
    if (adair_ctx) fprintf(stderr, "  POST /adair/restore     — {\"image\": \"noisy.png\"} (Restormer+AFLB+FFT restore)\n");
    fprintf(stderr, "  POST /scan/cleanup    — {\"image\": \"scan.png\"} (deskew, crop, whiten)\n");
    fprintf(stderr, "  POST /pdf/dpi              — {\"file\": \"...\"} (PDF DPI profiling)\n");
    fprintf(stderr, "  POST /preprocess/skew      — {\"image\": \"...\"} (find skew angle)\n");
    fprintf(stderr, "  POST /preprocess/dewarp    — {\"image\": \"...\"} (straighten curved text)\n");
    fprintf(stderr, "  POST /preprocess/tps-dewarp — {\"image\": \"...\", \"model\": \"tps-loc.gguf\"}\n");
    fprintf(stderr, "  POST /preprocess/cc-detect — {\"image\": \"...\"} (model-free line detection)\n");
    fprintf(stderr, "  POST /render/ocr           — {\"results\": [...], \"format\": \"hocr|alto|pdf\"}\n");
    fprintf(stderr, "  POST /ocr/document         — multi-page OCR → searchable PDF/hOCR/text (upload or paths)\n");
    if (ctx && crispembed_has_colbert(ctx)) fprintf(stderr, "  POST /colbert/score   — {\"query\": \"...\", \"documents\": [...]}\n");
    fprintf(stderr, "  GET  /health\n\n");

    svr.listen(host, port);

    if (kie_ctx) crispembed_kie_free(kie_ctx);
    if (text_sr_ctx) crispembed_text_sr_free(text_sr_ctx);
    if (pan_sr_ctx) crispembed_pan_sr_free(pan_sr_ctx);
    if (hat_sr_ctx) crispembed_hat_sr_free(hat_sr_ctx);
    if (dat_sr_ctx) crispembed_dat_sr_free(dat_sr_ctx);
    if (safmn_sr_ctx) crispembed_safmn_sr_free(safmn_sr_ctx);
    if (esrgan_sr_ctx) crispembed_esrgan_sr_free(esrgan_sr_ctx);
    if (swinir_sr_ctx) crispembed_swinir_sr_free(swinir_sr_ctx);
    if (tbsrn_sr_ctx) crispembed_tbsrn_sr_free(tbsrn_sr_ctx);
    if (restormer_ctx) crispembed_restormer_free(restormer_ctx);
    if (scunet_ctx) crispembed_scunet_free(scunet_ctx);
    if (adair_ctx) crispembed_adair_free(adair_ctx);
    if (instructir_ctx) crispembed_instructir_free(instructir_ctx);
    if (ner_ctx) crispembed_ner_free(ner_ctx);
#if SERVER_HAS_LID
    if (lid_ctx) text_lid_free(lid_ctx);
#endif
    if (layout_ctx) crispembed_layout_free(layout_ctx);
    if (text_det_ctx) crispembed_text_det_free(text_det_ctx);
    if (ocr_orch_ctx) crispembed_ocr_pipeline_free(ocr_orch_ctx);
    if (ocr_pipeline_ctx) crispembed_ocr_free(ocr_pipeline_ctx);
    if (ocr_model_ctx) crispembed_ocr_model_free(ocr_model_ctx);
    if (clip_text_ctx) crispembed_clip_text_free(clip_text_ctx);
    if (vit_ctx) crispembed_vit_free(vit_ctx);
    if (pix2struct_ctx) crispembed_pix2struct_free(pix2struct_ctx);
    if (face_det) crispembed_face_free(face_det);
    if (face_rec) crispembed_face_free(face_rec);
    if (ctx) crispembed_free(ctx);
    return 0;
}
