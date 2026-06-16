// crispembed CLI — encode text to embedding vector.
//
// Usage:
//   crispembed -m model.gguf "query: hello world"
//   crispembed -m octen-0.6b "hello world"        # auto-download
//   crispembed -m model.gguf -f texts.txt          (one text per line)
//   crispembed --list-models

#include "crispembed.h"
#include "model_mgr.h"
#include "vit_embed.h"
#include "clip_text_embed.h"
#include "cnn_embed.h"
#include "hmer_ocr.h"
#include "bttr_ocr.h"
#include "scan_cleanup.h"
#include "pdf_info.h"

// stb_image for --detect image loading
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "../../ggml/examples/stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    return out;
}

static float dot_product(const float * a, const float * b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s -m MODEL [options] [TEXT ...]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m MODEL         path to GGUF model or model name (auto-download)\n");
    fprintf(stderr, "  -f FILE          read texts from file (one per line)\n");
    fprintf(stderr, "  -t N             number of threads (default: 4)\n");
    fprintf(stderr, "  -d N             output dimension (Matryoshka truncation)\n");
    fprintf(stderr, "  --prefix TEXT    prepend a prefix to all inputs before tokenization\n");
    fprintf(stderr, "  --json           output as JSON array\n");
    fprintf(stderr, "  --dim            print embedding dimension and exit\n");
    fprintf(stderr, "  --capabilities   print model capability flags and exit\n");
    fprintf(stderr, "  --sparse         encode sparse term-weight vectors\n");
    fprintf(stderr, "  --colbert        encode ColBERT per-token vectors\n");
    fprintf(stderr, "  --audio FILE     encode raw 16 kHz mono float32 PCM (.raw); cross-modal embedding\n");
    fprintf(stderr, "  --image-raw FILE encode preprocessed image patches as float32 rows\n");
    fprintf(stderr, "  --grid-thw T,H,W image patch grid for --image-raw\n");
    fprintf(stderr, "  --image FILE     encode JPG/PNG/BMP via in-process preprocessor (cross-modal embedding)\n");
    fprintf(stderr, "  --rerank QUERY   cross-encoder rerank documents against QUERY\n");
    fprintf(stderr, "  --biencoder QUERY  bi-encoder rerank documents against QUERY\n");
    fprintf(stderr, "  --top-n N        limit rerank output to top N documents\n");
    fprintf(stderr, "  --face FILE      encode face from image (recognition model)\n");
    fprintf(stderr, "  --detect FILE    detect faces in image (detection model)\n");
    fprintf(stderr, "  --ocr FILE       OCR → text (auto-detect: pix2tex/hmer/bttr/posformer/ppformulanet/ppformulanet-l/texo/mixtex/parseq/qwen2vl/internvl2/glm-ocr/tesseract-lstm)\n");
    fprintf(stderr, "  --hmer FILE      handwritten math OCR → LaTeX (HMER model)\n");
    fprintf(stderr, "  --bttr FILE      handwritten math OCR → LaTeX (BTTR model)\n");
    fprintf(stderr, "  --layout FILE    document layout detection (RT-DETRv2, needs -m layout_model.gguf)\n");
    fprintf(stderr, "  --ner TEXT       named entity recognition (GLiNER, needs -m ner_model.gguf)\n");
    fprintf(stderr, "  --ner-labels L   comma-separated entity types (default: person,organization,location)\n");
    fprintf(stderr, "  --ner-threshold F  confidence threshold for NER (default: 0.5)\n");
    fprintf(stderr, "  --kie FILE       key information extraction: OCR + NER on a document image\n");
    fprintf(stderr, "                   needs -m ner_model.gguf --ocr-det det.gguf --ocr-rec rec.gguf\n");
    fprintf(stderr, "  --kie-labels L   comma-separated field names (default: uses --ner-labels)\n");
    fprintf(stderr, "  --kie-threshold F  confidence threshold for KIE (default: 0.5)\n");
    fprintf(stderr, "  --lilt FILE      LiLT token classification from JSON input\n");
    fprintf(stderr, "                   JSON: {\"input_ids\": [...], \"bbox\": [[x0,y0,x1,y1], ...]}\n");
    fprintf(stderr, "  --det MODEL      detection model for --face-pipeline\n");
    fprintf(stderr, "  --face-pipeline  detect+align+encode faces (needs -m rec_model --det det_model)\n");
    fprintf(stderr, "  --punct-model M  post-process OCR text with punctuation model (FireRedPunc/PCS)\n");
    fprintf(stderr, "  --output-format F  OCR output format: text (default), hocr, alto\n");
    fprintf(stderr, "  --pdf-dpi FILE     analyse PDF DPI (per-page image resolution profiling)\n");
    fprintf(stderr, "  --find-skew FILE   detect skew angle (degrees) of a document image\n");
    fprintf(stderr, "  --dewarp FILE      dewarp a curved document page (straighten text lines)\n");
    fprintf(stderr, "  --tps-dewarp MODEL FILE  TPS-based dewarp (learned, needs model GGUF)\n");
    fprintf(stderr, "  --cc-detect FILE   detect text lines via connected components (model-free)\n");
    fprintf(stderr, "  --cleanup        preprocess scan before OCR (deskew, crop borders, whiten background)\n");
    fprintf(stderr, "  --cleanup-only F process scan and write cleaned image to stdout (no OCR)\n");
    fprintf(stderr, "  --ocr-pipeline F full OCR pipeline: source-type routing + cleanup + accept-gate\n");
    fprintf(stderr, "       --ocr-engine N  primary engine (dbnet_trocr|surya|tesseract|got|glm|qwen2vl|internvl2)\n");
    fprintf(stderr, "       --denoise       NAFNet pre-processor; --punct-model M  post-OCR punctuation/spacing\n");
    fprintf(stderr, "       --sr-model PATH text super-resolution GGUF for low-DPI upscaling (NAFNet+PixelShuffle)\n");
    fprintf(stderr, "  --pan-sr FILE    standalone PAN super-resolution: upscale image, write PGM to stdout\n");
    fprintf(stderr, "                   (needs --pan-model PATH: PAN GGUF, Pixel Attention Network, 2x or 4x)\n");
    fprintf(stderr, "  --pan-model PATH PAN super-resolution GGUF (used with --pan-sr)\n");
    fprintf(stderr, "  --hat-sr FILE    standalone HAT super-resolution: upscale image, write PPM to stdout\n");
    fprintf(stderr, "                   (needs --hat-model PATH: HAT GGUF, Hybrid Attention Transformer, 2x or 4x)\n");
    fprintf(stderr, "  --hat-model PATH HAT super-resolution GGUF (used with --hat-sr)\n");
    fprintf(stderr, "  --tbsrn-sr FILE  standalone TBSRN text-line SR: upscale text crop, write PPM to stdout\n");
    fprintf(stderr, "                   (needs --tbsrn-model PATH: TBSRN GGUF, Telescope, fixed 4x)\n");
    fprintf(stderr, "  --tbsrn-model PATH TBSRN text-line super-resolution GGUF (used with --tbsrn-sr)\n");
    fprintf(stderr, "  --ocr-det MODEL  general OCR: text detection model (DBNet/surya-det)\n");
    fprintf(stderr, "  --ocr-rec MODEL  general OCR: text recognition model (TrOCR, e.g. trocr-printed)\n");
    fprintf(stderr, "                   use with --ocr IMAGE: detects text regions then recognizes each crop\n");
    fprintf(stderr, "  --conf N         confidence threshold for detection (default: 0.5)\n");
    fprintf(stderr, "  --auto-download  download model automatically if not found\n");
    fprintf(stderr, "  --accept-license SPDX  pre-accept a restricted license (e.g. cc-by-nc-4.0, gemma)\n");
    fprintf(stderr, "                          required for non-commercial / vendor-licensed models in non-TTY mode.\n");
    fprintf(stderr, "                          alternatively set the CRISPEMBED_ACCEPT_LICENSE env var.\n");
    fprintf(stderr, "  --list-models    list available models (license column marks * for restricted)\n");
    fprintf(stderr, "  --cache-dir DIR  set model cache directory\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Model names (auto-download from HuggingFace):\n");
    fprintf(stderr, "  all-MiniLM-L6-v2, gte-small, arctic-embed-xs,\n");
    fprintf(stderr, "  multilingual-e5-small, pixie-rune-v1, arctic-embed-l-v2,\n");
    fprintf(stderr, "  octen-0.6b, f2llm-v2-0.6b, jina-v5-nano, jina-v5-small,\n");
    fprintf(stderr, "  harrier-0.6b, harrier-270m, qwen3-embed-0.6b,\n");
    fprintf(stderr, "  yunet (face detection, 0.2 MB)\n");
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
    std::string model_arg;
    std::string file_path;
    std::string prefix;
    bool prefix_set = false;  // true if --prefix was explicitly provided
    std::string rerank_query;
    std::string biencoder_query;
    std::vector<std::string> texts;
    int n_threads = 4;
    int output_dim = 0;  // 0 = model default
    int top_n = 0;       // 0 = all
    bool json_output = false;
    bool print_dim = false;
    bool print_capabilities = false;
    bool auto_download = false;
    std::string accepted_license;  // for cc-by-nc-*, gemma, etc.
    bool sparse_mode = false;
    bool colbert_mode = false;
    std::string audio_path;  // .raw float32 16 kHz mono PCM
    std::string image_raw_path;  // preprocessed float32 patches, n_patches x 1536
    std::string grid_thw_arg;
    std::string image_path;  // JPG/PNG/BMP — in-process preprocessor
    std::string face_path;   // face image for CNN face recognition
    std::string detect_path; // image for face detection
    std::string ocr_path;    // image for unified math OCR (auto-detect arch)
    std::string hmer_path;   // image for handwritten math OCR (HMER)
    std::string bttr_path;   // image for handwritten math OCR (BTTR)
    std::string layout_path; // image for layout detection
    std::string ner_text;    // text for NER extraction
    std::string ner_labels = "person,organization,location"; // comma-separated entity types
    float ner_threshold = 0.5f;
    std::string kie_path;    // image for KIE extraction
    std::string kie_labels;  // comma-separated field names (defaults to ner_labels)
    float kie_threshold = 0.5f;
    std::string lilt_path;   // JSON file for LiLT token classification
    std::string det_model;   // detection model for --face-pipeline
    std::string ocr_det_path;  // general OCR: text detection model (DBNet)
    std::string ocr_rec_path;  // general OCR: text recognition model (TrOCR)
    bool cleanup_mode = false;          // --cleanup: preprocess before OCR
    std::string cleanup_only_path;      // --cleanup-only FILE: standalone cleanup
    std::string ocr_pipeline_path;      // --ocr-pipeline FILE: full orchestrator
    bool pipeline_denoise = false;      // --denoise: NAFNet tier-2 in the pipeline
    std::string sr_model;              // --sr-model: text super-resolution GGUF
    std::string pan_model;             // --pan-model: PAN super-resolution GGUF
    std::string pan_sr_path;           // --pan-sr FILE: standalone PAN upscaling
    std::string hat_model;             // --hat-model: HAT super-resolution GGUF
    std::string hat_sr_path;           // --hat-sr FILE: standalone HAT upscaling
    std::string tbsrn_model;           // --tbsrn-model: TBSRN text-line SR GGUF
    std::string tbsrn_sr_path;         // --tbsrn-sr FILE: standalone TBSRN upscaling
    std::string restormer_model;        // --restormer MODEL: Restormer image restoration GGUF
    std::string restormer_path;         // --restormer FILE: standalone Restormer processing
    std::string pipeline_vlm_model;     // --vlm-model NAME: VLM escalation engine GGUF
    int pipeline_vlm_engine = 0;        // --vlm-engine: 0=got 1=glm 2=qwen2vl 3=internvl2
    int pipeline_min_chars = -1;        // --ocr-min-chars: accept-gate override (-1 = default)
    float pipeline_min_conf = -1.0f;    // --ocr-min-conf: accept-gate override (-1 = default)
    std::string punct_model;    // --punct-model: post-process OCR with punctuation
    std::string output_format;  // --output-format: text/hocr/alto
    std::string pipeline_engine; // --ocr-engine NAME
    std::string pdf_dpi_path;    // --pdf-dpi FILE
    std::string find_skew_path;  // --find-skew FILE
    std::string dewarp_path;     // --dewarp FILE
    std::string tps_dewarp_model; // --tps-dewarp MODEL FILE
    std::string tps_dewarp_path;
    std::string cc_detect_path;  // --cc-detect FILE
    bool face_pipeline_mode = false;
    float conf_threshold = 0.5f;
    std::string lora_adapter;   // LoRA adapter name (--lora)
    bool list_lora = false;     // --list-lora

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_arg = argv[++i];
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            file_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            output_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prefix") == 0 && i + 1 < argc) {
            prefix = argv[++i];
            prefix_set = true;
        } else if (strcmp(argv[i], "--json") == 0) {
            json_output = true;
        } else if (strcmp(argv[i], "--dim") == 0) {
            print_dim = true;
        } else if (strcmp(argv[i], "--capabilities") == 0) {
            print_capabilities = true;
        } else if (strcmp(argv[i], "--sparse") == 0) {
            sparse_mode = true;
        } else if (strcmp(argv[i], "--colbert") == 0) {
            colbert_mode = true;
        } else if (strcmp(argv[i], "--rerank") == 0 && i + 1 < argc) {
            rerank_query = argv[++i];
        } else if (strcmp(argv[i], "--biencoder") == 0 && i + 1 < argc) {
            biencoder_query = argv[++i];
        } else if (strcmp(argv[i], "--top-n") == 0 && i + 1 < argc) {
            top_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (strcmp(argv[i], "--image-raw") == 0 && i + 1 < argc) {
            image_raw_path = argv[++i];
        } else if (strcmp(argv[i], "--grid-thw") == 0 && i + 1 < argc) {
            grid_thw_arg = argv[++i];
        } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            image_path = argv[++i];
        } else if (strcmp(argv[i], "--face") == 0 && i + 1 < argc) {
            face_path = argv[++i];
        } else if (strcmp(argv[i], "--detect") == 0 && i + 1 < argc) {
            detect_path = argv[++i];
        } else if (strcmp(argv[i], "--ocr") == 0 && i + 1 < argc) {
            ocr_path = argv[++i];
        } else if (strcmp(argv[i], "--hmer") == 0 && i + 1 < argc) {
            hmer_path = argv[++i];
        } else if (strcmp(argv[i], "--bttr") == 0 && i + 1 < argc) {
            bttr_path = argv[++i];
        } else if (strcmp(argv[i], "--layout") == 0 && i + 1 < argc) {
            layout_path = argv[++i];
        } else if (strcmp(argv[i], "--ner") == 0 && i + 1 < argc) {
            ner_text = argv[++i];
        } else if (strcmp(argv[i], "--ner-labels") == 0 && i + 1 < argc) {
            ner_labels = argv[++i];
        } else if (strcmp(argv[i], "--ner-threshold") == 0 && i + 1 < argc) {
            ner_threshold = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--kie") == 0 && i + 1 < argc) {
            kie_path = argv[++i];
        } else if (strcmp(argv[i], "--kie-labels") == 0 && i + 1 < argc) {
            kie_labels = argv[++i];
        } else if (strcmp(argv[i], "--kie-threshold") == 0 && i + 1 < argc) {
            kie_threshold = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--lilt") == 0 && i + 1 < argc) {
            lilt_path = argv[++i];
        } else if (strcmp(argv[i], "--det") == 0 && i + 1 < argc) {
            det_model = argv[++i];
        } else if (strcmp(argv[i], "--face-pipeline") == 0) {
            face_pipeline_mode = true;
        } else if (strcmp(argv[i], "--pdf-dpi") == 0 && i + 1 < argc) {
            pdf_dpi_path = argv[++i];
        } else if (strcmp(argv[i], "--find-skew") == 0 && i + 1 < argc) {
            find_skew_path = argv[++i];
        } else if (strcmp(argv[i], "--dewarp") == 0 && i + 1 < argc) {
            dewarp_path = argv[++i];
        } else if (strcmp(argv[i], "--tps-dewarp") == 0 && i + 2 < argc) {
            tps_dewarp_model = argv[++i];
            tps_dewarp_path  = argv[++i];
        } else if (strcmp(argv[i], "--cc-detect") == 0 && i + 1 < argc) {
            cc_detect_path = argv[++i];
        } else if (strcmp(argv[i], "--cleanup") == 0) {
            cleanup_mode = true;
        } else if (strcmp(argv[i], "--punct-model") == 0 && i + 1 < argc) {
            punct_model = argv[++i];
        } else if (strcmp(argv[i], "--output-format") == 0 && i + 1 < argc) {
            output_format = argv[++i];
        } else if (strcmp(argv[i], "--cleanup-only") == 0 && i + 1 < argc) {
            cleanup_only_path = argv[++i];
        } else if (strcmp(argv[i], "--ocr-pipeline") == 0 && i + 1 < argc) {
            ocr_pipeline_path = argv[++i];
        } else if (strcmp(argv[i], "--ocr-engine") == 0 && i + 1 < argc) {
            pipeline_engine = argv[++i];
        } else if (strcmp(argv[i], "--denoise") == 0) {
            pipeline_denoise = true;
        } else if (strcmp(argv[i], "--sr-model") == 0 && i + 1 < argc) {
            sr_model = argv[++i];
        } else if (strcmp(argv[i], "--pan-model") == 0 && i + 1 < argc) {
            pan_model = argv[++i];
        } else if (strcmp(argv[i], "--pan-sr") == 0 && i + 1 < argc) {
            pan_sr_path = argv[++i];
        } else if (strcmp(argv[i], "--hat-model") == 0 && i + 1 < argc) {
            hat_model = argv[++i];
        } else if (strcmp(argv[i], "--hat-sr") == 0 && i + 1 < argc) {
            hat_sr_path = argv[++i];
        } else if (strcmp(argv[i], "--tbsrn-model") == 0 && i + 1 < argc) {
            tbsrn_model = argv[++i];
        } else if (strcmp(argv[i], "--tbsrn-sr") == 0 && i + 1 < argc) {
            tbsrn_sr_path = argv[++i];
        } else if (strcmp(argv[i], "--restormer") == 0 && i + 1 < argc) {
            restormer_model = argv[++i];
        } else if (strcmp(argv[i], "--restormer-input") == 0 && i + 1 < argc) {
            restormer_path = argv[++i];
        } else if (strcmp(argv[i], "--vlm-model") == 0 && i + 1 < argc) {
            pipeline_vlm_model = argv[++i];
        } else if (strcmp(argv[i], "--vlm-engine") == 0 && i + 1 < argc) {
            pipeline_vlm_engine = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ocr-min-chars") == 0 && i + 1 < argc) {
            pipeline_min_chars = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ocr-min-conf") == 0 && i + 1 < argc) {
            pipeline_min_conf = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--ocr-det") == 0 && i + 1 < argc) {
            ocr_det_path = argv[++i];
        } else if (strcmp(argv[i], "--ocr-rec") == 0 && i + 1 < argc) {
            ocr_rec_path = argv[++i];
        } else if (strcmp(argv[i], "--conf") == 0 && i + 1 < argc) {
            conf_threshold = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--lora") == 0 && i + 1 < argc) {
            lora_adapter = argv[++i];
        } else if (strcmp(argv[i], "--list-lora") == 0) {
            list_lora = true;
        } else if (strcmp(argv[i], "--auto-download") == 0) {
            auto_download = true;
        } else if (strcmp(argv[i], "--accept-license") == 0 && i + 1 < argc) {
            accepted_license = argv[++i];
        } else if (strcmp(argv[i], "--list-models") == 0) {
            crispembed_mgr::list_models();
            return 0;
        } else if (strcmp(argv[i], "--cache-dir") == 0 && i + 1 < argc) {
#ifdef _WIN32
            _putenv_s("CRISPEMBED_CACHE_DIR", argv[++i]);
#else
            setenv("CRISPEMBED_CACHE_DIR", argv[++i], 1);
#endif
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            texts.push_back(argv[i]);
        }
    }

    // Standalone preprocessing commands (no model needed)
    if (!pdf_dpi_path.empty()) {
        int n_pages = 0;
        pdf_page_dpi_result * results = pdf_all_pages_dpi(pdf_dpi_path.c_str(), &n_pages);
        if (!results || n_pages <= 0) {
            fprintf(stderr, "error: cannot read PDF '%s'\n", pdf_dpi_path.c_str());
            return 1;
        }
        printf("{\"pages\":[");
        for (int i = 0; i < n_pages; i++) {
            if (i > 0) printf(",");
            printf("{\"page\":%d,\"dpi\":%.1f,\"dpi_min\":%.1f,\"dpi_max\":%.1f,"
                   "\"n_images\":%d,\"page_width_pt\":%.1f,\"page_height_pt\":%.1f}",
                   i, results[i].dpi, results[i].dpi_min, results[i].dpi_max,
                   results[i].n_images, results[i].page_width_pt, results[i].page_height_pt);
        }
        printf("]}\n");
        pdf_dpi_free(results);
        return 0;
    }
    if (!find_skew_path.empty()) {
        int w, h, ch;
        unsigned char * data = stbi_load(find_skew_path.c_str(), &w, &h, &ch, 1);
        if (!data) { fprintf(stderr, "error: cannot load %s\n", find_skew_path.c_str()); return 1; }
        float angle = 0, conf = 0;
        crispembed_find_skew(data, w, h, &angle, &conf);
        stbi_image_free(data);
        if (json_output) printf("{\"angle\":%.3f,\"confidence\":%.3f}\n", angle, conf);
        else printf("angle=%.3f deg  confidence=%.3f\n", angle, conf);
        return 0;
    }
    if (!dewarp_path.empty()) {
        int w, h, ch;
        unsigned char * data = stbi_load(dewarp_path.c_str(), &w, &h, &ch, 1);
        if (!data) { fprintf(stderr, "error: cannot load %s\n", dewarp_path.c_str()); return 1; }
        std::vector<uint8_t> out(w * h);
        int ow = 0, oh = 0;
        int ret = crispembed_dewarp(data, w, h, out.data(), &ow, &oh);
        stbi_image_free(data);
        if (ret != 0) { fprintf(stderr, "dewarp failed (too few textlines?)\n"); return 1; }
        // Write result as PGM to stdout
        printf("P5\n%d %d\n255\n", ow, oh);
        fwrite(out.data(), 1, ow * oh, stdout);
        return 0;
    }
    if (!tps_dewarp_path.empty()) {
        int w, h, ch;
        unsigned char * data = stbi_load(tps_dewarp_path.c_str(), &w, &h, &ch, 1);
        if (!data) { fprintf(stderr, "error: cannot load %s\n", tps_dewarp_path.c_str()); return 1; }
        std::vector<uint8_t> out(w * h);
        int ret = crispembed_tps_auto_dewarp(data, w, h, tps_dewarp_model.c_str(), out.data());
        stbi_image_free(data);
        if (ret != 0) { fprintf(stderr, "tps-dewarp failed\n"); return 1; }
        printf("P5\n%d %d\n255\n", w, h);
        fwrite(out.data(), 1, w * h, stdout);
        return 0;
    }
    if (!cc_detect_path.empty()) {
        int w, h, ch;
        unsigned char * data = stbi_load(cc_detect_path.c_str(), &w, &h, &ch, 1);
        if (!data) { fprintf(stderr, "error: cannot load %s\n", cc_detect_path.c_str()); return 1; }
        int n = 0;
        crispembed_ocr_result * regions = crispembed_cc_detect(data, w, h, &n);
        stbi_image_free(data);
        if (json_output) {
            printf("[");
            for (int i = 0; i < n; i++) {
                if (i > 0) printf(",");
                printf("{\"x\":%.0f,\"y\":%.0f,\"w\":%.0f,\"h\":%.0f}",
                       regions[i].x, regions[i].y, regions[i].w, regions[i].h);
            }
            printf("]\n");
        } else {
            printf("detected %d text regions\n", n);
            for (int i = 0; i < n; i++)
                printf("  [%d] (%.0f, %.0f) %.0fx%.0f\n", i,
                       regions[i].x, regions[i].y, regions[i].w, regions[i].h);
        }
        if (regions) free(regions);
        return 0;
    }
    if (!pan_sr_path.empty()) {
        if (pan_model.empty()) {
            fprintf(stderr, "error: --pan-sr requires --pan-model <path>\n");
            return 1;
        }
        int w, h, ch;
        unsigned char * data = stbi_load(pan_sr_path.c_str(), &w, &h, &ch, 3);
        if (!data) { fprintf(stderr, "error: cannot load %s\n", pan_sr_path.c_str()); return 1; }
        void * pctx = crispembed_pan_sr_init(pan_model.c_str(), n_threads);
        if (!pctx) { stbi_image_free(data); fprintf(stderr, "error: cannot load PAN model '%s'\n", pan_model.c_str()); return 1; }
        uint8_t * out = nullptr;
        int ow = 0, oh = 0;
        int rc = crispembed_pan_sr_process(pctx, data, w, h, 0, 0, &out, &ow, &oh);
        stbi_image_free(data);
        crispembed_pan_sr_free(pctx);
        if (rc != 0 || !out) { fprintf(stderr, "error: PAN SR processing failed\n"); return 1; }
        // Write result as PPM (RGB) to stdout
        printf("P6\n%d %d\n255\n", ow, oh);
        fwrite(out, 1, (size_t)ow * oh * 3, stdout);
        crispembed_pan_sr_free_image(out);
        return 0;
    }
    if (!hat_sr_path.empty()) {
        if (hat_model.empty()) {
            fprintf(stderr, "error: --hat-sr requires --hat-model <path>\n");
            return 1;
        }
        int w, h, ch;
        unsigned char * data = stbi_load(hat_sr_path.c_str(), &w, &h, &ch, 3);
        if (!data) { fprintf(stderr, "error: cannot load %s\n", hat_sr_path.c_str()); return 1; }
        void * hctx = crispembed_hat_sr_init(hat_model.c_str(), n_threads);
        if (!hctx) { stbi_image_free(data); fprintf(stderr, "error: cannot load HAT model '%s'\n", hat_model.c_str()); return 1; }
        uint8_t * out = nullptr;
        int ow = 0, oh = 0;
        int rc = crispembed_hat_sr_process(hctx, data, w, h, 64, 8, &out, &ow, &oh);
        stbi_image_free(data);
        crispembed_hat_sr_free(hctx);
        if (rc != 0 || !out) { fprintf(stderr, "error: HAT SR processing failed\n"); return 1; }
        // Write result as PPM (RGB) to stdout
        printf("P6\n%d %d\n255\n", ow, oh);
        fwrite(out, 1, (size_t)ow * oh * 3, stdout);
        crispembed_hat_sr_free_image(out);
        return 0;
    }
    if (!tbsrn_sr_path.empty()) {
        if (tbsrn_model.empty()) {
            fprintf(stderr, "error: --tbsrn-sr requires --tbsrn-model <path>\n");
            return 1;
        }
        int w, h, ch;
        unsigned char * data = stbi_load(tbsrn_sr_path.c_str(), &w, &h, &ch, 3);
        if (!data) { fprintf(stderr, "error: cannot load %s\n", tbsrn_sr_path.c_str()); return 1; }
        void * tctx = crispembed_tbsrn_sr_init(tbsrn_model.c_str(), n_threads);
        if (!tctx) { stbi_image_free(data); fprintf(stderr, "error: cannot load TBSRN model '%s'\n", tbsrn_model.c_str()); return 1; }
        uint8_t * out = nullptr;
        int ow = 0, oh = 0;
        int rc = crispembed_tbsrn_sr_process(tctx, data, w, h, &out, &ow, &oh);
        stbi_image_free(data);
        crispembed_tbsrn_sr_free(tctx);
        if (rc != 0 || !out) { fprintf(stderr, "error: TBSRN SR processing failed\n"); return 1; }
        // Write result as PPM (RGB) to stdout
        printf("P6\n%d %d\n255\n", ow, oh);
        fwrite(out, 1, (size_t)ow * oh * 3, stdout);
        crispembed_tbsrn_sr_free_image(out);
        return 0;
    }
    if (!restormer_path.empty()) {
        if (restormer_model.empty()) {
            fprintf(stderr, "error: --restormer-input requires --restormer <model>\n");
            return 1;
        }
        int w, h, ch;
        unsigned char * data = stbi_load(restormer_path.c_str(), &w, &h, &ch, 3);
        if (!data) { fprintf(stderr, "error: cannot load %s\n", restormer_path.c_str()); return 1; }
        void * rctx = crispembed_restormer_init(restormer_model.c_str(), n_threads);
        if (!rctx) { stbi_image_free(data); fprintf(stderr, "error: cannot load Restormer model '%s'\n", restormer_model.c_str()); return 1; }
        uint8_t * out = nullptr;
        int rc = crispembed_restormer_process(rctx, data, w, h, 0, 0, &out);
        stbi_image_free(data);
        crispembed_restormer_free(rctx);
        if (rc != 0 || !out) { fprintf(stderr, "error: Restormer processing failed\n"); return 1; }
        // Write result as PPM (RGB) to stdout
        printf("P6\n%d %d\n255\n", w, h);
        fwrite(out, 1, (size_t)w * h * 3, stdout);
        crispembed_restormer_free_image(out);
        return 0;
    }

    if (model_arg.empty() && cleanup_only_path.empty() && ocr_pipeline_path.empty()) {
        fprintf(stderr, "error: no model specified (-m)\n");
        print_usage(argv[0]);
        return 1;
    }

    int mode_count = 0;
    mode_count += sparse_mode ? 1 : 0;
    mode_count += colbert_mode ? 1 : 0;
    mode_count += !rerank_query.empty() ? 1 : 0;
    mode_count += !biencoder_query.empty() ? 1 : 0;
    mode_count += !audio_path.empty() ? 1 : 0;
    mode_count += !image_raw_path.empty() ? 1 : 0;
    mode_count += !image_path.empty() ? 1 : 0;
    if (mode_count > 1) {
        fprintf(stderr, "error: choose only one of --sparse, --colbert, --rerank, --biencoder, --audio, --image, or --image-raw\n");
        return 1;
    }

    // Standalone scan cleanup (--cleanup-only FILE)
    if (!cleanup_only_path.empty()) {
        int w, h, ch;
        unsigned char * data = stbi_load(cleanup_only_path.c_str(), &w, &h, &ch, 0);
        if (!data) { fprintf(stderr, "error: cannot load %s\n", cleanup_only_path.c_str()); return 1; }
        auto * sctx = scan_cleanup_init(nullptr, n_threads);
        auto params = scan_cleanup_defaults();
        uint8_t * out = nullptr;
        int ow = 0, oh = 0;
        int rc = scan_cleanup_process(sctx, data, w, h, ch, params, &out, &ow, &oh);
        stbi_image_free(data);
        scan_cleanup_free(sctx);
        if (rc != 0 || !out) { fprintf(stderr, "error: scan cleanup failed\n"); return 1; }
        if (json_output) {
            printf("{\"width\":%d,\"height\":%d,\"original_width\":%d,\"original_height\":%d}\n", ow, oh, w, h);
        } else {
            printf("cleaned: %dx%d -> %dx%d\n", w, h, ow, oh);
        }
        scan_cleanup_free_image(out);
        return 0;
    }

    // Full OCR pipeline orchestrator (--ocr-pipeline FILE): source-type
    // routing + per-stage cleanup (classical + optional NAFNet denoise) +
    // accept-gate escalation. Models default to surya-det + qwen2vl-ocr.
    if (!ocr_pipeline_path.empty()) {
        auto resolve = [&](const std::string & n) {
            return crispembed_mgr::resolve_model(n, true, accepted_license);
        };
        auto eng_id = [](const std::string & n) -> int {
            if (n == "surya")     return 1;
            if (n == "got")       return 2;
            if (n == "glm")       return 3;
            if (n == "qwen2vl")   return 4;
            if (n == "internvl2") return 5;
            if (n == "tesseract") return 6;
            return 0; // dbnet_trocr
        };
        std::string nafnet, vlm, punct;
        if (pipeline_denoise)            nafnet = resolve("nafnet-denoise");
        if (!pipeline_vlm_model.empty()) vlm    = resolve(pipeline_vlm_model);
        if (!punct_model.empty())        punct  = resolve(punct_model);
        const int   min_chars = pipeline_min_chars >= 0 ? pipeline_min_chars : 8;
        const float min_conf  = pipeline_min_conf  >= 0 ? pipeline_min_conf  : 0.5f;

        void * pctx = nullptr;
        // Keep model strings alive until the init call returns (it copies them).
        std::string ma, mb, det, rec;
        if (!pipeline_engine.empty()) {
            // Explicit primary engine → single-stage pipeline via the builder.
            const int eid = eng_id(pipeline_engine);
            const bool is_vlm = (eid >= 2 && eid <= 5);
            if (is_vlm) {
                const char * dflt = (eid == 2) ? "got-ocr2"
                                  : (eid == 3) ? "glm-ocr"
                                  : (eid == 5) ? "internvl2-ocr" : "qwen2vl-ocr";
                ma = resolve(!ocr_rec_path.empty() ? ocr_rec_path : dflt);
            } else {
                ma = resolve(ocr_det_path.empty() ? "dbnet-det" : ocr_det_path);
                const char * rdflt = (eid == 6) ? "tesseract-eng" : "qwen2vl-ocr";
                mb = resolve(ocr_rec_path.empty() ? rdflt : ocr_rec_path);
            }
            crispembed_ocr_stage st;
            std::memset(&st, 0, sizeof(st));
            st.source_type        = 0;          // auto
            st.engine             = eid;
            st.model_a            = ma.c_str();
            st.model_b            = mb.empty() ? nullptr : mb.c_str();
            st.cleanup_enabled    = 1;
            st.denoise            = pipeline_denoise ? 1 : 0;
            st.cleanup            = crispembed_scan_cleanup_defaults();
            st.det_prob_threshold = 0.3f;
            st.det_box_threshold  = 0.5f;
            st.det_target_short   = 736;
            st.vlm_max_tokens     = 0;
            st.vlm_prompt         = nullptr;
            st.min_chars          = min_chars;
            st.min_confidence     = min_conf;
            pctx = crispembed_ocr_pipeline_init_stages(
                /*router=*/0,
                nafnet.empty()    ? nullptr : nafnet.c_str(),
                sr_model.empty()  ? nullptr : sr_model.c_str(),
                punct.empty()     ? nullptr : punct.c_str(),
                &st, 1, n_threads);
        } else {
            // Default flat path (DBNet+TrOCR + source-type routing).
            det = resolve(ocr_det_path.empty() ? "dbnet-det" : ocr_det_path);
            rec = resolve(ocr_rec_path.empty() ? "qwen2vl-ocr" : ocr_rec_path);
            crispembed_ocr_pipeline_params pp = crispembed_ocr_pipeline_defaults();
            pp.det_model    = det.c_str();
            pp.rec_model    = rec.c_str();
            pp.nafnet_model = nafnet.empty() ? nullptr : nafnet.c_str();
            pp.vlm_model    = vlm.empty()    ? nullptr : vlm.c_str();
            pp.vlm_engine   = pipeline_vlm_engine;
            pp.punct_model  = punct.empty()  ? nullptr : punct.c_str();
            pp.min_chars     = min_chars;
            pp.min_confidence = min_conf;
            pctx = crispembed_ocr_pipeline_init(&pp, n_threads);
        }
        if (!pctx) { fprintf(stderr, "error: cannot init OCR pipeline\n"); return 1; }
        int n_res = 0;
        const char* full_text = nullptr;
        float mean_conf = 0.0f;
        const crispembed_ocr_result* res = crispembed_ocr_pipeline_run(
            pctx, ocr_pipeline_path.c_str(), &n_res, &full_text, &mean_conf);

        // Output in requested format
        if (!output_format.empty() && output_format != "text") {
            // Structured output (hOCR, ALTO, PDF)
            // Need page dimensions — load image to get them
            int pw = 0, ph = 0, pc = 0;
            unsigned char * pimg = stbi_load(ocr_pipeline_path.c_str(), &pw, &ph, &pc, 0);
            if (pimg) stbi_image_free(pimg);
            if (pw == 0) { pw = 2480; ph = 3508; } // A4 fallback

            char * rendered = crispembed_ocr_render(res, n_res, pw, ph,
                                                     output_format.c_str());
            if (rendered) {
                printf("%s", rendered);
                free(rendered);
            }
        } else if (json_output) {
            printf("{\"n_regions\":%d,\"mean_confidence\":%.3f,\"full_text\":\"%s\"}\n",
                   n_res, mean_conf, json_escape(full_text ? full_text : "").c_str());
        } else {
            printf("regions=%d  mean_conf=%.2f\n%s\n",
                   n_res, mean_conf, full_text ? full_text : "");
        }
        crispembed_ocr_pipeline_free(pctx);
        return 0;
    }

    // General OCR pipeline via --ocr-det/--ocr-rec (preferred new flags)
    if (!ocr_det_path.empty() && !ocr_rec_path.empty() && !ocr_path.empty()) {
        void* ocr_ctx = crispembed_ocr_init(ocr_det_path.c_str(), ocr_rec_path.c_str(), n_threads);
        if (!ocr_ctx) { fprintf(stderr, "error: cannot load OCR models\n"); return 1; }
        int n_results = 0;
        const crispembed_ocr_result* results = crispembed_ocr(ocr_ctx, ocr_path.c_str(), &n_results);
        if (json_output) {
            printf("[");
            for (int i = 0; i < n_results; i++) {
                if (i > 0) printf(",");
                printf("{\"text\":\"%s\",\"bbox\":[%.0f,%.0f,%.0f,%.0f],\"conf\":%.3f}",
                       json_escape(results[i].text).c_str(),
                       results[i].x, results[i].y,
                       results[i].x + results[i].w, results[i].y + results[i].h,
                       results[i].confidence);
            }
            printf("]\n");
        } else {
            for (int i = 0; i < n_results; i++) {
                printf("[%2d] (%.0f,%.0f)-(%.0f,%.0f) conf=%.2f  \"%s\"\n",
                       i, results[i].x, results[i].y,
                       results[i].x + results[i].w, results[i].y + results[i].h,
                       results[i].confidence, results[i].text);
            }
            if (n_results == 0) printf("(no text detected)\n");
        }
        crispembed_ocr_free(ocr_ctx);
        return 0;
    }

    // OCR pipeline early exit — before model resolution which may interfere
    // Legacy path: detect text → crop → recognize (requires --det + -m)
    if (!ocr_path.empty() && !det_model.empty()) {
        if (model_arg.empty()) {
            fprintf(stderr, "error: --ocr with --det requires -m <trocr_model.gguf>\n");
            return 1;
        }

        void* ocr_ctx = crispembed_ocr_init(det_model.c_str(), model_arg.c_str(), n_threads);
        if (!ocr_ctx) {
            fprintf(stderr, "error: cannot load OCR models\n");
            return 1;
        }

        int n_results = 0;
        const crispembed_ocr_result* results = crispembed_ocr(ocr_ctx, ocr_path.c_str(), &n_results);

        if (json_output) {
            printf("[");
            for (int i = 0; i < n_results; i++) {
                if (i > 0) printf(",");
                printf("{\"text\":\"%s\",\"bbox\":[%.0f,%.0f,%.0f,%.0f],\"conf\":%.3f}",
                       json_escape(results[i].text).c_str(),
                       results[i].x, results[i].y,
                       results[i].x + results[i].w, results[i].y + results[i].h,
                       results[i].confidence);
            }
            printf("]\n");
        } else {
            for (int i = 0; i < n_results; i++) {
                printf("[%2d] (%.0f,%.0f)-(%.0f,%.0f) conf=%.2f  \"%s\"\n",
                       i, results[i].x, results[i].y,
                       results[i].x + results[i].w, results[i].y + results[i].h,
                       results[i].confidence, results[i].text);
            }
            if (n_results == 0) printf("(no text detected)\n");
        }

        crispembed_ocr_free(ocr_ctx);
        return 0;
    }

    // Resolve model path (handles auto-download) — after OCR early exit
    bool is_name = (model_arg.find(".gguf") == std::string::npos &&
                    model_arg.find('/') == std::string::npos &&
                    model_arg.find('\\') == std::string::npos);
    std::string model_path = crispembed_mgr::resolve_model(
        model_arg, auto_download || is_name, accepted_license);
    if (model_path.empty()) {
        return 1;
    }

    // Load from file if specified
    if (!file_path.empty()) {
        std::ifstream f(file_path);
        if (!f) {
            fprintf(stderr, "error: cannot open '%s'\n", file_path.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty()) texts.push_back(line);
        }
    }

    // Face pipeline mode: detect → align → encode → compare
    if (face_pipeline_mode) {
        if (det_model.empty()) {
            fprintf(stderr, "error: --face-pipeline requires --det <detection_model.gguf>\n");
            return 1;
        }
        if (texts.empty()) {
            fprintf(stderr, "error: --face-pipeline requires image file arguments\n");
            return 1;
        }

        // Load detection model
        cnn_embed::context* det_ctx = nullptr;
        if (!cnn_embed::load(&det_ctx, det_model.c_str(), n_threads)) {
            fprintf(stderr, "error: cannot load detection model '%s'\n", det_model.c_str());
            return 1;
        }

        // Load recognition model (-m)
        cnn_embed::context* rec_ctx = nullptr;
        if (!cnn_embed::load(&rec_ctx, model_path.c_str(), n_threads)) {
            fprintf(stderr, "error: cannot load recognition model '%s'\n", model_path.c_str());
            cnn_embed::free(det_ctx);
            return 1;
        }

        // Run pipeline on each image
        struct image_faces {
            std::string name;
            std::vector<cnn_embed::face_result> faces;
        };
        std::vector<image_faces> all_images;

        for (const auto& img_path : texts) {
            auto results = cnn_embed::face_pipeline(det_ctx, rec_ctx, img_path.c_str(),
                                                     conf_threshold);
            // Extract filename for display
            std::string name = img_path;
            auto sl = name.find_last_of("/\\");
            if (sl != std::string::npos) name = name.substr(sl + 1);

            if (json_output) {
                printf("{\"image\": \"%s\", \"faces\": [", json_escape(img_path).c_str());
                for (size_t j = 0; j < results.size(); j++) {
                    const auto& r = results[j];
                    printf("{\"bbox\":[%.1f,%.1f,%.1f,%.1f],\"conf\":%.4f,\"landmarks\":[",
                           r.det.x, r.det.y, r.det.w, r.det.h, r.det.confidence);
                    for (int k = 0; k < 10; k++)
                        printf("%.1f%s", r.det.landmarks[k], k < 9 ? "," : "");
                    printf("],\"embedding\":[");
                    for (size_t k = 0; k < r.embedding.size(); k++)
                        printf("%.6f%s", r.embedding[k], k + 1 < r.embedding.size() ? "," : "");
                    printf("]}%s", j + 1 < results.size() ? "," : "");
                }
                printf("]}\n");
            } else {
                fprintf(stderr, "%s: %zu faces detected\n", name.c_str(), results.size());
                for (size_t j = 0; j < results.size(); j++) {
                    const auto& r = results[j];
                    fprintf(stderr, "  face[%zu]: conf=%.4f bbox=[%.0f,%.0f,%.0f,%.0f] dim=%zu\n",
                            j, r.det.confidence, r.det.x, r.det.y, r.det.w, r.det.h,
                            r.embedding.size());
                }
            }

            all_images.push_back({name, std::move(results)});
        }

        // Cross-image face matching (if multiple images)
        if (all_images.size() > 1 && !json_output) {
            fprintf(stderr, "\n=== Cross-image face matching ===\n");
            for (size_t i = 0; i < all_images.size(); i++) {
                for (size_t j = i + 1; j < all_images.size(); j++) {
                    fprintf(stderr, "\n%s vs %s:\n",
                            all_images[i].name.c_str(), all_images[j].name.c_str());
                    for (size_t a = 0; a < std::min((size_t)3, all_images[i].faces.size()); a++) {
                        for (size_t b = 0; b < std::min((size_t)3, all_images[j].faces.size()); b++) {
                            const auto& ea = all_images[i].faces[a].embedding;
                            const auto& eb = all_images[j].faces[b].embedding;
                            if (ea.size() != eb.size() || ea.empty()) continue;
                            float cos = 0;
                            for (size_t k = 0; k < ea.size(); k++) cos += ea[k] * eb[k];
                            const char* tag = cos > 0.4f ? "MATCH" : "no";
                            fprintf(stderr, "  [%zu]vs[%zu] cos=%.4f %s\n", a, b, cos, tag);
                        }
                    }
                }
            }
        }

        cnn_embed::free(det_ctx);
        cnn_embed::free(rec_ctx);
        return 0;
    }

    // Check if this is a CNN face model (SFace/AuraFace/SCRFD).
    if (!face_path.empty() || !detect_path.empty() || print_dim) {
        cnn_embed::context* cctx = nullptr;
        if (cnn_embed::load(&cctx, model_path.c_str(), n_threads)) {
            if (print_dim) {
                printf("%d\n", cnn_embed::dim(cctx));
                cnn_embed::free(cctx);
                return 0;
            }
            // Face detection mode
            if (!detect_path.empty()) {
                auto faces = cnn_embed::detect_file(cctx, detect_path.c_str(),
                                                     conf_threshold);
                if (json_output) {
                    printf("{\"faces\": [");
                    for (size_t j = 0; j < faces.size(); j++) {
                        const auto& f = faces[j];
                        printf("{\"x\":%.1f,\"y\":%.1f,\"w\":%.1f,\"h\":%.1f,\"conf\":%.4f,"
                               "\"landmarks\":[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f]}",
                               f.x, f.y, f.w, f.h, f.confidence,
                               f.landmarks[0], f.landmarks[1], f.landmarks[2], f.landmarks[3],
                               f.landmarks[4], f.landmarks[5], f.landmarks[6], f.landmarks[7],
                               f.landmarks[8], f.landmarks[9]);
                        if (j + 1 < faces.size()) printf(",");
                    }
                    printf("]}\n");
                } else {
                    for (const auto& f : faces) {
                        printf("%.1f %.1f %.1f %.1f %.3f", f.x, f.y, f.w, f.h, f.confidence);
                        for (int k = 0; k < 10; k++) printf(" %.1f", f.landmarks[k]);
                        printf("\n");
                    }
                    if (faces.empty()) printf("(no faces detected)\n");
                }
                cnn_embed::free(cctx);
                return 0;
            }
            auto emb = cnn_embed::encode_file(cctx, face_path.c_str());
            if (emb.empty()) {
                fprintf(stderr, "error: face encoding failed for '%s'\n", face_path.c_str());
                cnn_embed::free(cctx);
                return 1;
            }
            if (json_output) {
                printf("{\"face\": \"%s\", \"embedding\": [", json_escape(face_path).c_str());
                for (size_t j = 0; j < emb.size(); j++)
                    printf("%.6f%s", emb[j], j + 1 < emb.size() ? ", " : "");
                printf("]}\n");
            } else {
                for (size_t j = 0; j < emb.size(); j++)
                    printf("%.6f%s", emb[j], j + 1 < emb.size() ? " " : "\n");
            }
            cnn_embed::free(cctx);
            return 0;
        }
        // Not a CNN model — fall through
    }

    // Layout detection (RT-DETRv2)
    if (!layout_path.empty()) {
        void* lctx = crispembed_layout_init(model_path.c_str(), n_threads);
        if (!lctx) { fprintf(stderr, "error: failed to load layout model\n"); return 1; }
        int n_regions = 0;
        const crispembed_layout_region* regions = crispembed_layout_detect(
            lctx, layout_path.c_str(), conf_threshold, &n_regions);
        if (json_output) {
            printf("{\"regions\": [");
            for (int i = 0; i < n_regions; i++) {
                if (i > 0) printf(", ");
                printf("{\"label\": \"%s\", \"score\": %.4f, \"bbox\": [%.1f, %.1f, %.1f, %.1f]}",
                       regions[i].label_name, regions[i].score,
                       regions[i].x1, regions[i].y1, regions[i].x2, regions[i].y2);
            }
            printf("]}\n");
        } else {
            printf("%d regions detected:\n", n_regions);
            for (int i = 0; i < n_regions; i++) {
                printf("  [%d] %s (%.3f) [%.1f, %.1f, %.1f, %.1f]\n",
                       i, regions[i].label_name, regions[i].score,
                       regions[i].x1, regions[i].y1, regions[i].x2, regions[i].y2);
            }
        }
        crispembed_layout_free(lctx);
        return 0;
    }

    // Named Entity Recognition (GLiNER)
    if (!ner_text.empty()) {
        void* nctx = crispembed_ner_init(model_path.c_str(), n_threads);
        if (!nctx) { fprintf(stderr, "error: failed to load NER model\n"); return 1; }

        // Parse comma-separated labels
        std::vector<std::string> label_strs;
        std::istringstream lss(ner_labels);
        std::string lbl;
        while (std::getline(lss, lbl, ',')) {
            // trim whitespace
            size_t start = lbl.find_first_not_of(" \t");
            size_t end = lbl.find_last_not_of(" \t");
            if (start != std::string::npos)
                label_strs.push_back(lbl.substr(start, end - start + 1));
        }
        std::vector<const char*> label_ptrs;
        for (const auto& s : label_strs) label_ptrs.push_back(s.c_str());

        crispembed_ner_entity* entities = nullptr;
        int n_entities = crispembed_ner_extract(
            nctx, ner_text.c_str(),
            label_ptrs.data(), (int)label_ptrs.size(),
            ner_threshold, &entities);

        if (json_output) {
            printf("{\"entities\": [");
            for (int i = 0; i < n_entities; i++) {
                if (i > 0) printf(", ");
                printf("{\"text\": \"%s\", \"label\": \"%s\", \"start\": %d, \"end\": %d, \"score\": %.4f}",
                       json_escape(entities[i].text).c_str(),
                       json_escape(entities[i].label).c_str(),
                       entities[i].start_char, entities[i].end_char,
                       entities[i].score);
            }
            printf("]}\n");
        } else {
            printf("%d entities found:\n", n_entities);
            for (int i = 0; i < n_entities; i++) {
                printf("  [%d] %s (%s) [%d, %d) score=%.4f\n",
                       i, entities[i].text, entities[i].label,
                       entities[i].start_char, entities[i].end_char,
                       entities[i].score);
            }
        }
        crispembed_ner_free(nctx);
        return 0;
    }

    // Key Information Extraction (OCR + NER pipeline)
    if (!kie_path.empty()) {
        auto resolve = [&](const std::string & n) {
            return crispembed_mgr::resolve_model(n, auto_download, accepted_license);
        };
        std::string det = resolve(ocr_det_path.empty() ? "dbnet-det" : ocr_det_path);
        std::string rec = resolve(ocr_rec_path.empty() ? "qwen2vl-ocr" : ocr_rec_path);

        void* kctx = crispembed_kie_init(det.c_str(), rec.c_str(),
                                          model_path.c_str(), n_threads);
        if (!kctx) { fprintf(stderr, "error: failed to init KIE pipeline\n"); return 1; }

        // Parse comma-separated labels (use --kie-labels, fall back to --ner-labels).
        const std::string& labels_str = kie_labels.empty() ? ner_labels : kie_labels;
        std::vector<std::string> label_strs;
        std::istringstream lss(labels_str);
        std::string lbl;
        while (std::getline(lss, lbl, ',')) {
            size_t start = lbl.find_first_not_of(" \t");
            size_t end = lbl.find_last_not_of(" \t");
            if (start != std::string::npos)
                label_strs.push_back(lbl.substr(start, end - start + 1));
        }
        std::vector<const char*> label_ptrs;
        for (const auto& s : label_strs) label_ptrs.push_back(s.c_str());

        crispembed_kie_result res = crispembed_kie_extract(
            kctx, kie_path.c_str(),
            label_ptrs.data(), (int)label_ptrs.size(),
            kie_threshold);

        if (json_output) {
            printf("{\"n_ocr_regions\":%d,\"ocr_confidence\":%.3f,\"fields\":[",
                   res.n_ocr_regions, res.ocr_confidence);
            for (int i = 0; i < res.n_fields; i++) {
                if (i > 0) printf(",");
                printf("{\"label\":\"%s\",\"value\":\"%s\",\"score\":%.4f,"
                       "\"bbox\":[%.1f,%.1f,%.1f,%.1f]}",
                       json_escape(res.fields[i].label).c_str(),
                       json_escape(res.fields[i].value).c_str(),
                       res.fields[i].score,
                       res.fields[i].x, res.fields[i].y,
                       res.fields[i].w, res.fields[i].h);
            }
            printf("]}\n");
        } else {
            printf("OCR: %d regions, confidence=%.2f\n", res.n_ocr_regions, res.ocr_confidence);
            printf("%d fields extracted:\n", res.n_fields);
            for (int i = 0; i < res.n_fields; i++) {
                printf("  %s = \"%s\"  (score=%.3f, bbox=[%.0f,%.0f,%.0f,%.0f])\n",
                       res.fields[i].label, res.fields[i].value,
                       res.fields[i].score,
                       res.fields[i].x, res.fields[i].y,
                       res.fields[i].w, res.fields[i].h);
            }
        }
        crispembed_kie_free(kctx);
        return 0;
    }

    // LiLT token classification from JSON input
    if (!lilt_path.empty()) {
        void* lctx = crispembed_lilt_init(model_path.c_str(), n_threads);
        if (!lctx) { fprintf(stderr, "error: failed to load LiLT model\n"); return 1; }

        // Read JSON file: {"input_ids": [...], "bbox": [[x0,y0,x1,y1], ...]}
        std::ifstream jf(lilt_path);
        if (!jf.is_open()) { fprintf(stderr, "error: cannot open %s\n", lilt_path.c_str()); crispembed_lilt_free(lctx); return 1; }
        std::string jstr((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());

        // Minimal JSON parsing for input_ids and bbox arrays
        std::vector<int32_t> ids, bbox_flat;
        {
            auto parse_int_array = [](const std::string& s, size_t start) -> std::vector<int32_t> {
                std::vector<int32_t> result;
                auto pos = s.find('[', start);
                if (pos == std::string::npos) return result;
                auto end = s.find(']', pos);
                if (end == std::string::npos) return result;
                std::string arr = s.substr(pos + 1, end - pos - 1);
                std::istringstream iss(arr);
                std::string tok;
                while (std::getline(iss, tok, ',')) {
                    try { result.push_back(std::stoi(tok)); } catch (...) {}
                }
                return result;
            };
            auto id_pos = jstr.find("\"input_ids\"");
            if (id_pos != std::string::npos) ids = parse_int_array(jstr, id_pos);
            auto bb_pos = jstr.find("\"bbox\"");
            if (bb_pos != std::string::npos) {
                // bbox is [[x0,y0,x1,y1], ...] — flatten all nested arrays
                auto outer_start = jstr.find('[', bb_pos);
                auto outer_end = jstr.rfind(']');
                if (outer_start != std::string::npos && outer_end != std::string::npos) {
                    std::string flat = jstr.substr(outer_start, outer_end - outer_start + 1);
                    // Remove all [ and ]
                    for (char& c : flat) if (c == '[' || c == ']') c = ' ';
                    std::istringstream iss(flat);
                    std::string tok;
                    while (std::getline(iss, tok, ',')) {
                        try { bbox_flat.push_back(std::stoi(tok)); } catch (...) {}
                    }
                }
            }
        }

        int T = (int)ids.size();
        if (T == 0 || (int)bbox_flat.size() < T * 4) {
            fprintf(stderr, "error: invalid JSON (need input_ids + bbox)\n");
            crispembed_lilt_free(lctx); return 1;
        }

        int out_n = 0;
        const crispembed_lilt_token* toks = crispembed_lilt_classify(
            lctx, ids.data(), bbox_flat.data(), T, &out_n);

        if (json_output) {
            printf("{\"tokens\":[");
            for (int i = 0; i < out_n; i++) {
                if (i > 0) printf(",");
                printf("{\"token_id\":%d,\"label\":\"%s\",\"score\":%.4f}",
                       toks[i].token_id,
                       json_escape(toks[i].label ? toks[i].label : "").c_str(),
                       toks[i].score);
            }
            printf("]}\n");
        } else {
            printf("%d tokens classified:\n", out_n);
            for (int i = 0; i < out_n; i++) {
                printf("  token=%5d  label=%-15s  score=%.3f\n",
                       toks[i].token_id,
                       toks[i].label ? toks[i].label : "",
                       toks[i].score);
            }
        }
        crispembed_lilt_free(lctx);
        return 0;
    }

    // Unified math OCR (auto-detect architecture from GGUF metadata)
    if (!ocr_path.empty()) {
        void* octx = crispembed_math_ocr_init(model_path.c_str(), n_threads);
        if (!octx) { fprintf(stderr, "error: failed to load OCR model\n"); return 1; }
        int w, h, ch;
        unsigned char* data = stbi_load(ocr_path.c_str(), &w, &h, &ch, 0);
        if (!data) { fprintf(stderr, "error: cannot load %s\n", ocr_path.c_str()); crispembed_math_ocr_free(octx); return 1; }

        // Optional scan cleanup before OCR
        uint8_t * cleaned = nullptr;
        if (cleanup_mode) {
            auto * sctx = scan_cleanup_init(nullptr, n_threads);
            auto sp = scan_cleanup_defaults();
            int cw = 0, ch2 = 0;
            if (scan_cleanup_process(sctx, data, w, h, ch, sp, &cleaned, &cw, &ch2) == 0 && cleaned) {
                stbi_image_free(data);
                data = cleaned;
                w = cw; h = ch2; ch = 3;  // cleanup outputs RGB
            }
            scan_cleanup_free(sctx);
        }

        int out_len = 0;
        const char* latex = crispembed_math_ocr_recognize(octx, data, w, h, ch, &out_len);
        if (cleaned) scan_cleanup_free_image(cleaned); else stbi_image_free(data);
        if (latex && out_len > 0) {
            // Apply punctuation restoration if --punct-model is set
            std::string output_text = latex;
            void * pctx = nullptr;
            if (!punct_model.empty()) {
                std::string pm = crispembed_mgr::resolve_model(punct_model, auto_download, accepted_license);
                pctx = crispembed_punct_init(pm.c_str(), n_threads);
                if (pctx) {
                    const char * punctuated = crispembed_punct_process(pctx, latex);
                    if (punctuated) output_text = punctuated;
                }
            }
            if (json_output) {
                printf("{\"text\":\"%s\"}\n", output_text.c_str());
            } else {
                printf("%s\n", output_text.c_str());
            }
            if (pctx) crispembed_punct_free(pctx);
        } else {
            fprintf(stderr, "error: OCR recognition failed\n");
        }
        crispembed_math_ocr_free(octx);
        return 0;
    }

    // Handwritten math OCR (HMER)
    if (!hmer_path.empty()) {
        hmer_ocr_context* hctx = hmer_ocr_init(model_path.c_str(), n_threads);
        if (!hctx) { fprintf(stderr, "error: failed to load HMER model\n"); return 1; }
        int w, h, ch;
        unsigned char* data = stbi_load(hmer_path.c_str(), &w, &h, &ch, 0);
        if (!data) { fprintf(stderr, "error: cannot load %s\n", hmer_path.c_str()); hmer_ocr_free(hctx); return 1; }
        int out_len = 0;
        const char* latex = hmer_ocr_recognize_raw(hctx, data, w, h, ch, &out_len);
        stbi_image_free(data);
        if (latex && out_len > 0) {
            if (json_output) {
                printf("{\"latex\": \"%s\"}\n", latex);
            } else {
                printf("%s\n", latex);
            }
        } else {
            fprintf(stderr, "error: HMER recognition failed\n");
        }
        hmer_ocr_free(hctx);
        return 0;
    }

    // Handwritten math OCR (BTTR)
    if (!bttr_path.empty()) {
        bttr_ocr_context* bctx = bttr_ocr_init(model_path.c_str(), n_threads);
        if (!bctx) { fprintf(stderr, "error: failed to load BTTR model\n"); return 1; }
        int w, h, ch;
        unsigned char* data = stbi_load(bttr_path.c_str(), &w, &h, &ch, 0);
        if (!data) { fprintf(stderr, "error: cannot load %s\n", bttr_path.c_str()); bttr_ocr_free(bctx); return 1; }
        int out_len = 0;
        const char* latex = bttr_ocr_recognize_raw(bctx, data, w, h, ch, &out_len);
        stbi_image_free(data);
        if (latex && out_len > 0) {
            if (json_output) {
                printf("{\"latex\": \"%s\"}\n", latex);
            } else {
                printf("%s\n", latex);
            }
        } else {
            fprintf(stderr, "error: BTTR recognition failed\n");
        }
        bttr_ocr_free(bctx);
        return 0;
    }

    // Check if this is a CLIP text encoder GGUF.
    if (!texts.empty() && detect_path.empty() && face_path.empty() &&
        image_path.empty() && image_raw_path.empty()) {
        clip_text::context* cltx = nullptr;
        if (clip_text::load(&cltx, model_path.c_str(), n_threads)) {
            for (const auto& t : texts) {
                auto emb = clip_text::encode(cltx, t.c_str());
                if (emb.empty()) { fprintf(stderr, "error: CLIP text encode failed\n"); continue; }
                if (json_output) {
                    printf("{\"text\": \"%s\", \"embedding\": [", json_escape(t).c_str());
                    for (size_t j = 0; j < emb.size(); j++)
                        printf("%.6f%s", emb[j], j + 1 < emb.size() ? ", " : "");
                    printf("]}\n");
                } else {
                    for (size_t j = 0; j < emb.size(); j++)
                        printf("%.6f%s", emb[j], j + 1 < emb.size() ? " " : "\n");
                }
            }
            clip_text::free(cltx);
            return 0;
        }
        // Not a CLIP text model — fall through
    }

    // Check if this is a standalone ViT GGUF (SigLIP/CLIP image encoder).
    // Route to vit_embed for --image-raw, --image, --dim, --list-models.
    if (!image_path.empty() || !image_raw_path.empty() || print_dim) {
        // Try loading as ViT first
        vit_embed::context* vctx = nullptr;
        if (vit_embed::load(&vctx, model_path.c_str(), n_threads)) {
            // It's a ViT model — encode image
            const char* img = !image_path.empty() ? image_path.c_str() : image_raw_path.c_str();

            if (print_dim) {
                printf("%d\n", vit_embed::dim(vctx));
                vit_embed::free(vctx);
                return 0;
            } else if (!image_raw_path.empty()) {
                // Raw preprocessed pixels: float32 [3, H, W]
                int sz = vit_embed::image_size(vctx);
                size_t expected = 3 * sz * sz;
                std::ifstream pf(image_raw_path, std::ios::binary);
                if (!pf) { fprintf(stderr, "error: cannot open %s\n", image_raw_path.c_str()); return 1; }
                std::vector<float> pixels(expected);
                pf.read(reinterpret_cast<char*>(pixels.data()), expected * sizeof(float));
                auto emb = vit_embed::encode(vctx, pixels.data(), sz, sz);
                if (emb.empty()) { fprintf(stderr, "error: ViT encoding failed\n"); return 1; }
                if (print_dim) { printf("%d\n", vit_embed::dim(vctx)); vit_embed::free(vctx); return 0; }
                for (size_t j = 0; j < emb.size(); j++)
                    printf("%.6f%s", emb[j], j + 1 < emb.size() ? " " : "\n");
            } else {
                // Image file (JPG/PNG/BMP) — native resize + normalize
                auto emb = vit_embed::encode_file(vctx, image_path.c_str());
                if (emb.empty()) {
                    fprintf(stderr, "error: ViT image encoding failed for '%s'\n", image_path.c_str());
                    vit_embed::free(vctx);
                    return 1;
                }
                if (json_output) {
                    printf("{\"image\": \"%s\", \"embedding\": [", json_escape(image_path).c_str());
                    for (size_t j = 0; j < emb.size(); j++)
                        printf("%.6f%s", emb[j], j + 1 < emb.size() ? ", " : "");
                    printf("]}\n");
                } else {
                    for (size_t j = 0; j < emb.size(); j++)
                        printf("%.6f%s", emb[j], j + 1 < emb.size() ? " " : "\n");
                }
            }
            vit_embed::free(vctx);
            return 0;
        }
        // Not a ViT model — fall through to text model path
    }

    // Init model
    crispembed_context * ctx = crispembed_init(model_path.c_str(), n_threads);
    if (!ctx) {
        fprintf(stderr, "error: failed to load model '%s'\n", model_path.c_str());
        return 1;
    }

    // Set Matryoshka output dimension if requested
    if (output_dim > 0) {
        crispembed_set_dim(ctx, output_dim);
    }
    if (prefix_set) {
        // Explicit --prefix (even empty "" disables auto-prefix)
        if (!prefix.empty())
            crispembed_set_prefix(ctx, prefix.c_str());
    } else {
        // Auto-apply query prefix if model needs one (GGUF metadata first, name-table fallback)
        const char * auto_pfx = crispembed_ctx_query_prefix(ctx);
        if (!auto_pfx) auto_pfx = crispembed_mgr::get_query_prefix(model_arg.c_str());
        if (auto_pfx) {
            crispembed_set_prefix(ctx, auto_pfx);
            fprintf(stderr, "crispembed: auto-prefix \"%s\" (use --prefix \"\" to disable)\n", auto_pfx);
        }
    }

    // LoRA adapter activation
    if (list_lora) {
        const char ** names = nullptr;
        int count = 0;
        if (crispembed_list_lora(ctx, &names, &count) && count > 0) {
            printf("Available LoRA adapters (%d):\n", count);
            for (int i = 0; i < count; i++) {
                printf("  %s\n", names[i]);
            }
        } else {
            printf("No LoRA adapters available in this model.\n");
        }
        crispembed_free(ctx);
        return 0;
    }
    if (!lora_adapter.empty()) {
        if (!crispembed_set_lora(ctx, lora_adapter.c_str())) {
            fprintf(stderr, "crispembed: failed to set LoRA adapter '%s'\n", lora_adapter.c_str());
            crispembed_free(ctx);
            return 1;
        }
    }

    const auto * hp = crispembed_get_hparams(ctx);
    if (print_dim) {
        printf("%d\n", hp->n_output > 0 ? hp->n_output : hp->n_embd);
        crispembed_free(ctx);
        return 0;
    }
    if (print_capabilities) {
        const int dim = hp->n_output > 0 ? hp->n_output : hp->n_embd;
        if (json_output) {
            printf("{\"dim\": %d, \"prefix\": \"%s\", \"has_sparse\": %s, "
                   "\"has_colbert\": %s, \"is_reranker\": %s}\n",
                   dim,
                   json_escape(crispembed_get_prefix(ctx)).c_str(),
                   crispembed_has_sparse(ctx) ? "true" : "false",
                   crispembed_has_colbert(ctx) ? "true" : "false",
                   crispembed_is_reranker(ctx) ? "true" : "false");
        } else {
            printf("dim=%d prefix=\"%s\" has_sparse=%d has_colbert=%d is_reranker=%d\n",
                   dim,
                   crispembed_get_prefix(ctx),
                   crispembed_has_sparse(ctx),
                   crispembed_has_colbert(ctx),
                   crispembed_is_reranker(ctx));
        }
        crispembed_free(ctx);
        return 0;
    }

    // Image encoding from preprocessed BidirLM/Qwen2VL patch buffers.
    if (!image_raw_path.empty()) {
        if (grid_thw_arg.empty()) {
            fprintf(stderr, "error: --image-raw requires --grid-thw T,H,W\n");
            crispembed_free(ctx);
            return 1;
        }
        int t = 0, h = 0, w = 0;
        if (std::sscanf(grid_thw_arg.c_str(), "%d,%d,%d", &t, &h, &w) != 3 ||
            t <= 0 || h <= 0 || w <= 0) {
            fprintf(stderr, "error: invalid --grid-thw '%s' (expected T,H,W)\n",
                    grid_thw_arg.c_str());
            crispembed_free(ctx);
            return 1;
        }

        FILE * imf = std::fopen(image_raw_path.c_str(), "rb");
        if (!imf) {
            fprintf(stderr, "error: cannot open image patch file '%s'\n", image_raw_path.c_str());
            crispembed_free(ctx);
            return 1;
        }
        std::fseek(imf, 0, SEEK_END);
        long sz = std::ftell(imf);
        std::fseek(imf, 0, SEEK_SET);
        const int patch_dim = 1536;
        if (sz <= 0 || sz % (long)(sizeof(float) * patch_dim) != 0) {
            fprintf(stderr,
                    "error: '%s' is not float32 patch rows of width %d (size=%ld)\n",
                    image_raw_path.c_str(), patch_dim, sz);
            std::fclose(imf);
            crispembed_free(ctx);
            return 1;
        }
        const int n_patches = (int)(sz / (long)(sizeof(float) * patch_dim));
        std::vector<float> patches((size_t)n_patches * patch_dim);
        if (std::fread(patches.data(), sizeof(float), patches.size(), imf) != patches.size()) {
            fprintf(stderr, "error: short read on '%s'\n", image_raw_path.c_str());
            std::fclose(imf);
            crispembed_free(ctx);
            return 1;
        }
        std::fclose(imf);
        if (n_patches != t * h * w) {
            fprintf(stderr,
                    "error: --grid-thw implies %d patches but file has %d rows\n",
                    t * h * w, n_patches);
            crispembed_free(ctx);
            return 1;
        }

        int32_t grid_thw[3] = { t, h, w };
        int dim = 0;
        const float * vec = crispembed_encode_image(ctx, patches.data(), n_patches,
                                                    grid_thw, 1, &dim);
        if (!vec || dim <= 0) {
            fprintf(stderr, "error: image encoding failed (model lacks vision tower?)\n");
            crispembed_free(ctx);
            return 1;
        }
        if (json_output) {
            printf("{\"image\": \"%s\", \"grid_thw\": [%d, %d, %d], \"embedding\": [",
                   json_escape(image_raw_path).c_str(), t, h, w);
            for (int j = 0; j < dim; ++j) {
                printf("%.6f%s", vec[j], j + 1 < dim ? ", " : "");
            }
            printf("]}\n");
        } else {
            for (int j = 0; j < dim; ++j) {
                printf("%.6f%s", vec[j], j + 1 < dim ? " " : "\n");
            }
        }
        crispembed_free(ctx);
        return 0;
    }

    // Image encoding from a JPG/PNG/BMP file via the in-process preprocessor.
    if (!image_path.empty()) {
        int dim = 0;
        const float * vec = crispembed_encode_image_file(ctx, image_path.c_str(), &dim);
        if (!vec || dim <= 0) {
            fprintf(stderr,
                    "error: image encoding failed (model lacks vision tower or "
                    "preprocessor failed) for '%s'\n", image_path.c_str());
            crispembed_free(ctx);
            return 1;
        }
        if (json_output) {
            printf("{\"image\": \"%s\", \"embedding\": [", json_escape(image_path).c_str());
            for (int j = 0; j < dim; j++) {
                printf("%.6f%s", vec[j], j + 1 < dim ? ", " : "");
            }
            printf("]}\n");
        } else {
            for (int j = 0; j < dim; j++) {
                printf("%.6f%s", vec[j], j + 1 < dim ? " " : "\n");
            }
        }
        crispembed_free(ctx);
        return 0;
    }

    // Audio encoding (BidirLM-Omni etc.). Mutually exclusive with text modes;
    // checked here before the texts-required gate.
    if (!audio_path.empty()) {
        FILE * af = std::fopen(audio_path.c_str(), "rb");
        if (!af) {
            fprintf(stderr, "error: cannot open audio file '%s'\n", audio_path.c_str());
            crispembed_free(ctx);
            return 1;
        }
        std::fseek(af, 0, SEEK_END);
        long sz = std::ftell(af);
        std::fseek(af, 0, SEEK_SET);
        if (sz <= 0 || sz % sizeof(float) != 0) {
            fprintf(stderr, "error: '%s' is not f32le PCM (size=%ld)\n",
                    audio_path.c_str(), sz);
            std::fclose(af);
            crispembed_free(ctx);
            return 1;
        }
        std::vector<float> pcm(sz / sizeof(float));
        if (std::fread(pcm.data(), sizeof(float), pcm.size(), af) != pcm.size()) {
            fprintf(stderr, "error: short read on '%s'\n", audio_path.c_str());
            std::fclose(af);
            crispembed_free(ctx);
            return 1;
        }
        std::fclose(af);

        int dim = 0;
        const float * vec = crispembed_encode_audio(ctx, pcm.data(),
                                                    (int)pcm.size(), &dim);
        if (!vec || dim <= 0) {
            fprintf(stderr, "error: audio encoding failed (model lacks audio tower?)\n");
            crispembed_free(ctx);
            return 1;
        }
        if (json_output) {
            printf("{\"audio\": \"%s\", \"embedding\": [", json_escape(audio_path).c_str());
            for (int j = 0; j < dim; ++j) {
                printf("%.6f%s", vec[j], j + 1 < dim ? ", " : "");
            }
            printf("]}\n");
        } else {
            for (int j = 0; j < dim; ++j) {
                printf("%.6f%s", vec[j], j + 1 < dim ? " " : "\n");
            }
        }
        crispembed_free(ctx);
        return 0;
    }

    if (texts.empty()) {
        fprintf(stderr, "error: no texts provided\n");
        crispembed_free(ctx);
        return 1;
    }

    if (sparse_mode) {
        if (!crispembed_has_sparse(ctx)) {
            fprintf(stderr, "error: model does not support sparse retrieval\n");
            crispembed_free(ctx);
            return 1;
        }
        if (json_output) {
            printf("[\n");
        }
        for (size_t i = 0; i < texts.size(); ++i) {
            const int32_t * indices = nullptr;
            const float * values = nullptr;
            const int n = crispembed_encode_sparse(ctx, texts[i].c_str(), &indices, &values);
            if (n <= 0 || !indices || !values) {
                fprintf(stderr, "error: sparse encoding failed for text %zu\n", i);
                continue;
            }

            if (json_output) {
                printf("  {\"text\": \"%s\", \"sparse\": [", json_escape(texts[i]).c_str());
                for (int j = 0; j < n; ++j) {
                    printf("{\"token_id\": %d, \"weight\": %.6f}%s",
                           (int)indices[j], values[j], j + 1 < n ? ", " : "");
                }
                printf("]}%s\n", i + 1 < texts.size() ? "," : "");
            } else {
                printf("%s\n", texts[i].c_str());
                for (int j = 0; j < n; ++j) {
                    printf("  %d %.6f\n", (int)indices[j], values[j]);
                }
            }
        }
        if (json_output) printf("]\n");
        crispembed_free(ctx);
        return 0;
    }

    if (colbert_mode) {
        if (!crispembed_has_colbert(ctx)) {
            fprintf(stderr, "error: model does not support ColBERT multi-vector retrieval\n");
            crispembed_free(ctx);
            return 1;
        }
        if (json_output) {
            printf("[\n");
        }
        for (size_t i = 0; i < texts.size(); ++i) {
            int n_tokens = 0;
            int dim = 0;
            const float * vecs = crispembed_encode_multivec(ctx, texts[i].c_str(), &n_tokens, &dim);
            if (!vecs || n_tokens <= 0 || dim <= 0) {
                fprintf(stderr, "error: colbert encoding failed for text %zu\n", i);
                continue;
            }

            if (json_output) {
                printf("  {\"text\": \"%s\", \"n_tokens\": %d, \"dim\": %d, \"vectors\": [",
                       json_escape(texts[i]).c_str(), n_tokens, dim);
                for (int t = 0; t < n_tokens; ++t) {
                    if (t > 0) printf(", ");
                    printf("[");
                    for (int d = 0; d < dim; ++d) {
                        printf("%.6f%s", vecs[t * dim + d], d + 1 < dim ? ", " : "");
                    }
                    printf("]");
                }
                printf("]}%s\n", i + 1 < texts.size() ? "," : "");
            } else {
                printf("%s\n", texts[i].c_str());
                for (int t = 0; t < n_tokens; ++t) {
                    printf("  token %d:", t);
                    for (int d = 0; d < dim; ++d) {
                        printf(" %.6f", vecs[t * dim + d]);
                    }
                    printf("\n");
                }
            }
        }
        if (json_output) printf("]\n");
        crispembed_free(ctx);
        return 0;
    }

    if (!rerank_query.empty()) {
        if (!crispembed_is_reranker(ctx)) {
            fprintf(stderr, "error: model is not a cross-encoder reranker\n");
            crispembed_free(ctx);
            return 1;
        }

        std::vector<std::pair<size_t, float>> ranked;
        ranked.reserve(texts.size());
        for (size_t i = 0; i < texts.size(); ++i) {
            const float score = crispembed_rerank(ctx, rerank_query.c_str(), texts[i].c_str());
            if (!std::isfinite(score)) {
                fprintf(stderr, "error: rerank failed for document %zu\n", i);
                continue;
            }
            ranked.emplace_back(i, score);
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
            return a.second > b.second;
        });
        if (top_n > 0 && (int)ranked.size() > top_n) {
            ranked.resize(top_n);
        }

        if (json_output) {
            printf("{\"query\": \"%s\", \"results\": [", json_escape(rerank_query).c_str());
            for (size_t i = 0; i < ranked.size(); ++i) {
                const auto & item = ranked[i];
                printf("%s{\"index\": %zu, \"score\": %.6f, \"document\": \"%s\"}",
                       i > 0 ? ", " : "",
                       item.first,
                       item.second,
                       json_escape(texts[item.first]).c_str());
            }
            printf("]}\n");
        } else {
            for (const auto & item : ranked) {
                printf("[%zu] %.6f %s\n", item.first, item.second, texts[item.first].c_str());
            }
        }

        crispembed_free(ctx);
        return 0;
    }

    if (!biencoder_query.empty()) {
        std::vector<const char *> batch_ptrs;
        batch_ptrs.reserve(texts.size() + 1);
        batch_ptrs.push_back(biencoder_query.c_str());
        for (const auto & text : texts) {
            batch_ptrs.push_back(text.c_str());
        }

        int dim = 0;
        const float * vecs = crispembed_encode_batch(ctx, batch_ptrs.data(), (int)batch_ptrs.size(), &dim);
        if (!vecs || dim <= 0) {
            fprintf(stderr, "error: bi-encoder batch encoding failed\n");
            crispembed_free(ctx);
            return 1;
        }

        const float * query_vec = vecs;
        std::vector<std::pair<size_t, float>> ranked;
        ranked.reserve(texts.size());
        for (size_t i = 0; i < texts.size(); ++i) {
            const float * doc_vec = vecs + (i + 1) * dim;
            ranked.emplace_back(i, dot_product(query_vec, doc_vec, dim));
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
            return a.second > b.second;
        });
        if (top_n > 0 && (int)ranked.size() > top_n) {
            ranked.resize(top_n);
        }

        if (json_output) {
            printf("{\"query\": \"%s\", \"results\": [", json_escape(biencoder_query).c_str());
            for (size_t i = 0; i < ranked.size(); ++i) {
                const auto & item = ranked[i];
                printf("%s{\"index\": %zu, \"score\": %.6f, \"document\": \"%s\"}",
                       i > 0 ? ", " : "",
                       item.first,
                       item.second,
                       json_escape(texts[item.first]).c_str());
            }
            printf("]}\n");
        } else {
            for (const auto & item : ranked) {
                printf("[%zu] %.6f %s\n", item.first, item.second, texts[item.first].c_str());
            }
        }

        crispembed_free(ctx);
        return 0;
    }

    // Dense encode
    if (json_output) printf("[\n");
    for (size_t i = 0; i < texts.size(); i++) {
        int dim = 0;
        const float * vec = crispembed_encode(ctx, texts[i].c_str(), &dim);
        if (!vec) {
            fprintf(stderr, "error: encoding failed for text %zu\n", i);
            continue;
        }

        if (json_output) {
            printf("  {\"text\": \"%s\", \"embedding\": [", json_escape(texts[i]).c_str());
            for (int d = 0; d < dim; d++) {
                printf("%.6f%s", vec[d], d + 1 < dim ? ", " : "");
            }
            printf("]}%s\n", i + 1 < texts.size() ? "," : "");
        } else {
            // Plain format: one line per text, space-separated floats
            for (int d = 0; d < dim; d++) {
                printf("%.6f%s", vec[d], d + 1 < dim ? " " : "");
            }
            printf("\n");
        }
    }
    if (json_output) printf("]\n");

    crispembed_free(ctx);
    return 0;
}
