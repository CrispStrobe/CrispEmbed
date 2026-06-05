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
    fprintf(stderr, "  --hmer FILE      handwritten math OCR → LaTeX (HMER model)\n");
    fprintf(stderr, "  --det MODEL      detection model for --face-pipeline\n");
    fprintf(stderr, "  --face-pipeline  detect+align+encode faces (needs -m rec_model --det det_model)\n");
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
    std::string hmer_path;   // image for handwritten math OCR
    std::string det_model;   // detection model for --face-pipeline
    bool face_pipeline_mode = false;
    float conf_threshold = 0.5f;

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
        } else if (strcmp(argv[i], "--hmer") == 0 && i + 1 < argc) {
            hmer_path = argv[++i];
        } else if (strcmp(argv[i], "--det") == 0 && i + 1 < argc) {
            det_model = argv[++i];
        } else if (strcmp(argv[i], "--face-pipeline") == 0) {
            face_pipeline_mode = true;
        } else if (strcmp(argv[i], "--conf") == 0 && i + 1 < argc) {
            conf_threshold = (float)atof(argv[++i]);
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

    if (model_arg.empty()) {
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

    // Resolve model path (handles auto-download)
    // Auto-download if arg looks like a model name (not a file path)
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
