// crispembed server — HTTP API for text embedding.
//
// Usage: crispembed --server -m model.gguf [--port 8080]
//
// Endpoints:
//   POST /embed           — {"texts": ["hello", "world"]} → {"embeddings": [[...], [...]]}
//   POST /v1/embeddings   — OpenAI-compatible
//   POST /api/embed       — Ollama-compatible (batch)
//   POST /api/embeddings  — Ollama-compatible (single, legacy)
//   GET  /health          — server status

#include "crispembed.h"
#include "model_mgr.h"
#include "httplib.h"

#include <atomic>
#include <chrono>
#include <cstdio>
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
    int port = 8080;
    int n_threads = 4;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) host = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--det") == 0 && i + 1 < argc) det_model_path = argv[++i];
        else if (strcmp(argv[i], "--rec") == 0 && i + 1 < argc) rec_model_path = argv[++i];
    }

    if (model_path.empty() && det_model_path.empty()) {
        fprintf(stderr, "Usage: crispembed-server -m MODEL [--port 8080] [--host 127.0.0.1]\n");
        fprintf(stderr, "  MODEL can be a .gguf path or a model name (auto-downloads from HuggingFace)\n");
        fprintf(stderr, "  Examples: -m all-MiniLM-L6-v2   -m octen-0.6b   -m model.gguf\n");
        fprintf(stderr, "\nFace pipeline:\n");
        fprintf(stderr, "  --det MODEL   face detection model (SCRFD GGUF)\n");
        fprintf(stderr, "  --rec MODEL   face recognition model (ArcFace/SFace GGUF)\n");
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
    fprintf(stderr, "  GET  /health\n\n");

    svr.listen(host, port);

    if (face_det) crispembed_face_free(face_det);
    if (face_rec) crispembed_face_free(face_rec);
    if (ctx) crispembed_free(ctx);
    return 0;
}
