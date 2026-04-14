// crispembed server — HTTP API for text embedding.
//
// Usage: crispembed --server -m model.gguf [--port 8080]
//
// Endpoints:
//   POST /embed     — {"texts": ["hello", "world"]} → {"embeddings": [[...], [...]]}
//   POST /v1/embeddings — OpenAI-compatible
//   GET  /health    — server status

#include "crispembed.h"
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
    int port = 8080;
    int n_threads = 4;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) host = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
    }

    if (model_path.empty()) {
        fprintf(stderr, "Usage: crispembed-server -m model.gguf [--port 8080] [--host 127.0.0.1]\n");
        return 1;
    }

    crispembed_context * ctx = crispembed_init(model_path.c_str(), n_threads);
    if (!ctx) {
        fprintf(stderr, "Failed to load model '%s'\n", model_path.c_str());
        return 1;
    }

    const auto * hp = crispembed_get_hparams(ctx);
    int dim = hp->n_output > 0 ? hp->n_output : hp->n_embd;
    std::mutex model_mutex;

    httplib::Server svr;

    // POST /embed — simple API
    svr.Post("/embed", [&](const httplib::Request & req, httplib::Response & res) {
        // Parse JSON manually (minimal, no deps)
        // Expect: {"texts": ["hello", "world"]} or {"text": "hello"}
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
        for (size_t i = 0; i < texts.size(); i++) {
            int d = 0;
            const float * vec = crispembed_encode(ctx, texts[i].c_str(), &d);
            if (i > 0) js << ", ";
            js << "[";
            for (int j = 0; j < d; j++) {
                if (j > 0) js << ", ";
                js << vec[j];
            }
            js << "]";
        }
        js << "], \"dim\": " << dim << "}";

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "crispembed-server: encoded %zu text(s) in %.1f ms\n", texts.size(), ms);

        res.set_content(js.str(), "application/json");
    });

    // POST /v1/embeddings — OpenAI-compatible
    svr.Post("/v1/embeddings", [&](const httplib::Request & req, httplib::Response & res) {
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
        js << "{\"object\": \"list\", \"data\": [";
        for (size_t i = 0; i < texts.size(); i++) {
            int d = 0;
            const float * vec = crispembed_encode(ctx, texts[i].c_str(), &d);
            if (i > 0) js << ", ";
            js << "{\"object\": \"embedding\", \"index\": " << i << ", \"embedding\": [";
            for (int j = 0; j < d; j++) {
                if (j > 0) js << ", ";
                js << vec[j];
            }
            js << "]}";
        }
        js << "], \"model\": \"crispembed\", \"usage\": {\"prompt_tokens\": 0, \"total_tokens\": 0}}";
        res.set_content(js.str(), "application/json");
    });

    // GET /health
    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        std::ostringstream js;
        js << "{\"status\": \"ok\", \"dim\": " << dim
           << ", \"layers\": " << hp->n_layer
           << ", \"vocab\": " << hp->n_vocab << "}";
        res.set_content(js.str(), "application/json");
    });

    fprintf(stderr, "\ncrispembed-server: listening on %s:%d\n", host.c_str(), port);
    fprintf(stderr, "  POST /embed          — {\"texts\": [\"hello\"]}\n");
    fprintf(stderr, "  POST /v1/embeddings  — OpenAI-compatible\n");
    fprintf(stderr, "  GET  /health\n\n");

    svr.listen(host, port);

    crispembed_free(ctx);
    return 0;
}
