// crispembed CLI — encode text to embedding vector.
//
// Usage:
//   crispembed -m model.gguf "query: hello world"
//   crispembed -m octen-0.6b "hello world"        # auto-download
//   crispembed -m model.gguf -f texts.txt          (one text per line)
//   crispembed --list-models

#include "crispembed.h"
#include "model_mgr.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s -m MODEL [options] [TEXT ...]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m MODEL         path to GGUF model or model name (auto-download)\n");
    fprintf(stderr, "  -f FILE          read texts from file (one per line)\n");
    fprintf(stderr, "  -t N             number of threads (default: 4)\n");
    fprintf(stderr, "  --json           output as JSON array\n");
    fprintf(stderr, "  --dim            print embedding dimension and exit\n");
    fprintf(stderr, "  --auto-download  download model automatically if not found\n");
    fprintf(stderr, "  --list-models    list available models\n");
    fprintf(stderr, "  --cache-dir DIR  set model cache directory\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Model names (auto-download from HuggingFace):\n");
    fprintf(stderr, "  all-MiniLM-L6-v2, gte-small, arctic-embed-xs,\n");
    fprintf(stderr, "  multilingual-e5-small, pixie-rune-v1, arctic-embed-l-v2,\n");
    fprintf(stderr, "  octen-0.6b, f2llm-v2-0.6b, jina-v5-nano, jina-v5-small,\n");
    fprintf(stderr, "  harrier-0.6b, harrier-270m, qwen3-embed-0.6b\n");
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
    std::string model_arg;
    std::string file_path;
    std::vector<std::string> texts;
    int n_threads = 4;
    bool json_output = false;
    bool print_dim = false;
    bool auto_download = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_arg = argv[++i];
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            file_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0) {
            json_output = true;
        } else if (strcmp(argv[i], "--dim") == 0) {
            print_dim = true;
        } else if (strcmp(argv[i], "--auto-download") == 0) {
            auto_download = true;
        } else if (strcmp(argv[i], "--list-models") == 0) {
            crispembed_mgr::list_models();
            return 0;
        } else if (strcmp(argv[i], "--cache-dir") == 0 && i + 1 < argc) {
            setenv("CRISPEMBED_CACHE_DIR", argv[++i], 1);
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

    // Resolve model path (handles auto-download)
    std::string model_path = crispembed_mgr::resolve_model(model_arg, auto_download);
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

    // Init model
    crispembed_context * ctx = crispembed_init(model_path.c_str(), n_threads);
    if (!ctx) {
        fprintf(stderr, "error: failed to load model '%s'\n", model_path.c_str());
        return 1;
    }

    const auto * hp = crispembed_get_hparams(ctx);
    if (print_dim) {
        printf("%d\n", hp->n_output > 0 ? hp->n_output : hp->n_embd);
        crispembed_free(ctx);
        return 0;
    }

    if (texts.empty()) {
        fprintf(stderr, "error: no texts to encode\n");
        crispembed_free(ctx);
        return 1;
    }

    // Encode
    if (json_output) printf("[\n");
    for (size_t i = 0; i < texts.size(); i++) {
        int dim = 0;
        const float * vec = crispembed_encode(ctx, texts[i].c_str(), &dim);
        if (!vec) {
            fprintf(stderr, "error: encoding failed for text %zu\n", i);
            continue;
        }

        if (json_output) {
            printf("  {\"text\": \"%s\", \"embedding\": [", texts[i].c_str());
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
