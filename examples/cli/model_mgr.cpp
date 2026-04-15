// model_mgr.cpp — Auto-download model manager for CrispEmbed.

#include "model_mgr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define mkdir(p, m) _mkdir(p)
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

namespace crispembed_mgr {

struct ModelEntry {
    const char * name;
    const char * filename;
    const char * url;
    const char * desc;
    const char * approx_size;
};

static const ModelEntry k_registry[] = {
    {"all-MiniLM-L6-v2",
     "all-MiniLM-L6-v2.gguf",
     "https://huggingface.co/cstr/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.gguf",
     "BERT 384d English (22M)", "87 MB"},

    {"gte-small",
     "gte-small.gguf",
     "https://huggingface.co/cstr/gte-small-GGUF/resolve/main/gte-small.gguf",
     "BERT 384d English (33M)", "128 MB"},

    {"arctic-embed-xs",
     "arctic-embed-xs.gguf",
     "https://huggingface.co/cstr/arctic-embed-xs-GGUF/resolve/main/arctic-embed-xs.gguf",
     "BERT 384d CLS English (22M)", "87 MB"},

    {"multilingual-e5-small",
     "multilingual-e5-small.gguf",
     "https://huggingface.co/cstr/multilingual-e5-small-GGUF/resolve/main/multilingual-e5-small.gguf",
     "XLM-R 384d multilingual (118M)", "454 MB"},

    {"pixie-rune-v1",
     "pixie-rune-v1.gguf",
     "https://huggingface.co/cstr/pixie-rune-v1-GGUF/resolve/main/pixie-rune-v1.gguf",
     "XLM-R 1024d 74-lang CLS (560M)", "2.2 GB"},

    {"arctic-embed-l-v2",
     "arctic-embed-l-v2.gguf",
     "https://huggingface.co/cstr/arctic-embed-l-v2-GGUF/resolve/main/arctic-embed-l-v2.gguf",
     "XLM-R 1024d CLS English (560M)", "2.2 GB"},

    {"octen-0.6b",
     "octen-0.6b-q8_0.gguf",
     "https://huggingface.co/cstr/octen-0.6b-GGUF/resolve/main/octen-0.6b-q8_0.gguf",
     "Qwen3 1024d multilingual (600M)", "609 MB"},

    {"f2llm-v2-0.6b",
     "f2llm-v2-0.6b-q8_0.gguf",
     "https://huggingface.co/cstr/f2llm-v2-0.6b-GGUF/resolve/main/f2llm-v2-0.6b-q8_0.gguf",
     "Qwen3 1024d multilingual (600M)", "609 MB"},

    {"jina-v5-nano",
     "jina-v5-nano-q8_0.gguf",
     "https://huggingface.co/cstr/jina-v5-nano-GGUF/resolve/main/jina-v5-nano-q8_0.gguf",
     "Qwen3 1024d compact (210M)", "222 MB"},

    {"jina-v5-small",
     "jina-v5-small-q8_0.gguf",
     "https://huggingface.co/cstr/jina-v5-small-GGUF/resolve/main/jina-v5-small-q8_0.gguf",
     "Qwen3 1024d multilingual (600M)", "609 MB"},

    {"harrier-0.6b",
     "harrier-0.6b-q8_0.gguf",
     "https://huggingface.co/cstr/harrier-0.6b-GGUF/resolve/main/harrier-0.6b-q8_0.gguf",
     "Qwen3 1024d SOTA (600M)", "609 MB"},

    {"harrier-270m",
     "harrier-270m-q8_0.gguf",
     "https://huggingface.co/cstr/harrier-270m-GGUF/resolve/main/harrier-270m-q8_0.gguf",
     "Gemma3 640d compact (270M)", "755 MB"},

    {"qwen3-embed-0.6b",
     "qwen3-embed-0.6b-q8_0.gguf",
     "https://huggingface.co/cstr/qwen3-embed-0.6b-GGUF/resolve/main/qwen3-embed-0.6b-q8_0.gguf",
     "Qwen3 1024d official (600M)", "1.0 GB"},

    {nullptr, nullptr, nullptr, nullptr, nullptr}
};

std::string cache_dir() {
    // Check env override
    const char * env = std::getenv("CRISPEMBED_CACHE_DIR");
    if (env && env[0]) return env;

    // Default: ~/.cache/crispembed
    std::string home;
#ifdef _WIN32
    const char * h = std::getenv("USERPROFILE");
    if (!h) h = std::getenv("HOME");
    if (h) home = h;
#else
    const char * h = std::getenv("HOME");
    if (h) home = h;
#endif
    if (home.empty()) home = "/tmp";
    return home + "/.cache/crispembed";
}

static bool file_exists(const std::string & path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && st.st_size > 0;
}

static void mkdirs(const std::string & path) {
    // Simple recursive mkdir
    for (size_t i = 1; i < path.size(); i++) {
        if (path[i] == '/' || path[i] == '\\') {
            std::string sub = path.substr(0, i);
            mkdir(sub.c_str(), 0755);
        }
    }
    mkdir(path.c_str(), 0755);
}

static bool download_file(const std::string & url, const std::string & dest) {
    // Try curl first, then wget
    std::string tmp = dest + ".tmp";

    // curl
    std::string cmd = "curl -fL --progress-bar -o '" + tmp + "' '" + url + "'";
    int ret = system(cmd.c_str());
    if (ret == 0 && file_exists(tmp)) {
        rename(tmp.c_str(), dest.c_str());
        return true;
    }

    // wget fallback
    cmd = "wget -q --show-progress -O '" + tmp + "' '" + url + "'";
    ret = system(cmd.c_str());
    if (ret == 0 && file_exists(tmp)) {
        rename(tmp.c_str(), dest.c_str());
        return true;
    }

    // Cleanup
    remove(tmp.c_str());
    return false;
}

std::string resolve_model(const std::string & arg, bool auto_download) {
    // If it's already a file path, use it directly
    if (file_exists(arg)) return arg;

    // Look up in registry
    const ModelEntry * entry = nullptr;
    for (const ModelEntry * e = k_registry; e->name; e++) {
        if (arg == e->name || arg == e->filename) {
            entry = e;
            break;
        }
    }

    // Fuzzy match: check if arg is a substring of any model name
    if (!entry) {
        for (const ModelEntry * e = k_registry; e->name; e++) {
            if (strstr(e->name, arg.c_str()) || strstr(e->filename, arg.c_str())) {
                entry = e;
                break;
            }
        }
    }

    if (!entry) {
        fprintf(stderr, "Unknown model: '%s'\n", arg.c_str());
        fprintf(stderr, "Use --list-models to see available models.\n");
        return "";
    }

    // Check cache
    std::string dir = cache_dir();
    std::string cached = dir + "/" + entry->filename;

    if (file_exists(cached)) {
        return cached;
    }

    // Download
    if (!auto_download) {
        // Check if TTY for interactive prompt
        bool is_tty = isatty(fileno(stdin));
        if (is_tty) {
            fprintf(stderr, "Model '%s' not found locally.\n", entry->name);
            fprintf(stderr, "Download %s (%s) from HuggingFace? [y/N] ",
                    entry->filename, entry->approx_size);
            char c = 0;
            if (scanf(" %c", &c) != 1 || (c != 'y' && c != 'Y')) {
                return "";
            }
        } else {
            fprintf(stderr, "Model '%s' not found. Use --auto-download to download automatically.\n",
                    entry->name);
            return "";
        }
    }

    mkdirs(dir);
    fprintf(stderr, "Downloading %s (%s)...\n", entry->filename, entry->approx_size);
    if (download_file(entry->url, cached)) {
        fprintf(stderr, "Downloaded to %s\n", cached.c_str());
        return cached;
    } else {
        fprintf(stderr, "Download failed.\n");
        return "";
    }
}

void list_models() {
    fprintf(stderr, "Available models:\n");
    fprintf(stderr, "  %-25s %-40s %s\n", "Name", "Description", "Size");
    fprintf(stderr, "  %-25s %-40s %s\n",
            "----", "-----------", "----");
    for (const ModelEntry * e = k_registry; e->name; e++) {
        // Check if cached
        std::string cached = cache_dir() + "/" + e->filename;
        const char * status = file_exists(cached) ? " [cached]" : "";
        fprintf(stderr, "  %-25s %-40s %s%s\n",
                e->name, e->desc, e->approx_size, status);
    }
}

}  // namespace crispembed_mgr
