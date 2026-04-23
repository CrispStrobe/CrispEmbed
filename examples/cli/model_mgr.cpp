// model_mgr.cpp — Auto-download model manager for CrispEmbed.

#include "model_mgr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <cctype>
#include <string>
#include <sys/stat.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

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

namespace {

bool download_supported() {
#if defined(__APPLE__) && defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
    return false;
#else
    return true;
#endif
}

}  // namespace

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

    // --- RAG-critical models (Phase 3) ---

    {"bge-small-en-v1.5",
     "bge-small-en-v1.5.gguf",
     "https://huggingface.co/cstr/bge-small-en-v1.5-GGUF/resolve/main/bge-small-en-v1.5.gguf",
     "BERT 384d English (33M)", "128 MB"},

    {"bge-base-en-v1.5",
     "bge-base-en-v1.5.gguf",
     "https://huggingface.co/cstr/bge-base-en-v1.5-GGUF/resolve/main/bge-base-en-v1.5.gguf",
     "BERT 768d English (109M)", "418 MB"},

    {"bge-large-en-v1.5",
     "bge-large-en-v1.5.gguf",
     "https://huggingface.co/cstr/bge-large-en-v1.5-GGUF/resolve/main/bge-large-en-v1.5.gguf",
     "BERT 1024d English (335M)", "1.3 GB"},

    {"nomic-embed-text-v1.5",
     "nomic-embed-text-v1.5.gguf",
     "https://huggingface.co/cstr/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.gguf",
     "BERT 768d 8K context Matryoshka (137M)", "523 MB"},

    {"all-MiniLM-L12-v2",
     "all-MiniLM-L12-v2.gguf",
     "https://huggingface.co/cstr/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2.gguf",
     "BERT 384d English (33M)", "128 MB"},

    {"all-mpnet-base-v2",
     "all-mpnet-base-v2.gguf",
     "https://huggingface.co/cstr/all-mpnet-base-v2-GGUF/resolve/main/all-mpnet-base-v2.gguf",
     "BERT 768d English (109M)", "418 MB"},

    {"mxbai-embed-large-v1",
     "mxbai-embed-large-v1.gguf",
     "https://huggingface.co/cstr/mxbai-embed-large-v1-GGUF/resolve/main/mxbai-embed-large-v1.gguf",
     "BERT 1024d English (335M)", "1.3 GB"},

    {"snowflake-arctic-embed-m",
     "snowflake-arctic-embed-m.gguf",
     "https://huggingface.co/cstr/snowflake-arctic-embed-m-GGUF/resolve/main/snowflake-arctic-embed-m.gguf",
     "BERT 768d CLS English (109M)", "418 MB"},

    {"snowflake-arctic-embed-l",
     "snowflake-arctic-embed-l.gguf",
     "https://huggingface.co/cstr/snowflake-arctic-embed-l-GGUF/resolve/main/snowflake-arctic-embed-l.gguf",
     "XLM-R 1024d CLS English (335M)", "1.3 GB"},

    {"bge-m3",
     "bge-m3-q4_k.gguf",
     "https://huggingface.co/cstr/bge-m3-GGUF/resolve/main/bge-m3-q4_k.gguf",
     "XLM-R 1024d dense+sparse+ColBERT multilingual (568M)", "438 MB"},

    // --- Reranker models (Phase 4) ---

    {"bge-reranker-v2-m3",
     "bge-reranker-v2-m3.gguf",
     "https://huggingface.co/cstr/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3.gguf",
     "XLM-R reranker multilingual (568M)", "2.2 GB"},

    {"bge-reranker-base",
     "bge-reranker-base.gguf",
     "https://huggingface.co/cstr/bge-reranker-base-GGUF/resolve/main/bge-reranker-base.gguf",
     "BERT reranker EN+ZH (278M)", "1.1 GB"},

    {"ms-marco-MiniLM-L-6-v2",
     "ms-marco-MiniLM-L-6-v2.gguf",
     "https://huggingface.co/cstr/ms-marco-MiniLM-L-6-v2-GGUF/resolve/main/ms-marco-MiniLM-L-6-v2.gguf",
     "BERT reranker English fast (22M)", "87 MB"},

    {"ms-marco-MiniLM-L-12-v2",
     "ms-marco-MiniLM-L-12-v2.gguf",
     "https://huggingface.co/cstr/ms-marco-MiniLM-L-12-v2-GGUF/resolve/main/ms-marco-MiniLM-L-12-v2.gguf",
     "BERT reranker English (33M)", "128 MB"},

    {"jina-reranker-v2-base-multilingual",
     "jina-reranker-v2-base-multilingual.gguf",
     "https://huggingface.co/cstr/jina-reranker-v2-base-multilingual-GGUF/resolve/main/jina-reranker-v2-base-multilingual.gguf",
     "XLM-R reranker multilingual (278M)", "1.1 GB"},

    {"mxbai-rerank-xsmall-v1",
     "mxbai-rerank-xsmall-v1.gguf",
     "https://huggingface.co/cstr/mxbai-rerank-xsmall-v1-GGUF/resolve/main/mxbai-rerank-xsmall-v1.gguf",
     "BERT reranker English fast (33M)", "128 MB"},

    {"mxbai-rerank-base-v1",
     "mxbai-rerank-base-v1.gguf",
     "https://huggingface.co/cstr/mxbai-rerank-base-v1-GGUF/resolve/main/mxbai-rerank-base-v1.gguf",
     "BERT reranker English (86M)", "330 MB"},

    // --- MTEB top multilingual models ---

    {"multilingual-e5-base",
     "multilingual-e5-base.gguf",
     "https://huggingface.co/cstr/multilingual-e5-base-GGUF/resolve/main/multilingual-e5-base.gguf",
     "XLM-R 768d 100+ languages (278M)", "1.1 GB"},

    {"multilingual-e5-large",
     "multilingual-e5-large.gguf",
     "https://huggingface.co/cstr/multilingual-e5-large-GGUF/resolve/main/multilingual-e5-large.gguf",
     "XLM-R 1024d 100+ languages (560M)", "2.2 GB"},

    {"granite-embedding-278m",
     "granite-embedding-278m-multilingual.gguf",
     "https://huggingface.co/cstr/granite-embedding-278m-multilingual-GGUF/resolve/main/granite-embedding-278m-multilingual.gguf",
     "XLM-R 768d IBM multilingual (278M)", "1.1 GB"},

    {"granite-embedding-107m",
     "granite-embedding-107m-multilingual.gguf",
     "https://huggingface.co/cstr/granite-embedding-107m-multilingual-GGUF/resolve/main/granite-embedding-107m-multilingual.gguf",
     "XLM-R 384d IBM multilingual (107M)", "418 MB"},

    // --- Sparse models ---

    {"splade-pp-en-v1",
     "splade-pp-en-v1.gguf",
     "https://huggingface.co/cstr/splade-pp-en-v1-GGUF/resolve/main/splade-pp-en-v1.gguf",
     "BERT sparse (SPLADE) English (109M)", "418 MB"},

    // --- GTE v1.5 (new BERT) ---

    {"gte-base-en-v1.5",
     "gte-base-en-v1.5.gguf",
     "https://huggingface.co/cstr/gte-base-en-v1.5-GGUF/resolve/main/gte-base-en-v1.5.gguf",
     "GTE 768d English pre-LN+RoPE+GeGLU (109M)", "522 MB"},

    {"gte-large-en-v1.5",
     "gte-large-en-v1.5.gguf",
     "https://huggingface.co/cstr/gte-large-en-v1.5-GGUF/resolve/main/gte-large-en-v1.5.gguf",
     "GTE 1024d English pre-LN+RoPE+GeGLU (335M)", "1.7 GB"},

    {"embeddinggemma-300m",
     "embeddinggemma-300m.gguf",
     "https://huggingface.co/cstr/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300m.gguf",
     "Gemma3 768d 24-layer last-token (300M)", "741 MB"},

    {nullptr, nullptr, nullptr, nullptr, nullptr}
};

std::string cache_dir() {
    // Check env override
    const char * env = std::getenv("CRISPEMBED_CACHE_DIR");
    if (env && env[0]) {
        std::string value = env;
        size_t start = 0;
        while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
            start++;
        }
        size_t end = value.size();
        while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
            end--;
        }
        value = value.substr(start, end - start);
        if (!value.empty()) return value;
    }

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
    // Use _stat64 on Windows: regular stat() has 32-bit st_size which overflows for files > 2 GB
#ifdef _WIN32
    struct __stat64 st;
    return _stat64(path.c_str(), &st) == 0 && st.st_size > 0;
#else
    struct stat st;
    return stat(path.c_str(), &st) == 0 && st.st_size > 0;
#endif
}

static void mkdirs(const std::string & path) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path), ec);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
static bool download_file(const std::string & source_url, const std::string & dest_path) {
#if defined(__APPLE__) && defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
    (void)source_url;
    (void)dest_path;
    return false;
#else
    std::string tmp = dest_path + ".tmp";

    // Use double quotes for Windows compatibility
#ifdef _WIN32
    // Try curl.exe (bundled with Windows 10+)
    std::string cmd = "curl.exe -fL --progress-bar -o \"" + tmp + "\" \"" + source_url + "\"";
#else
    std::string cmd = "curl -fL --progress-bar -o \"" + tmp + "\" \"" + source_url + "\"";
#endif
    // NOLINTNEXTLINE(bugprone-command-processor)
    int ret = system(cmd.c_str());
    if (ret == 0 && file_exists(tmp)) {
        rename(tmp.c_str(), dest_path.c_str());
        return true;
    }

#ifndef _WIN32
    // wget fallback (Linux/macOS only)
    cmd = "wget -q --show-progress -O \"" + tmp + "\" \"" + source_url + "\"";
    // NOLINTNEXTLINE(bugprone-command-processor)
    ret = system(cmd.c_str());
    if (ret == 0 && file_exists(tmp)) {
        rename(tmp.c_str(), dest_path.c_str());
        return true;
    }
#endif

    remove(tmp.c_str());
    return false;
#endif
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

    if (!download_supported()) {
        fprintf(stderr,
                "Model '%s' is not cached, and auto-download is unavailable on iOS builds.\n",
                entry->name);
        return "";
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

int n_models() {
    int n = 0;
    for (const ModelEntry * e = k_registry; e->name; e++) n++;
    return n;
}

const char * model_name(int i) {
    int n = 0;
    for (const ModelEntry * e = k_registry; e->name; e++, n++)
        if (n == i) return e->name;
    return nullptr;
}

const char * model_desc(int i) {
    int n = 0;
    for (const ModelEntry * e = k_registry; e->name; e++, n++)
        if (n == i) return e->desc;
    return nullptr;
}

const char * model_filename(int i) {
    int n = 0;
    for (const ModelEntry * e = k_registry; e->name; e++, n++)
        if (n == i) return e->filename;
    return nullptr;
}

const char * model_size(int i) {
    int n = 0;
    for (const ModelEntry * e = k_registry; e->name; e++, n++)
        if (n == i) return e->approx_size;
    return nullptr;
}

}  // namespace crispembed_mgr
