#include "inpaint_cli.h"

#include "core/env_gate.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/image_out.h"
#include "inpaint.h"
#include "lama.h"
#include "latent_inpaint.h"
#include "ocr_detect.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "../../ggml/examples/stb_image.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace inpaint_cli {
namespace {

namespace fs = std::filesystem;

struct path_less {
    bool operator()(const fs::path & left, const fs::path & right) const {
#ifdef _WIN32
        // Default Windows volumes alias names that differ only in case, even
        // before either output exists. Compare normalized UTF-16 paths using
        // Windows ordinal folding; POSIX keeps its case-sensitive ordering.
        const int order = CompareStringOrdinal(left.c_str(), -1, right.c_str(), -1, TRUE);
        if (order == 0)
            throw std::system_error(int(GetLastError()), std::system_category(), "cannot compare Windows paths");
        return order == CSTR_LESS_THAN;
#else
        return left < right;
#endif
    }
};

struct options {
    std::string model, image, mask, output, list, directory, mask_directory, output_directory, text_model;
    std::string engine = "migan256", vae, model_sha256, vae_sha256, checkpoint_sha256;
    std::string backend;
    int threads = 1;
    int steps = 20;
    float guidance = 2.0f, noise_offset = 0.0357f;
    uint64_t seed = 0;
    bool tiled = false;
    int dilation = -1; // 3 for text masks; 0 for supplied masks.
    bool from_text = false;
    bool overwrite = false;
    bool has_region = false;
    std::array<int, 4> region{ 0, 0, 0, 0 };
};

struct frame {
    fs::path image, mask, output;
    int width = 0, height = 0;
};

struct context_deleter {
    void operator()(inpaint::context * ctx) const { inpaint::free(ctx); }
};
struct latent_context_deleter {
    void operator()(latent_inpaint::context * ctx) const { latent_inpaint::free(ctx); }
};
struct lama_context_deleter {
    void operator()(lama::context * ctx) const { lama::free(ctx); }
};
struct detector_deleter {
    void operator()(ocr_detect::context * ctx) const { ocr_detect::free(ctx); }
};
struct image_deleter {
    void operator()(unsigned char * pixels) const { stbi_image_free(pixels); }
};
using image_ptr = std::unique_ptr<unsigned char, image_deleter>;

struct temporary_output {
    fs::path path;
    ~temporary_output() {
        if (!path.empty()) {
            std::error_code ignored;
            fs::remove(path, ignored);
        }
    }
};

bool fail(std::string & error, const std::string & message) {
    error = message;
    return false;
}

bool integer(const std::string & value, int minimum, int maximum, int & result) {
    if (value.empty()) return false;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    return parsed.ec == std::errc() && parsed.ptr == value.data() + value.size() && result >= minimum &&
           result <= maximum;
}

bool latent_engine(const options & opt) {
    return opt.engine == "moebius" || opt.engine == "pixelhacker";
}
bool digest(const std::string & value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(),
                                             [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
bool decimal(const std::string & value, float minimum, float maximum, float & result) {
    if (value.empty()) return false;
    char * end = nullptr;
    errno = 0;
    result = std::strtof(value.c_str(), &end);
    return errno == 0 && end == value.c_str() + value.size() && std::isfinite(result) && result >= minimum &&
           result <= maximum;
}

bool parse_region(const std::string & value, std::array<int, 4> & region) {
    size_t begin = 0;
    for (int i = 0; i < 4; ++i) {
        const size_t end = value.find(',', begin);
        if ((i == 3) != (end == std::string::npos)) return false;
        if (!integer(value.substr(begin, end == std::string::npos ? end : end - begin), i < 2 ? 0 : 1,
                     inpaint::max_dimension, region[i]))
            return false;
        begin = end == std::string::npos ? value.size() : end + 1;
    }
    return true;
}

bool parse(int argc, char ** argv, options & opt, std::string & error) {
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        std::string name = argv[i];
        if (name == "-t") name = "--threads";
        if (!seen.insert(name).second) return fail(error, "duplicate option: " + name);
        if (name == "--mask-from-text") {
            opt.from_text = true;
            continue;
        }
        if (name == "--inpaint-overwrite") {
            opt.overwrite = true;
            continue;
        }
        if (name == "--inpaint-tiles") {
            opt.tiled = true;
            continue;
        }
        const std::set<std::string> values{ "--inpaint-model",
                                            "--inpaint",
                                            "--mask",
                                            "--inpaint-output",
                                            "--inpaint-list",
                                            "--inpaint-dir",
                                            "--mask-dir",
                                            "--output-dir",
                                            "--mask-text-model",
                                            "--mask-region",
                                            "--mask-dilate",
                                            "--threads",
                                            "--gpu-backend",
                                            "--inpaint-engine",
                                            "--inpaint-vae",
                                            "--inpaint-model-sha256",
                                            "--inpaint-vae-sha256",
                                            "--inpaint-checkpoint-sha256",
                                            "--inpaint-steps",
                                            "--inpaint-guidance",
                                            "--inpaint-noise-offset",
                                            "--inpaint-seed" };
        if (values.count(name) == 0) return fail(error, "unsupported inpainting option: " + name);
        if (i + 1 == argc || !*argv[i + 1] || std::strncmp(argv[i + 1], "--", 2) == 0)
            return fail(error, "missing value for " + name);
        const std::string value = argv[++i];
        if (name == "--gpu-backend") {
            if (value != "auto" && value != "cpu" && value != "cuda" && value != "metal" && value != "mps")
                return fail(error, "--gpu-backend must be auto, cpu, cuda or metal (mps aliases metal)");
            opt.backend = value == "mps" ? "metal" : value;
        } else if (name == "--inpaint-engine")
            opt.engine = value;
        else if (name == "--inpaint-vae")
            opt.vae = value;
        else if (name == "--inpaint-model-sha256")
            opt.model_sha256 = value;
        else if (name == "--inpaint-vae-sha256")
            opt.vae_sha256 = value;
        else if (name == "--inpaint-checkpoint-sha256")
            opt.checkpoint_sha256 = value;
        else if (name == "--inpaint-steps") {
            if (!integer(value, 2, 100, opt.steps))
                return fail(error, "--inpaint-steps must be an integer from 2 to 100");
        } else if (name == "--inpaint-guidance") {
            if (!decimal(value, 0, 30, opt.guidance)) return fail(error, "--inpaint-guidance must be finite in [0,30]");
        } else if (name == "--inpaint-noise-offset") {
            if (!decimal(value, 0, 1, opt.noise_offset))
                return fail(error, "--inpaint-noise-offset must be finite in [0,1]");
        } else if (name == "--inpaint-seed") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), opt.seed);
            if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size())
                return fail(error, "--inpaint-seed must be an unsigned 64-bit integer");
        } else if (name == "--inpaint-model")
            opt.model = value;
        else if (name == "--inpaint")
            opt.image = value;
        else if (name == "--mask")
            opt.mask = value;
        else if (name == "--inpaint-output")
            opt.output = value;
        else if (name == "--inpaint-list")
            opt.list = value;
        else if (name == "--inpaint-dir")
            opt.directory = value;
        else if (name == "--mask-dir")
            opt.mask_directory = value;
        else if (name == "--output-dir")
            opt.output_directory = value;
        else if (name == "--mask-text-model")
            opt.text_model = value;
        else if (name == "--mask-region") {
            if (!parse_region(value, opt.region)) return fail(error, "--mask-region must be x,y,w,h with positive w,h");
            opt.has_region = true;
        } else if (name == "--mask-dilate") {
            if (!integer(value, 0, 256, opt.dilation))
                return fail(error, "--mask-dilate must be an integer from 0 to 256");
        } else if (name == "--threads") {
            if (!integer(value, 1, 64, opt.threads)) return fail(error, "--threads must be an integer from 1 to 64");
        }
    }
    const int modes = !opt.image.empty() + !opt.list.empty() + !opt.directory.empty();
    if (modes != 1) return fail(error, "choose exactly one of --inpaint, --inpaint-list, or --inpaint-dir");
    if (opt.model.empty()) return fail(error, "--inpaint-model is required (reviewed F32 GGUF)");
    if (opt.engine != "migan256" && opt.engine != "migan512" && opt.engine != "lama" && !latent_engine(opt))
        return fail(error, "--inpaint-engine must be migan256, migan512, lama, moebius or pixelhacker");
    if (opt.engine == "pixelhacker" && !seen.count("--inpaint-guidance")) opt.guidance = 4.5f;
    if (latent_engine(opt)) {
        if (opt.vae.empty() || !digest(opt.model_sha256) || !digest(opt.vae_sha256))
            return fail(error, "latent models require --inpaint-vae, --inpaint-model-sha256 and --inpaint-vae-sha256 "
                               "from conversion receipts");
        if (!opt.checkpoint_sha256.empty())
            return fail(error,
                        "latent source checkpoints are pinned; --inpaint-checkpoint-sha256 applies only to migan512");
    } else {
        if (!opt.vae.empty() || !opt.vae_sha256.empty() || opt.tiled || seen.count("--inpaint-steps") ||
            seen.count("--inpaint-guidance") || seen.count("--inpaint-seed") || seen.count("--inpaint-noise-offset"))
            return fail(error, "VAE, sampler and --inpaint-tiles options require moebius or pixelhacker");
        if (opt.engine == "migan512") {
            if (!digest(opt.model_sha256) || !digest(opt.checkpoint_sha256))
                return fail(error, "migan512 requires --inpaint-model-sha256 and --inpaint-checkpoint-sha256 from "
                                   "reviewed capture/conversion receipts");
        } else if (opt.engine == "lama") {
            if (!digest(opt.model_sha256) || !opt.checkpoint_sha256.empty())
                return fail(error, "lama requires --inpaint-model-sha256 from its conversion receipt; the source "
                                   "checkpoint is pinned");
        } else if (!opt.model_sha256.empty() || !opt.checkpoint_sha256.empty()) {
            return fail(error, "migan256 uses its existing immutable model/checkpoint pins");
        }
    }
    if (opt.from_text != !opt.text_model.empty())
        return fail(error, "--mask-from-text and --mask-text-model must be used together");
    if (opt.from_text && (!opt.mask.empty() || !opt.mask_directory.empty()))
        return fail(error, "--mask-from-text cannot be combined with --mask or --mask-dir");
    if (!opt.image.empty()) {
        if (!opt.from_text && opt.mask.empty()) return fail(error, "--inpaint requires --mask or --mask-from-text");
        if (!opt.mask_directory.empty() || !opt.output_directory.empty())
            return fail(error, "--mask-dir and --output-dir require --inpaint-dir");
    } else {
        if (!opt.mask.empty() || !opt.output.empty())
            return fail(error, "--mask and --inpaint-output apply only to a single --inpaint image");
        if (!opt.list.empty() && (!opt.mask_directory.empty() || !opt.output_directory.empty()))
            return fail(error, "list mode reads mask and output paths from each TSV row");
        if (!opt.directory.empty() && (opt.output_directory.empty() || (!opt.from_text && opt.mask_directory.empty())))
            return fail(error, "--inpaint-dir requires --output-dir and either --mask-dir or --mask-from-text");
    }
    if (opt.dilation < 0) opt.dilation = opt.from_text ? 3 : 0;
    if (opt.overwrite && !opt.image.empty() && (opt.output.empty() || opt.output == "-"))
        return fail(error, "--inpaint-overwrite requires file output");
    return true;
}

bool regular_file(const fs::path & path, std::string & error) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) return fail(error, "not a readable regular file: " + path.string());
    return true;
}

fs::path canonical_path(const fs::path & path) {
    fs::path normalized = fs::weakly_canonical(fs::absolute(path));
    return normalized.make_preferred();
}

bool image_info(const fs::path & path, int & width, int & height, std::string & error) {
    if (!regular_file(path, error)) return false;
    int channels = 0;
    if (!stbi_info(path.string().c_str(), &width, &height, &channels))
        return fail(error, "cannot read image header: " + path.string());
    if (width <= 0 || height <= 0 || width > inpaint::max_dimension || height > inpaint::max_dimension ||
        size_t(width) > inpaint::max_pixels / size_t(height))
        return fail(error, "image exceeds the inpainting size limit: " + path.string());
    return true;
}

bool collect_frames(const options & opt, std::vector<frame> & frames, std::string & error) {
    if (!opt.image.empty()) {
        frames.push_back({ opt.image, opt.mask, opt.output == "-" ? fs::path() : fs::path(opt.output) });
    } else if (!opt.list.empty()) {
        std::ifstream file;
        std::istream * input = &std::cin;
        fs::path base = fs::current_path();
        if (opt.list != "-") {
            file.open(opt.list, std::ios::binary);
            if (!file) return fail(error, "cannot open --inpaint-list: " + opt.list);
            input = &file;
            base = fs::absolute(opt.list).parent_path();
        }
        std::string line;
        size_t line_number = 0;
        while (std::getline(*input, line)) {
            ++line_number;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || (line.front() == '#' && line.find('\t') == std::string::npos)) continue;
            const size_t first = line.find('\t');
            const size_t second = first == std::string::npos ? first : line.find('\t', first + 1);
            if (line.size() > 1024 * 1024 || line.find('\0') != std::string::npos || first == std::string::npos ||
                second == std::string::npos || line.find('\t', second + 1) != std::string::npos || first == 0 ||
                second == first + 1 || second + 1 == line.size())
                return fail(error,
                            "invalid TSV row " + std::to_string(line_number) + ": expected image<TAB>mask<TAB>output");
            const std::string image = line.substr(0, first);
            const std::string mask = line.substr(first + 1, second - first - 1);
            const std::string output = line.substr(second + 1);
            if (image == "-" || output == "-" || (opt.from_text ? mask != "-" : mask == "-"))
                return fail(error, "invalid TSV row " + std::to_string(line_number) +
                                       ": use file paths; only text-derived masks use '-' in the mask column");
            frames.push_back({ base / image, opt.from_text ? fs::path() : base / mask, base / output });
        }
        if (input->bad()) return fail(error, "cannot read --inpaint-list");
    } else {
        if (!fs::is_directory(opt.directory)) return fail(error, "--inpaint-dir is not a directory");
        if (!opt.from_text && !fs::is_directory(opt.mask_directory))
            return fail(error, "--mask-dir is not a directory");
        for (const auto & entry : fs::directory_iterator(opt.directory)) {
            if (!entry.is_regular_file()) continue;
            std::string extension = entry.path().extension().string();
            for (char & c : extension)
                if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
            if (extension != ".png") continue;
            frames.push_back({ entry.path(),
                               opt.from_text ? fs::path() : fs::path(opt.mask_directory) / entry.path().filename(),
                               fs::path(opt.output_directory) / entry.path().filename() });
        }
        std::sort(frames.begin(), frames.end(), [](const frame & a, const frame & b) { return a.image < b.image; });
    }
    if (frames.empty()) return fail(error, "no input frames (directory mode reads non-recursive *.png files)");
    return true;
}

// Complete path/header preflight precedes model loading or output creation.
// Protect every batch input, not only the frame currently being processed.
bool preflight(const options & opt, std::vector<frame> & frames, std::string & error) {
    std::set<fs::path, path_less> protected_paths{ canonical_path(opt.model) };
    if (!opt.text_model.empty()) protected_paths.insert(canonical_path(opt.text_model));
    if (!opt.vae.empty()) protected_paths.insert(canonical_path(opt.vae));
    if (!opt.list.empty() && opt.list != "-") protected_paths.insert(canonical_path(opt.list));
    for (auto & item : frames) {
        if (!image_info(item.image, item.width, item.height, error)) return false;
        if (latent_engine(opt) && !opt.tiled && (item.width != 512 || item.height != 512))
            return fail(
                error, "latent models require 512x512; use --inpaint-tiles to explicitly select independent 512 tiles");
        if (opt.engine == "lama" && (item.width < lama::min_dimension || item.height < lama::min_dimension ||
                                     item.width > lama::max_dimension || item.height > lama::max_dimension))
            return fail(error, "lama requires each dimension from 9 to 1024 pixels");
        protected_paths.insert(canonical_path(item.image));
        if (!opt.from_text) {
            int width = 0, height = 0;
            if (!image_info(item.mask, width, height, error)) return false;
            if (width != item.width || height != item.height)
                return fail(error, "mask dimensions must match the image: " + item.mask.string());
            protected_paths.insert(canonical_path(item.mask));
        }
        if (opt.has_region &&
            (opt.region[0] >= item.width || opt.region[1] >= item.height ||
             opt.region[2] > item.width - opt.region[0] || opt.region[3] > item.height - opt.region[1]))
            return fail(error, "--mask-region must be inside every image: " + item.image.string());
    }
    std::set<fs::path, path_less> output_paths;
    for (const auto & item : frames) {
        if (item.output.empty()) continue;
        const fs::path output = canonical_path(item.output);
        if (protected_paths.count(output))
            return fail(error, "output would overwrite an input or model: " + item.output.string());
        if (!output_paths.insert(output).second) return fail(error, "duplicate output path: " + item.output.string());
        if (fs::is_symlink(fs::symlink_status(item.output)))
            return fail(error, "output must not be a symlink: " + item.output.string());
        if (fs::exists(item.output)) {
            if (!opt.overwrite) return fail(error, "output exists; use --inpaint-overwrite: " + item.output.string());
            if (!regular_file(item.output, error)) return false;
            for (const auto & source : protected_paths)
                if (fs::exists(source) && fs::equivalent(item.output, source))
                    return fail(error, "output aliases an input or model: " + item.output.string());
        }
        const fs::path parent = fs::absolute(item.output).parent_path();
        if (opt.directory.empty()) {
            if (!fs::is_directory(parent))
                return fail(error, "output parent directory does not exist: " + parent.string());
        } else if (fs::exists(parent) && !fs::is_directory(parent)) {
            return fail(error, "--output-dir is not a directory");
        }
    }
    return regular_file(opt.model, error) && (opt.text_model.empty() || regular_file(opt.text_model, error)) &&
           (opt.vae.empty() || regular_file(opt.vae, error));
}

image_ptr load_image(const fs::path & path, int expected_width, int expected_height, int channels,
                     std::string & error) {
    // Recheck immediately before decoding; a batch may run long after preflight.
    int width = 0, height = 0;
    if (!image_info(path, width, height, error)) return {};
    if (width != expected_width || height != expected_height) {
        fail(error, "image dimensions changed after preflight: " + path.string());
        return {};
    }
    int original_channels = 0;
    image_ptr pixels(stbi_load(path.string().c_str(), &width, &height, &original_channels, channels));
    if (!pixels || width != expected_width || height != expected_height) {
        fail(error, "cannot decode image with expected dimensions: " + path.string());
        return {};
    }
    return pixels;
}

void dilate(std::vector<uint8_t> & mask, int width, int height, int radius) {
    if (!radius) return;
    std::vector<uint8_t> horizontal(mask.size());
    // Separable sliding-window square dilation is O(width*height), independent
    // of radius. A subtitle band need not pay O(radius^2) per pixel.
    for (int y = 0; y < height; ++y) {
        int count = 0;
        const size_t row = size_t(y) * width;
        for (int x = 0; x <= std::min(radius, width - 1); ++x) count += mask[row + x] != 0;
        for (int x = 0; x < width; ++x) {
            horizontal[row + x] = count ? 255 : 0;
            if (x - radius >= 0) count -= mask[row + x - radius] != 0;
            if (x + radius + 1 < width) count += mask[row + x + radius + 1] != 0;
        }
    }
    for (int x = 0; x < width; ++x) {
        int count = 0;
        for (int y = 0; y <= std::min(radius, height - 1); ++y) count += horizontal[size_t(y) * width + x] != 0;
        for (int y = 0; y < height; ++y) {
            mask[size_t(y) * width + x] = count ? 255 : 0;
            if (y - radius >= 0) count -= horizontal[size_t(y - radius) * width + x] != 0;
            if (y + radius + 1 < height) count += horizontal[size_t(y + radius + 1) * width + x] != 0;
        }
    }
}

bool rasterize(const ocr_detect::text_box & box, int width, int height, std::vector<uint8_t> & mask,
               std::string & error) {
    if (!std::isfinite(box.x) || !std::isfinite(box.y) || !std::isfinite(box.w) || !std::isfinite(box.h) ||
        !std::isfinite(box.score))
        return fail(error, "text detector returned a nonfinite box");
    double area = 0.0;
    for (int i = 0; i < 4; ++i) {
        if (!std::isfinite(box.qx[i]) || !std::isfinite(box.qy[i]))
            return fail(error, "text detector returned a nonfinite quadrilateral");
        const int j = (i + 1) % 4;
        area += double(box.qx[i]) * box.qy[j] - double(box.qx[j]) * box.qy[i];
    }
    std::array<double, 4> xs{}, ys{};
    for (int i = 0; i < 4; ++i) {
        // Older DBNet models may provide only an axis-aligned box.
        xs[i] = std::fabs(area) > 1e-6 ? box.qx[i] : box.x + ((i == 1 || i == 2) ? box.w : 0);
        ys[i] = std::fabs(area) > 1e-6 ? box.qy[i] : box.y + (i >= 2 ? box.h : 0);
        xs[i] = std::max(0.0, std::min(double(width), xs[i]));
        ys[i] = std::max(0.0, std::min(double(height), ys[i]));
    }
    const int top = std::max(0, int(std::floor(*std::min_element(ys.begin(), ys.end()))));
    const int bottom = std::min(height, int(std::ceil(*std::max_element(ys.begin(), ys.end()))));
    for (int y = top; y < bottom; ++y) {
        const double scan = y + 0.5;
        std::array<double, 4> intersections{};
        size_t count = 0;
        for (int i = 0; i < 4; ++i) {
            const int j = (i + 1) % 4;
            if ((ys[i] <= scan && ys[j] > scan) || (ys[j] <= scan && ys[i] > scan))
                intersections[count++] = xs[i] + (scan - ys[i]) * (xs[j] - xs[i]) / (ys[j] - ys[i]);
        }
        std::sort(intersections.begin(), intersections.begin() + count);
        for (size_t i = 0; i + 1 < count; i += 2) {
            const int left = std::max(0, int(std::ceil(intersections[i] - 0.5)));
            const int right = std::min(width, int(std::ceil(intersections[i + 1] - 0.5)));
            std::fill(mask.begin() + size_t(y) * width + left, mask.begin() + size_t(y) * width + right, uint8_t(255));
        }
    }
    return true;
}

bool make_mask(const options & opt, const frame & item, const uint8_t * rgb, ocr_detect::context * detector,
               std::vector<uint8_t> & mask, std::string & error) {
    mask.assign(size_t(item.width) * item.height, 0);
    if (opt.from_text) {
        std::vector<ocr_detect::text_box> boxes;
        if (!ocr_detect::detect_rgb_checked(detector, rgb, item.width, item.height, 3, ocr_detect::rapid_defaults(),
                                            boxes, error))
            return false;
        for (const auto & box : boxes)
            if (!rasterize(box, item.width, item.height, mask, error)) return false;
    } else {
        image_ptr grayscale = load_image(item.mask, item.width, item.height, 1, error);
        if (!grayscale) return false;
        for (size_t i = 0; i < mask.size(); ++i) mask[i] = grayscale.get()[i] >= 128 ? 255 : 0;
    }
    dilate(mask, item.width, item.height, opt.dilation);
    if (opt.has_region) {
        for (int y = 0; y < item.height; ++y)
            for (int x = 0; x < item.width; ++x)
                if (x < opt.region[0] || y < opt.region[1] || x >= opt.region[0] + opt.region[2] ||
                    y >= opt.region[1] + opt.region[3])
                    mask[size_t(y) * item.width + x] = 0;
    }
    return true;
}

std::FILE * open_exclusive_output(const fs::path & path) {
#ifdef _WIN32
    const int flags = _O_WRONLY | _O_CREAT | _O_EXCL;
    const int fd = _wopen(path.c_str(), flags | _O_BINARY, _S_IREAD | _S_IWRITE);
    if (fd < 0) return nullptr;
    std::FILE * out = _fdopen(fd, "wb");
    if (!out) _close(fd);
#else
    const int flags = O_WRONLY | O_CREAT | O_EXCL;
    const int fd = open(path.c_str(), flags | O_NOFOLLOW, S_IRUSR | S_IWUSR);
    if (fd < 0) return nullptr;
    std::FILE * out = fdopen(fd, "wb");
    if (!out) close(fd);
#endif
    return out;
}

bool write_output(const options & opt, const frame & item, const std::vector<uint8_t> & rgb, std::string & error) {
    // Encode before opening a destination so an encoding failure cannot
    // truncate an existing output. The common writer attaches provenance.
    std::string png, mime;
    std::string provenance = opt.engine == "migan256" ? "migan-inpaint" : opt.engine + "-inpaint";
    if (core_env::on("CRISPEMBED_INPAINT_GRAPH")) {
        provenance += ";execution=ggml-graph;backend_request=" + (opt.backend.empty() ? "auto" : opt.backend);
        provenance +=
            core_env::on("CRISPEMBED_INPAINT_FAST") ? ";precision_request=mixed-f16" : ";precision_request=f32";
    }
    if (opt.engine == "migan512" || opt.engine == "lama") provenance += ";model_sha256=" + opt.model_sha256;
    if (opt.engine == "migan512") provenance += ";checkpoint_sha256=" + opt.checkpoint_sha256;
    if (latent_engine(opt))
        provenance += ";steps=" + std::to_string(opt.steps) + ";seed=" + std::to_string(opt.seed) +
                      ";guidance=" + std::to_string(opt.guidance) +
                      ";noise_offset=" + std::to_string(opt.noise_offset) + ";tiles=" + (opt.tiled ? "1" : "0") +
                      ";model_sha256=" + opt.model_sha256 + ";vae_sha256=" + opt.vae_sha256;
    if (!core_imgout::emit_to_string(png, mime, rgb.data(), item.width, item.height, 3, provenance.c_str()) ||
        mime != "image/png")
        return fail(error, "cannot encode PNG output");
    std::FILE * out = stdout;
    temporary_output temporary;
    if (!item.output.empty()) {
        if (opt.overwrite) {
            // Stage replacement beside the destination. Renaming on the same
            // filesystem keeps the previous file intact if encoding/writing
            // fails and never truncates a hard-linked input inode.
            std::random_device random;
            for (int attempt = 0; attempt < 32; ++attempt) {
                const fs::path candidate =
                    item.output.parent_path() /
                    (".crispembed-inpaint-" + std::to_string(random()) + "-" + std::to_string(random()));
                out = open_exclusive_output(candidate);
                if (out) {
                    temporary.path = candidate;
                    break;
                }
                if (errno != EEXIST) break;
            }
        } else {
            out = open_exclusive_output(item.output);
        }
        if (!out) return fail(error, "cannot create output " + item.output.string() + ": " + std::strerror(errno));
    }
#ifdef _WIN32
    else if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
        return fail(error, "cannot set stdout to binary mode");
    }
#endif
    bool ok = std::fwrite(png.data(), 1, png.size(), out) == png.size();
    if (std::fflush(out) != 0) ok = false;
    if (out != stdout && std::fclose(out) != 0) ok = false;
    if (!ok) {
        if (!item.output.empty() && !opt.overwrite) {
            std::error_code ignored;
            fs::remove(item.output, ignored);
        }
        return fail(error,
                    "cannot write PNG output: " + (item.output.empty() ? std::string("stdout") : item.output.string()));
    }
    if (!temporary.path.empty()) {
        std::error_code ec;
        fs::rename(temporary.path, item.output, ec);
        if (ec) return fail(error, "cannot replace output " + item.output.string() + ": " + ec.message());
        temporary.path.clear();
    }
    return true;
}

int execute(const options & opt, std::string & error) {
    if (!opt.backend.empty() && opt.backend != "cpu" && !core_env::on("CRISPEMBED_INPAINT_GRAPH")) {
        fail(error, "GPU/auto inpainting requires CRISPEMBED_INPAINT_GRAPH=1");
        return 2;
    }
    if (core_env::on("CRISPEMBED_INPAINT_FAST") && !core_env::on("CRISPEMBED_INPAINT_GRAPH")) {
        fail(error, "CRISPEMBED_INPAINT_FAST requires CRISPEMBED_INPAINT_GRAPH=1");
        return 2;
    }
    if (!opt.backend.empty()) crispasr_set_gpu_backend_pref(opt.backend.c_str());
    const char * gate = opt.engine == "moebius"       ? "CRISPEMBED_MOEBIUS"
                        : opt.engine == "pixelhacker" ? "CRISPEMBED_PIXELHACKER"
                        : opt.engine == "lama"        ? "CRISPEMBED_LAMA"
                                                      : "CRISPEMBED_MIGAN";
    if (!core_env::on(gate) || (opt.engine == "migan512" && !core_env::on("CRISPEMBED_MIGAN512"))) {
        fail(error, std::string("experimental inpainting requires ") + gate + "=1" +
                        (opt.engine == "migan512" ? " and CRISPEMBED_MIGAN512=1" : ""));
        return 2;
    }
    if (core_imgout::want_ppm()) {
        fail(error, "inpainting emits PNG; unset CRISPEMBED_IMAGE_FORMAT or set it to png");
        return 2;
    }
    std::vector<frame> frames;
    if (!collect_frames(opt, frames, error) || !preflight(opt, frames, error)) return 2;

    std::unique_ptr<ocr_detect::context, detector_deleter> detector;
    if (opt.from_text) {
        gguf_context * metadata = core_gguf::open_metadata(opt.text_model.c_str());
        if (!metadata) {
            fail(error, "cannot read text detector GGUF");
            return 1;
        }
        const std::string architecture = core_gguf::kv_str(metadata, "general.architecture", "");
        core_gguf::free_metadata(metadata);
        if (architecture != "dbnet") {
            fail(error, "--mask-text-model requires a DBNet GGUF (general.architecture=dbnet)");
            return 2;
        }
        ocr_detect::context * raw = nullptr;
        const bool loaded = ocr_detect::load(&raw, opt.text_model.c_str(), opt.threads);
        detector.reset(raw);
        if (!loaded || !detector) {
            fail(error, "cannot load DBNet text detector");
            return 1;
        }
    }
    std::unique_ptr<inpaint::context, context_deleter> ctx;
    std::unique_ptr<latent_inpaint::context, latent_context_deleter> latent;
    std::unique_ptr<lama::context, lama_context_deleter> fourier;
    if (latent_engine(opt)) {
        latent_inpaint::Options settings;
        settings.model = opt.engine;
        settings.denoiser_path = opt.model;
        settings.vae_path = opt.vae;
        settings.denoiser_sha256 = opt.model_sha256;
        settings.vae_sha256 = opt.vae_sha256;
        settings.threads = opt.threads;
        settings.steps = opt.steps;
        settings.guidance = opt.guidance;
        settings.noise_offset = opt.noise_offset;
        settings.seed = opt.seed;
        settings.tiled = opt.tiled;
        latent.reset(latent_inpaint::init(settings, error));
        if (!latent) return 1;
    } else if (opt.engine == "lama") {
        fourier.reset(lama::init(opt.model.c_str(), opt.model_sha256.c_str(), opt.threads, error));
        if (!fourier) return 1;
    } else {
        ctx.reset(opt.engine == "migan512" ? inpaint::init_variant(opt.model.c_str(), opt.checkpoint_sha256.c_str(),
                                                                   opt.model_sha256.c_str(), 512, opt.threads, error)
                                           : inpaint::init(opt.model.c_str(), opt.threads, error));
        if (!ctx) return 1;
    }
    if (!opt.directory.empty()) fs::create_directories(opt.output_directory);

    for (size_t index = 0; index < frames.size(); ++index) {
        const frame & item = frames[index];
        image_ptr rgb = load_image(item.image, item.width, item.height, 3, error);
        std::vector<uint8_t> mask;
        const size_t rgb_bytes = size_t(item.width) * item.height * 3;
        std::vector<uint8_t> output(rgb_bytes);
        if (!rgb || !make_mask(opt, item, rgb.get(), detector.get(), mask, error)) {
            error = "frame " + std::to_string(index + 1) + " (" + item.image.string() + "): " + error;
            return 1;
        }
        const bool processed =
            latent    ? latent_inpaint::process_rgb8(latent.get(), rgb.get(), rgb_bytes, mask.data(), mask.size(),
                                                     item.width, item.height, output.data(), output.size(), error)
            : fourier ? lama::process_rgb8(fourier.get(), rgb.get(), rgb_bytes, mask.data(), mask.size(), item.width,
                                           item.height, output.data(), output.size(), error)
                      : inpaint::process_rgb8(ctx.get(), rgb.get(), rgb_bytes, mask.data(), mask.size(), item.width,
                                              item.height, output.data(), output.size(), error);
        if (!processed || !write_output(opt, item, output, error)) {
            error = "frame " + std::to_string(index + 1) + " (" + item.image.string() + "): " + error;
            return 1;
        }
        std::fprintf(stderr, "inpaint: %zu/%zu %s\n", index + 1, frames.size(), item.image.string().c_str());
    }
    return 0;
}

} // namespace

bool requested(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--inpaint" || arg.compare(0, 10, "--inpaint-") == 0 || arg == "--mask" ||
            arg.compare(0, 7, "--mask-") == 0 || arg == "--output-dir")
            return true;
    }
    return false;
}

void print_help(std::FILE * out) {
    std::fprintf(out, "\nExperimental image inpainting (each engine requires its opt-in gate):\n"
                      "  CRISPEMBED_INPAINT_GRAPH=1  Persistent native ggml graph execution\n"
                      "  CRISPEMBED_INPAINT_FAST=1   Explicit mixed precision graph profile (GPU only)\n"
                      "  CRISPEMBED_INPAINT_PROFILE=1  Print graph allocation, reuse and transfer counters\n"
                      "  --gpu-backend NAME      auto, cpu, cuda, metal (mps aliases native Metal)\n"
                      "  --inpaint-engine NAME    migan256(default), migan512, lama, moebius or pixelhacker\n"
                      "  --inpaint-model PATH     Converted F32 GGUF (see docs/inpainting-cli.md)\n"
                      "  --inpaint-model-sha256 H Model GGUF digest from conversion receipt (new engines)\n"
                      "  --inpaint-checkpoint-sha256 H  Reviewed original checkpoint digest (migan512)\n"
                      "  --inpaint-vae PATH       Shared SDXL VAE F32 GGUF (latent models)\n"
                      "  --inpaint-vae-sha256 H   VAE GGUF digest from conversion receipt\n"
                      "  --inpaint-steps N        Requested diffusion steps, 2..100 (default 20; 19 executed)\n"
                      "  --inpaint-guidance F     Guidance, 0..30 (default 2 Moebius; 4.5 PixelHacker)\n"
                      "  --inpaint-noise-offset F Channel noise offset, 0..1 (default 0.0357)\n"
                      "  --inpaint-seed N         Unsigned64-bit native noise seed (default 0)\n"
                      "  --inpaint-tiles          Explicit independent 512 tiles for non-512 latent input\n"
                      "  --inpaint FILE           Inpaint one image; PNG goes to stdout by default\n"
                      "  --mask FILE              Same-size grayscale mask; values >=128 mean fill\n"
                      "  --inpaint-output FILE    Write PNG to a file ('-' means stdout)\n"
                      "  --inpaint-dir DIR        Process sorted *.png frames without reloading the model\n"
                      "  --mask-dir DIR           Same filenames as --inpaint-dir\n"
                      "  --output-dir DIR         Directory for processed PNGs; created if needed\n"
                      "  --inpaint-list FILE|-    TSV: image<TAB>mask<TAB>output, one frame per line\n"
                      "                            Relative paths are relative to the list file (stdin: cwd)\n"
                      "                            Output parent directories must already exist\n"
                      "  --mask-from-text         Build a mask from detected text quadrilaterals\n"
                      "  --mask-text-model PATH   DBNet GGUF required with --mask-from-text\n"
                      "                            Text-mask list rows use '-' in the mask column\n"
                      "  --mask-region x,y,w,h    Restrict the final mask to this in-bounds rectangle\n"
                      "  --mask-dilate N          Expand mask by N pixels (0..256; default 3 text / 0 supplied)\n"
                      "  --inpaint-overwrite      Allow existing output files; inputs are always protected\n"
                      "  -t, --threads N          CPU threads (1..64; default 1)\n"
                      "  --inpaint-help           Show these options\n"
                      "  Images keep their dimensions and unmasked RGB bytes; batch frames are independent.\n");
}

int run(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--inpaint-help") == 0 || std::strcmp(argv[i], "--help") == 0 ||
            std::strcmp(argv[i], "-h") == 0) {
            print_help(stdout);
            return 0;
        }
    try {
        options opt;
        std::string error;
        const int status = parse(argc, argv, opt, error) ? execute(opt, error) : 2;
        if (status) std::fprintf(stderr, "inpaint: error: %s\n", error.c_str());
        return status;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "inpaint: error: %s\n", error.what());
        return 1;
    }
}

} // namespace inpaint_cli
