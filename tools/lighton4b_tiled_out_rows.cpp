// Bounded original-F16 -> paired HF/tiled weighted Q4_K controls. No model graph.
#include "ggml.h"
#include "core/sha256.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
namespace fs = std::filesystem;
static void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
static void hash_floats(core_sha256::sha256_state & hash, const std::vector<float> & values, size_t n) {
    core_sha256::sha256_update(hash, reinterpret_cast<const uint8_t *>(values.data()), n * sizeof(float));
}
static bool fresh(const fs::path & path) { return !fs::exists(path) && !fs::is_symlink(path); }
int main(int argc, char ** argv) {
    fs::path tiled_partial, hf_partial;
    bool tiled_created = false, hf_created = false;
    try {
        require(argc == 10, "expected source offset rows physical-means source-span-sha tiled-output hf-control report max-elements");
        const uint16_t endian = 1;
        require(*reinterpret_cast<const uint8_t *>(&endian) == 1, "little-endian worker required");
        constexpr size_t columns = 4096, block = 256, bytes_per_block = 144;
        const uint64_t offset = std::stoull(argv[2]), rows = std::stoull(argv[3]);
        const size_t maximum = std::stoull(argv[9]);
        require(rows > 0 && rows <= 2560 && maximum >= columns && maximum <= 65536 && maximum % columns == 0 && offset % 2 == 0, "bounded complete-row geometry required");
        require(std::strlen(argv[5]) == 64, "source span identity required");
        const fs::path tiled_output(argv[6]), hf_output(argv[7]), report_path(argv[8]);
        require(tiled_output != hf_output && tiled_output != report_path && hf_output != report_path, "distinct new output paths required");
        tiled_partial = tiled_output.string() + ".partial"; hf_partial = hf_output.string() + ".partial";
        std::set<fs::path> outputs;
        for (const auto & path : {tiled_output, hf_output, report_path, tiled_partial, hf_partial}) outputs.insert(fs::absolute(path).lexically_normal());
        require(outputs.size() == 5, "output/partial path aliases rejected");
        require(fresh(tiled_output) && fresh(hf_output) && fresh(report_path) && fresh(tiled_partial) && fresh(hf_partial), "outputs/partials must not exist");
        std::ifstream input(argv[1], std::ios::binary), means_file(argv[4], std::ios::binary);
        require(bool(input) && bool(means_file), "source/statistics unavailable");
        input.seekg(0, std::ios::end); const auto source_size = input.tellg();
        const uint64_t elements = rows * columns;
        require(source_size >= 0 && offset <= uint64_t(source_size) && 2 * elements <= uint64_t(source_size) - offset, "source F16 span truncated");
        means_file.seekg(0, std::ios::end); require(means_file.tellg() == std::streamoff(columns * 4), "exact physical mean vector required"); means_file.seekg(0);
        std::vector<float> physical_mean(columns), hf_mean(columns);
        means_file.read(reinterpret_cast<char *>(physical_mean.data()), columns * 4); require(size_t(means_file.gcount()) == columns * 4, "mean vector truncated");
        std::vector<uint32_t> permutation(columns); long double weight_sum = 0;
        for (size_t i = 0; i < columns; ++i) {
            const size_t head = i / 128, lane = i % 128;
            permutation[i] = uint32_t((16 * (head % 2) + head / 2) * 128 + lane);
            require(std::isfinite(physical_mean[i]) && physical_mean[i] >= 0, "finite nonnegative physical means required");
            weight_sum += physical_mean[i];
        }
        require(weight_sum > 0, "nonempty importance energy required");
        for (size_t i = 0; i < columns; ++i) hf_mean[i] = physical_mean[permutation[i]];
        core_sha256::sha256_state physical_mean_hash, hf_mean_hash, permutation_hash;
        hash_floats(physical_mean_hash, physical_mean, columns); hash_floats(hf_mean_hash, hf_mean, columns);
        core_sha256::sha256_update(permutation_hash, reinterpret_cast<const uint8_t *>(permutation.data()), columns * 4);
        std::ofstream tiled(tiled_partial, std::ios::binary); tiled_created = bool(tiled);
        std::ofstream hf(hf_partial, std::ios::binary); hf_created = bool(hf);
        require(tiled_created && hf_created, "fresh partial output unavailable"); input.seekg(offset);
        const size_t chunk_rows = maximum / columns, chunk_elements = chunk_rows * columns;
        std::vector<ggml_fp16_t> source(chunk_elements);
        std::vector<float> hf_values(chunk_elements), physical_values(chunk_elements), hf_decoded(chunk_elements), physical_decoded(chunk_elements), tiled_as_hf(chunk_elements);
        std::vector<uint8_t> hf_quant(chunk_elements / block * bytes_per_block), tiled_quant(hf_quant.size());
        core_sha256::sha256_state source_hash, hf_hash, tiled_hash, hf_decoded_hash, physical_decoded_hash, canonical_tiled_hash;
        long double hf_squared = 0, tiled_squared = 0, hf_weighted = 0, tiled_weighted = 0, canonical_squared = 0;
        double hf_maximum = 0, tiled_maximum = 0, canonical_maximum = 0;
        uint64_t canonical_different_bits = 0;
        for (uint64_t begin = 0; begin < elements; begin += chunk_elements) {
            const size_t count = size_t(std::min<uint64_t>(chunk_elements, elements - begin));
            input.read(reinterpret_cast<char *>(source.data()), count * 2); require(size_t(input.gcount()) == count * 2, "source row chunk truncated");
            core_sha256::sha256_update(source_hash, reinterpret_cast<const uint8_t *>(source.data()), count * 2);
            for (size_t i = 0; i < count; ++i) {
                const float value = ggml_fp16_to_fp32(source[i]); require(std::isfinite(value), "nonfinite original F16");
                hf_values[i] = value; physical_values[(i / columns) * columns + permutation[i % columns]] = value;
            }
            const auto hf_written = ggml_quantize_chunk(GGML_TYPE_Q4_K, hf_values.data(), hf_quant.data(), 0, count / columns, columns, hf_mean.data());
            const auto tiled_written = ggml_quantize_chunk(GGML_TYPE_Q4_K, physical_values.data(), tiled_quant.data(), 0, count / columns, columns, physical_mean.data());
            require(hf_written == count / block * bytes_per_block && tiled_written == hf_written, "weighted Q4 row byte count differs");
            hf.write(reinterpret_cast<const char *>(hf_quant.data()), hf_written); tiled.write(reinterpret_cast<const char *>(tiled_quant.data()), tiled_written); require(bool(hf) && bool(tiled), "quantized chunk write failed");
            core_sha256::sha256_update(hf_hash, hf_quant.data(), hf_written); core_sha256::sha256_update(tiled_hash, tiled_quant.data(), tiled_written);
            ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(hf_quant.data(), hf_decoded.data(), count);
            ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(tiled_quant.data(), physical_decoded.data(), count);
            for (size_t i = 0; i < count; ++i) {
                const size_t physical_index = (i / columns) * columns + permutation[i % columns];
                tiled_as_hf[i] = physical_decoded[physical_index];
                require(std::isfinite(hf_decoded[i]) && std::isfinite(tiled_as_hf[i]), "nonfinite Q4 decode");
                const double a = double(hf_decoded[i]) - hf_values[i], b = double(tiled_as_hf[i]) - hf_values[i], difference = double(tiled_as_hf[i]) - hf_decoded[i];
                hf_squared += a * a; tiled_squared += b * b; canonical_squared += difference * difference;
                hf_weighted += hf_mean[i % columns] * (a * a); tiled_weighted += hf_mean[i % columns] * (b * b);
                hf_maximum = std::max(hf_maximum, std::abs(a)); tiled_maximum = std::max(tiled_maximum, std::abs(b)); canonical_maximum = std::max(canonical_maximum, std::abs(difference));
                if (std::memcmp(&hf_decoded[i], &tiled_as_hf[i], 4) != 0) ++canonical_different_bits;
            }
            hash_floats(hf_decoded_hash, hf_decoded, count); hash_floats(physical_decoded_hash, physical_decoded, count); hash_floats(canonical_tiled_hash, tiled_as_hf, count);
        }
        const auto original = core_sha256::sha256_finish(source_hash); require(original == argv[5], "original F16 tensor span SHA differs");
        hf.flush(); tiled.flush(); require(bool(hf) && bool(tiled), "quantized output flush failed"); hf.close(); tiled.close();
        fs::rename(hf_partial, hf_output); hf_created = false; fs::rename(tiled_partial, tiled_output); tiled_created = false;
        std::ofstream report(report_path); report << std::setprecision(17)
            << "{\"schema\":\"lighton4b-tiled-out-row-quantization/v1\",\"source_f16_sha256\":\"" << original
            << "\",\"source_offset\":" << offset << ",\"rows\":" << rows << ",\"columns\":" << columns << ",\"elements\":" << elements
            << ",\"bytes\":" << elements / block * bytes_per_block << ",\"max_chunk_elements\":" << chunk_elements
            << ",\"physical_means_sha256\":\"" << core_sha256::sha256_finish(physical_mean_hash) << "\",\"hf_means_sha256\":\"" << core_sha256::sha256_finish(hf_mean_hash)
            << "\",\"coordinate_map_sha256\":\"" << core_sha256::sha256_finish(permutation_hash) << "\",\"hf_control_q4_sha256\":\"" << core_sha256::sha256_finish(hf_hash)
            << "\",\"tiled_q4_sha256\":\"" << core_sha256::sha256_finish(tiled_hash) << "\",\"hf_control_dequant_f32_sha256\":\"" << core_sha256::sha256_finish(hf_decoded_hash)
            << "\",\"physical_dequant_f32_sha256\":\"" << core_sha256::sha256_finish(physical_decoded_hash) << "\",\"tiled_dequant_as_hf_f32_sha256\":\"" << core_sha256::sha256_finish(canonical_tiled_hash)
            << "\",\"canonical_different_bits\":" << canonical_different_bits << ",\"canonical_max_abs_difference\":" << canonical_maximum
            << ",\"canonical_rms_difference\":" << std::sqrt(double(canonical_squared / elements))
            << ",\"hf_rmse_vs_original_f16\":" << std::sqrt(double(hf_squared / elements)) << ",\"tiled_rmse_vs_original_f16\":" << std::sqrt(double(tiled_squared / elements))
            << ",\"hf_max_abs_vs_original_f16\":" << hf_maximum << ",\"tiled_max_abs_vs_original_f16\":" << tiled_maximum
            << ",\"hf_importance_normalized_weight_rmse\":" << std::sqrt(double(hf_weighted / (weight_sum * rows)))
            << ",\"tiled_importance_normalized_weight_rmse\":" << std::sqrt(double(tiled_weighted / (weight_sum * rows)))
            << ",\"importance_supplied\":true,\"model_inference_executed\":false,\"full_f32_model\":false,\"quantized_donor_used\":false}\n";
        require(bool(report), "row receipt write failed"); return 0;
    } catch (const std::exception & error) {
        if (tiled_created) { std::error_code ignored; fs::remove(tiled_partial, ignored); }
        if (hf_created) { std::error_code ignored; fs::remove(hf_partial, ignored); }
        std::cerr << error.what() << '\n'; return 2;
    }
}
