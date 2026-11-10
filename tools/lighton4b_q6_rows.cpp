// Bounded BF16 checkpoint rows -> new Q6_K bytes. No model graph or requantization.
#include "ggml.h"
#include "ggml-cpu.h"
#include "core/sha256.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
static void require(bool ok, const char * text) { if (!ok) throw std::runtime_error(text); }
int main(int argc, char ** argv) {
    try {
        require(argc == 10, "expected checkpoint offset elements columns candidate offset expected-bf16-sha report max-elements");
        const uint16_t endian = 1; require(*reinterpret_cast<const uint8_t *>(&endian) == 1, "little-endian source worker required");
        const uint64_t input_offset = std::stoull(argv[2]), elements = std::stoull(argv[3]), columns = std::stoull(argv[4]), output_offset = std::stoull(argv[6]);
        const size_t maximum = std::stoull(argv[9]);
        require(maximum == 65536 && columns > 0 && columns <= maximum && columns % 256 == 0 && elements > 0 && elements % columns == 0 && elements <= 67108864, "bounded whole-row Q6 geometry differs");
        require(std::strlen(argv[7]) == 64 && input_offset % 2 == 0, "source BF16 identity differs");
        std::ifstream input(argv[1], std::ios::binary); std::fstream output(argv[5], std::ios::in | std::ios::out | std::ios::binary);
        require(bool(input) && bool(output), "source/candidate unavailable");
        input.seekg(0, std::ios::end); require(input_offset + 2 * elements <= uint64_t(input.tellg()), "source span truncated");
        output.seekp(0, std::ios::end); const auto bytes = elements / 256 * 210;
        require(output_offset + bytes <= uint64_t(output.tellp()), "candidate span truncated");
        input.seekg(input_offset); output.seekp(output_offset);
        const size_t chunk_rows = maximum / columns, chunk_elements = chunk_rows * columns;
        std::vector<uint16_t> source(chunk_elements); std::vector<float> values(chunk_elements); std::vector<uint8_t> quant(chunk_elements / 256 * 210); std::vector<float> dequant(chunk_elements); std::vector<ggml_bf16_t> bf16(chunk_elements);
        core_sha256::sha256_state source_hash, output_hash, bf16_hash;
        for (uint64_t begin = 0; begin < elements; begin += chunk_elements) {
            const size_t count = size_t(std::min<uint64_t>(chunk_elements, elements - begin));
            input.read(reinterpret_cast<char *>(source.data()), count * 2); require(size_t(input.gcount()) == count * 2, "source BF16 read truncated");
            core_sha256::sha256_update(source_hash, reinterpret_cast<const uint8_t *>(source.data()), count * 2);
            for (size_t j = 0; j < count; ++j) { const uint32_t bits = uint32_t(source[j]) << 16; std::memcpy(&values[j], &bits, 4); require(std::isfinite(values[j]), "nonfinite original BF16"); }
            const size_t written = ggml_quantize_chunk(GGML_TYPE_Q6_K, values.data(), quant.data(), 0, count / columns, columns, nullptr);
            require(written == count / 256 * 210, "Q6 byte count differs"); output.write(reinterpret_cast<const char *>(quant.data()), written); require(bool(output), "Q6 output write failed");
            core_sha256::sha256_update(output_hash, quant.data(), written);
            ggml_get_type_traits(GGML_TYPE_Q6_K)->to_float(quant.data(), dequant.data(), count);
            for (size_t j = 0; j < count; ++j) bf16[j] = ggml_fp32_to_bf16(dequant[j]);
            core_sha256::sha256_update(bf16_hash, reinterpret_cast<const uint8_t *>(bf16.data()), count * sizeof(ggml_bf16_t));
        }
        const auto original = core_sha256::sha256_finish(source_hash); require(original == argv[7], "original BF16 tensor hash differs");
        output.flush(); require(bool(output), "Q6 output flush failed");
        std::ofstream report(argv[8]); report << "{\"source_bf16_sha256\":\"" << original << "\",\"q6_sha256\":\"" << core_sha256::sha256_finish(output_hash) << "\",\"dequant_bf16_sha256\":\"" << core_sha256::sha256_finish(bf16_hash) << "\",\"elements\":" << elements << ",\"bytes\":" << bytes << ",\"max_chunk_elements\":" << chunk_elements << ",\"requantized\":false,\"model_inference_executed\":false}\n"; require(bool(report), "Q6 receipt write failed");
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 2; }
}
