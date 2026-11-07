// Bounded physical Q4_K/Q6_K decode audit. No graph, model forward or F32 model output.
#include "ggml.h"
#include "core/sha256.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
int main(int argc, char ** argv) {
    try {
        const uint16_t endian = 1; require(*reinterpret_cast<const uint8_t *>(&endian) == 1 && sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "little-endian IEEE F32 required");
        require(argc == 4, "expected physical-model plan.tsv receipt.tsv");
        std::ifstream model(argv[1], std::ios::binary), plan(argv[2]); std::ofstream output(argv[3]);
        require(bool(model) && bool(plan) && bool(output), "audit inputs unavailable");
        model.seekg(0, std::ios::end); const uint64_t size = uint64_t(model.tellg());
        std::string name; int type; uint64_t offset, elements, columns; size_t tensors = 0;
        while (plan >> name >> type >> offset >> elements >> columns) {
            require(++tensors <= 249 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K) && elements > 0 &&
                    elements <= 635699200 && columns > 0 && columns <= 9216 && columns % 256 == 0 &&
                    elements % columns == 0, "bounded exact physical tensor geometry required");
            const auto kind = ggml_type(type); const auto * traits = ggml_get_type_traits(kind);
            require(traits && traits->to_float && ggml_blck_size(kind) == 256, "native block decoder missing");
            const size_t width = ggml_type_size(kind); const uint64_t bytes = elements / 256 * width;
            require(offset < size && bytes <= size - offset, "physical span truncated");
            std::vector<uint8_t> packed(65536 / 256 * width); std::vector<float> values(65536);
            core_sha256::sha256_state packed_hash, float_hash; double squares = 0; float maximum = 0;
            model.seekg(offset);
            for (uint64_t begin = 0; begin < elements; begin += 65536) {
                const size_t count = size_t(std::min<uint64_t>(65536, elements - begin)), n = count / 256 * width;
                model.read(reinterpret_cast<char *>(packed.data()), std::streamsize(n)); require(bool(model), "physical chunk truncated");
                traits->to_float(packed.data(), values.data(), count);
                for (size_t i = 0; i < count; ++i) { require(std::isfinite(values[i]), "nonfinite native decoded weight"); maximum = std::max(maximum, std::fabs(values[i])); squares += double(values[i]) * values[i]; }
                core_sha256::sha256_update(packed_hash, packed.data(), n);
                core_sha256::sha256_update(float_hash, reinterpret_cast<const uint8_t *>(values.data()), count * sizeof(float));
            }
            output << name << '\t' << type << '\t' << elements << '\t' << core_sha256::sha256_finish(packed_hash) << '\t'
                   << core_sha256::sha256_finish(float_hash) << '\t' << std::setprecision(17) << maximum << '\t' << squares << '\n';
        }
        require(tensors == 249 && plan.eof() && bool(output), "all249 physical quantized matrices required");
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 2; }
}
