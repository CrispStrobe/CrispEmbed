"""Source-bound, build-only quantization experiment; production sources stay intact.

This creates a private alternate library, never installed into an F32 capsule.
Only F16, Q8_0 and Q4_0 storage are admitted, behind an additional default-OFF
environment gate. Quantized rows expand once into owned finite F32 weights;
the accepted numerical implementation and scalar execution remain unchanged.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

PIPELINE_SHA = '9725a3e16ed04ccb2fe34fc6d2a10390427303bf1b6391d20aa53260ce3bcde1'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Experimental overlay no longer matches its reviewed source')
    return text.replace(old, new)


def pipeline(source):
    if hashlib.sha256(source.encode()).hexdigest() != PIPELINE_SHA:
        raise ValueError('F32 source differs from the frozen baseline')
    source = replace_once(source, '    size_t bytes;\n};\n\nstruct Profile',
                          '    size_t bytes;\n    ggml_type type;\n};\n\nstruct Profile')
    source = replace_once(source, '    size_t data_offset;\n    std::map',
                          '    size_t data_offset;\n    bool quantized = false;\n    std::map')
    source = replace_once(source, '    require(reader.u64() == 13, "Unexpected spaCy GGUF metadata inventory");',
        '''    const uint64_t metadata_count = reader.u64();
    const bool candidate = metadata_count == 15;
    require(metadata_count == 13 || (candidate && core_env::on("CRISPEMBED_SPACY_QUANTIZED")),
            "Experimental quantized spaCy storage requires explicit opt-in");
    ggml_type requested = GGML_TYPE_F32;
    uint32_t requested_file_type = 0;''')
    source = replace_once(source, '    for (size_t i = 0; i < 13; ++i) {',
                          '    for (size_t i = 0; i < metadata_count; ++i) {')
    source = replace_once(source, '        if (text != strings.end()) {',
        '''        if (candidate && key == "spacy.precision") {
            require(type == GGUF_TYPE_STRING, "Mistyped quantized precision");
            const auto precision = reader.string(256);
            require(precision == "f16" || precision == "q8_0" || precision == "q4_0", "Unsupported storage profile");
            requested = precision == "f16" ? GGML_TYPE_F16 : precision == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
        } else if (candidate && key == "general.quantization_version") {
            require(type == GGUF_TYPE_UINT32 && reader.u32() == GGML_QNT_VERSION, "Quantization version mismatch");
        } else if (candidate && key == "general.file_type") {
            require(type == GGUF_TYPE_UINT32, "Mistyped quantization file type");
            requested_file_type = reader.u32();
            require(requested_file_type == 1 || requested_file_type == 2 || requested_file_type == 7,
                    "Unsupported quantization file type");
        } else if (text != strings.end()) {''')
    source = replace_once(source, '    Profile profile{};',
        '''    require(!candidate || requested != GGML_TYPE_F32, "Missing experimental storage precision");
    require(!candidate || requested_file_type == (requested == GGML_TYPE_F16 ? 1u : requested == GGML_TYPE_Q8_0 ? 7u : 2u),
            "Quantization file type does not match the storage precision");
    Profile profile{};
    profile.quantized = candidate;''')
    source = replace_once(source, '''        require(reader.u32() == shape.size(), "Incorrect spaCy tensor rank");
        size_t elements = 1;
        for (auto dimension = shape.rbegin(); dimension != shape.rend(); ++dimension) {
            require(reader.u64() == *dimension, "Incorrect spaCy tensor shape");
            elements *= *dimension; // dimensions are fixed by the closed profile
        }''', '''        const size_t rank = reader.u32();
        require(rank == shape.size() || (candidate && shape.front() == 1 && rank + 1 == shape.size()),
                "Incorrect spaCy tensor rank");
        size_t elements = 1;
        size_t axis = 0;
        for (auto dimension = shape.rbegin(); dimension != shape.rend(); ++dimension, ++axis) {
            if (axis < rank) require(reader.u64() == *dimension, "Incorrect spaCy tensor shape");
            elements *= *dimension;
        }''')
    source = replace_once(source, '        require(reader.u32() == GGML_TYPE_F32, "Only F32 spaCy tensors are admitted");\n        const uint64_t offset = reader.u64();\n        const size_t tensor_bytes = elements * sizeof(float);',
        '''        const uint32_t raw_type = reader.u32();
        require(raw_type == GGML_TYPE_F32 || (candidate && raw_type == static_cast<uint32_t>(requested)) ||
                    (candidate && requested == GGML_TYPE_Q4_0 && name.rfind("embed.", 0) == 0 && raw_type == GGML_TYPE_Q8_0),
                "Tensor type is outside the admitted storage profile");
        const auto tensor_type = static_cast<ggml_type>(raw_type);
        const size_t block = static_cast<size_t>(ggml_blck_size(tensor_type));
        require(shape.back() % block == 0, "Quantized row block is incomplete");
        const uint64_t offset = reader.u64();
        const size_t tensor_bytes = elements / block * ggml_type_size(tensor_type);''')
    source = replace_once(source, 'Extent{ shape, static_cast<size_t>(offset), tensor_bytes }',
                          'Extent{ shape, static_cast<size_t>(offset), tensor_bytes, tensor_type }')
    source = replace_once(source, '        for (size_t i = 0; i < extent.bytes; i += sizeof(float)) {',
        '        for (size_t i = 0; extent.type == GGML_TYPE_F32 && i < extent.bytes; i += sizeof(float)) {')
    source = replace_once(source, '    core_gguf::WeightLoad weights;',
                          '    core_gguf::WeightLoad weights;\n    std::map<std::string, std::vector<float>> expanded;')
    source = replace_once(source, '            require(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && ggml_nbytes(tensor) == extent.bytes,',
        '            require(tensor->type == extent.type && ggml_is_contiguous(tensor) && ggml_nbytes(tensor) == extent.bytes,')
    source = replace_once(source, '''            tensors.emplace(item.first,
                            crispembed::spacy::TensorView{ extent.shape, static_cast<const float *>(tensor->data),
                                                           extent.bytes / sizeof(float) });''',
        '''            size_t count = 1;
            for (const auto dim : extent.shape) count *= dim;
            const float * values = static_cast<const float *>(tensor->data);
            if (extent.type != GGML_TYPE_F32) {
                require(profile.quantized && core_env::on("CRISPEMBED_SPACY_QUANTIZED"), "Storage gate is disabled");
                auto & owned = context->expanded[item.first];
                owned.resize(count);
                const auto * traits = ggml_get_type_traits(extent.type);
                require(traits && traits->to_float, "Missing dequantization primitive");
                traits->to_float(tensor->data, owned.data(), static_cast<int64_t>(count));
                for (const float value : owned) require(std::isfinite(value), "Nonfinite expanded weight");
                values = owned.data();
            }
            tensors.emplace(item.first, crispembed::spacy::TensorView{ extent.shape, values, count });''')
    return source


def generate(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.mkdir(exist_ok=False)
    (destination / 'spacy_pipeline.cpp').write_text(pipeline((root / 'src/spacy_pipeline.cpp').read_text()))
    quantizer = (root / 'tools/quantize.cpp').read_text()
    # spaCy's rank-four parser cache is not a convolution: preserve its shape.
    quantizer = replace_once(quantizer, '!patch_f16[i] && !is_oxr) {',
        '!patch_f16[i] && !is_oxr && !(arch_key >= 0 && std::string(gguf_get_val_str(ctx_in, arch_key)) == "spacy_de_core_news_sm")) {')
    quantizer = quantizer.replace('"../src/', '"' + str(root / 'src') + '/')
    (destination / 'quantize.cpp').write_text(quantizer)
    cmake = (root / 'tools/spacy/CMakeLists.txt').read_text()
    cmake = replace_once(cmake, 'get_filename_component(CRISPEMBED_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)',
                         'set(CRISPEMBED_ROOT "' + str(root) + '")')
    cmake = replace_once(cmake, '"${CRISPEMBED_ROOT}/src/spacy_pipeline.cpp"',
                         '"${CMAKE_CURRENT_LIST_DIR}/spacy_pipeline.cpp"')
    cmake += '\nadd_executable(crispembed-quantize "${CMAKE_CURRENT_LIST_DIR}/quantize.cpp")\n'
    cmake += 'target_link_libraries(crispembed-quantize PRIVATE ggml ggml-cpu)\n'
    cmake += 'add_library(spacy-diff SHARED "${CRISPEMBED_ROOT}/tools/spacy/quantized_diff.cpp")\n'
    cmake += 'target_include_directories(spacy-diff PRIVATE "${CRISPEMBED_ROOT}/src")\n'
    (destination / 'CMakeLists.txt').write_text(cmake)
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()}
