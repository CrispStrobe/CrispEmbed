#!/usr/bin/env python3
"""Reference-only loader metadata observer for exact upstream 6c01 source.

Apply alongside lighton4b_upstream_observer_patch, before building the reference
server. This reads metadata only after weights load/repack; it neither reads nor
writes tensor payloads and adds no graph, callback, sampler or trait changes.
"""
import hashlib
from pathlib import Path
SOURCE = '6c01da6278ab6c69ed67d9124806b50d2b299ba7'
FILE = 'src/llama-model.cpp'
SHA256 = 'c41727dd626ef2e68e6c2af82bfb2600f3c77222b70f391e214b1c9368157290'
ENV = 'CRISPEMBED_TEST_UPSTREAM_DISPATCH_DIR'
OUTPUT = 'source.tensor-dispatch.tsv'
ANCHOR = '    load_tensors_hadamard();\n\n    return true;\n}'
INCLUDES = '#include <fstream>\n#include <cstdlib>\n'
INSERT = r'''
    // Reference-only: after every load/repack and Hadamard setup, before any
    // model forward. Do not dereference or modify tensor payloads or extra.
    if (const char * directory = std::getenv("CRISPEMBED_TEST_UPSTREAM_DISPATCH_DIR")) {
        if (!*directory) throw std::runtime_error("dispatch observer requires nonempty output directory");
        if (!tok_embd || !output || !tok_embd->data || tok_embd->ne[0] != 2560 ||
            tok_embd->ne[1] != 248320 || tok_embd->ne[2] != 1 || tok_embd->ne[3] != 1) {
            throw std::runtime_error("dispatch observer requires pinned LightOn4B text geometry");
        }
        std::ofstream probe(std::string(directory) + "/source.tensor-dispatch.tsv");
        if (!probe) throw std::runtime_error("dispatch observer output unavailable");
        probe << "role\tname\ttype\ttype_name\tne0\tne1\tne2\tne3\tbuffer\tbuffer_type\thost\textra\tis_embedding_header\tis_output_header\tshares_embedding_data\n";
        auto field = [](const char * value) {
            const std::string text = value ? value : "<none>";
            if (text.find_first_of("\t\r\n") != std::string::npos) {
                throw std::runtime_error("dispatch observer field is not a single TSV value");
            }
            return text;
        };
        auto record = [&](const char * role, const ggml_tensor * tensor) {
            if (!tensor || !tensor->data || !tensor->buffer) {
                throw std::runtime_error("dispatch observer requires fully loaded tensor metadata");
            }
            const auto buffer = tensor->buffer;
            const auto buft = ggml_backend_buffer_get_type(buffer);
            if (!buft) throw std::runtime_error("dispatch observer missing buffer type");
            probe << role << '\t' << field(ggml_get_name(tensor)) << '\t' << int(tensor->type)
                  << '\t' << field(ggml_type_name(tensor->type));
            for (int axis = 0; axis < 4; ++axis) probe << '\t' << tensor->ne[axis];
            probe << '\t' << field(ggml_backend_buffer_name(buffer))
                  << '\t' << field(ggml_backend_buft_name(buft))
                  << '\t' << int(ggml_backend_buffer_is_host(buffer))
                  << '\t' << int(tensor->extra != nullptr)
                  << '\t' << int(tensor == tok_embd) << '\t' << int(tensor == output)
                  << '\t' << int(tensor->data == tok_embd->data) << '\n';
        };
        const char * requested[] = {
            "blk.0.attn_norm.weight", "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",
            "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_out.weight",
            "blk.0.ffn_gate.weight"
        };
        for (const auto * name : requested) {
            const ggml_tensor * found = nullptr;
            for (const auto & entry : tensors_by_name) {
                if (entry.first == name) {
                    if (found) throw std::runtime_error("dispatch observer tensor name is ambiguous");
                    found = entry.second;
                }
            }
            if (!found) throw std::runtime_error("dispatch observer tensor name is absent");
            record(name, found);
        }
        record("embedding", tok_embd);
        record("output_head", output);
        probe.flush();
        if (!probe) throw std::runtime_error("dispatch observer metadata write failed");
    }
'''

def patch(root):
    path = Path(root) / FILE
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != SHA256:
        raise ValueError('pinned loader observer source differs: ' + FILE)
    text = original.decode('utf-8')
    if text.count(ANCHOR) != 1:
        raise ValueError('exact post-load observer insertion anchor differs')
    text = INCLUDES + text.replace(ANCHOR, '    load_tensors_hadamard();\n' + INSERT + '\n    return true;\n}', 1)
    if text.removeprefix(INCLUDES).replace(INSERT, '', 1) != original.decode('utf-8'):
        raise ValueError('observer patch changes bytes beyond the declared metadata insertion')
    path.write_text(text)
    return {'source_commit': SOURCE, 'file': FILE, 'original_sha256': SHA256,
            'patched_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'environment': ENV, 'output': OUTPUT, 'rows': 9,
            'scope': 'post-load metadata only; no payload access or graph/math/trait changes'}
