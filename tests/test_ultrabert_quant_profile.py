"""Source-only closure checks; compiled metadata and real comparisons stay hosted."""
from pathlib import Path
import subprocess
import unittest
ROOT=Path(__file__).resolve().parents[1]
PROVEN='02402a0d086a7335280d907392c4f4667e27ce28'
class Controls(unittest.TestCase):
    # Reviewed default-preserving amendment (commit 59ecb0b9, repo-wide core_env
    # audit): only these exact byte substitutions are admitted; any other drift
    # from the proven tokenizer source still fails. CRISPEMBED_BPE_LEGACY_WHITESPACE
    # is unset in every UltraBERT run, so both forms select the same path.
    AMENDMENTS={'src/tokenizer_bpe.cpp':[
        (b'#include <vector>\n\nbool BPETokenizer::load(',b'#include <vector>\n#include "core/env_gate.h"\n\nbool BPETokenizer::load('),
        (b'static const bool legacy_ws = (std::getenv("CRISPEMBED_BPE_LEGACY_WHITESPACE") != nullptr);',
         b'static const bool legacy_ws = core_env::on("CRISPEMBED_BPE_LEGACY_WHITESPACE");')]}
    # Reviewed whole-commit re-pin (recorded in tests/results/frozen-guards/ultrabert-wasm.json):
    # 0fb9e738 adds default-off tokenizer setters (set_max_length, set_pair_double_sep)
    # and a pair_double_sep_ member (default false) used only by encode_pair. UltraBERT's
    # own BPETokenizer never calls either setter or encode_pair, so encode() is unchanged.
    # The parent of each re-pin commit must equal the proven (+amended) bytes exactly,
    # and the checkout must equal the re-pin commit's bytes exactly.
    REPINS={'src/tokenizer.h':'0fb9e738722c97fb5183373008f61f60bda62961',
            'src/tokenizer_bpe.cpp':'0fb9e738722c97fb5183373008f61f60bda62961'}
    def test_exact_already_proven_tokenizer_sources(self):
        for name in ['src/tokenizer.h','src/tokenizer_bpe.cpp','src/ultrabert_tokenizer_mode.h']:
            expected=subprocess.check_output(['git','show',PROVEN+':'+name],cwd=ROOT)
            for old,new in self.AMENDMENTS.get(name,[]):
                self.assertEqual(expected.count(old),1,name);expected=expected.replace(old,new)
            if name in self.REPINS:
                commit=self.REPINS[name]
                self.assertEqual(subprocess.check_output(['git','show',commit+'^:'+name],cwd=ROOT),expected,name)
                expected=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
            self.assertEqual((ROOT/name).read_bytes(),expected,name)
    def test_truncation_body_retained(self):
        old=subprocess.check_output(['git','show',PROVEN+':src/ultrabert_profile.h'],cwd=ROOT,text=True)
        now=(ROOT/'src/ultrabert_profile.h').read_text(); marker='inline bool tokenize('
        self.assertEqual(old[old.index(marker):],now[now.index(marker):])
    def test_quantized_gate_separate_from_legacy_profile(self):
        text=(ROOT/'src/ultrabert_embed.cpp').read_text()
        self.assertIn('core_env::on("CRISPEMBED_ULTRABERT_QUANT")',text)
        self.assertIn('ultrabert_profile::validate(meta.get(), quantized)',text)
        self.assertIn('bool quantized = false',(ROOT/'src/ultrabert_profile.h').read_text())
    def test_actual_q4_and_critical_guards_not_global_relaxation(self):
        text=(ROOT/'src/ultrabert_profile.h').read_text()
        for required in ['core_gguf_file_type::known_q4_k','GGML_FTYPE_ALL_F32','item.first == "token_embd.weight"','!critical','matches_quantized_payload','actual_q8','actual_q4']:
            self.assertIn(required,text)
if __name__=='__main__': unittest.main()
