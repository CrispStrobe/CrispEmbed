"""Source-only closure checks; compiled metadata and real comparisons stay hosted."""
from pathlib import Path
import subprocess
import unittest
ROOT=Path(__file__).resolve().parents[1]
PROVEN='02402a0d086a7335280d907392c4f4667e27ce28'
class Controls(unittest.TestCase):
    def test_exact_already_proven_tokenizer_sources(self):
        for name in ['src/tokenizer.h','src/tokenizer_bpe.cpp','src/ultrabert_tokenizer_mode.h']:
            expected=subprocess.check_output(['git','show',PROVEN+':'+name],cwd=ROOT)
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
