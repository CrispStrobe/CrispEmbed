import re
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


class MainLinkContract(unittest.TestCase):
    def test_every_attention_library_also_owns_helper(self):
        code=(ROOT/'CMakeLists.txt').read_text()
        main=code[:code.index('add_library(crispembed STATIC')]
        self.assertRegex(main,r'list\(APPEND CRISPEMBED_SOURCES src/lighton4b_physical_f16_attention\.cpp\)')
        self.assertIn('target_sources(crispembed-qwen35-delta PRIVATE src/lighton4b_physical_f16_attention.cpp)',code)
        definitions=re.findall(r'target_compile_definitions\(([^\n]*CRISPEMBED_LIGHTON4B_F16_SGEMM_ABI[^\n]*)',code)
        self.assertTrue(any(row.startswith('crispembed-qwen35-delta PRIVATE ')for row in definitions))
        self.assertFalse(any(row.startswith('crispembed ')for row in definitions))
        # Ignore source-property mentions; only main's source list and the
        # separate delta add_library directly consume the attention source.
        consumers=re.findall(r'(?:add_library|list)\([^)]*src/qwen35_attention\.cpp[^)]*\)',code,re.S)
        self.assertEqual(len(consumers),2)
        self.assertTrue(any(x.startswith('list(APPEND CRISPEMBED_SOURCES')for x in consumers))
        self.assertTrue(any(x.startswith('add_library(crispembed-qwen35-delta ')for x in consumers))
        excluded=code.split('set(CRISPEMBED_WASM_EXCLUDED_SOURCES',1)[1].split(')',1)[0]
        self.assertNotIn('lighton4b_physical_f16_attention.cpp',excluded)
        self.assertNotIn('qwen35_attention.cpp',excluded)


if __name__=='__main__':unittest.main()
