# MinerU conversion contract fixtures

These JSON files come from the public
[MinerU2.5-Pro-2604-1.2B snapshot](https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B/tree/d3f5e08d073c21466bbabe21c71bb1e9c2e595da).
The checkpoint and its metadata are Apache-2.0 licensed.

`safetensors-header.json` is only the 73,384-byte header of the original
`model.safetensors`, fetched with HTTP byte ranges. It describes 681 tensors
in a 2,312,126,640-byte checkpoint. No weight data is included. These independent
upstream shapes check the converter's full mapping without loading the model.

The 11 MB tokenizer and model payload remain download-time dependencies.
The converter checks all 151,674 tokenizer entries, all 31 added special tokens,
and the original 151,387 BPE merges when run on the source snapshot.
