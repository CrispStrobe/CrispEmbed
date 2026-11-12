# Retained book repetition and EOS diagnostics

`audit.py` is a bounded artifact-only Kaggle CPU worker. It authenticates three
immutable receipts and their complete lossless raw-row archives before reporting
numerical diagnostics. No model is loaded, no original/native forward is repeated,
and no text or token ID sequence is written into the output report.

Run with the original source book artifact, the planner timeout relay containing
the complete old660-token arm, and the completed1536-token plannerON replay mounted.
The script emits `/kaggle/working/book-repetition-eos.json`; archives remain remote.
Only NumPy and the Python standard library are required. Three small controls test
repetition boundaries, same-prefix indexing and stable EOS softmax/rank arithmetic.

Token-structure screens report duplicate4/8/16-token windows and first/longest
periodic runs of at least3 cycles and12 tokens, testing periods1..64. These are
fixed descriptive screens, not repetition penalties or output-quality gates.
Every raw score row reports EOS rank (strictly-greater count plus1, ties separate),
EOS softmax probabilities, summed EOS probability, top-two score gap and entropy.
The two EOS entries follow the existing stop-slot order declared in the audit;
vocabulary IDs and sampled top tokens are omitted.

If source/native generated IDs first disagree at position K, score rows0..K have
identical consumed prefixes. Rows after K describe different trajectories and
must not be labeled numerical parity comparisons. Shared input pixels/prompt
identity remains inherited from the authenticated historical receipts. No runtime,
recipe, label, cap, scorer or default changes follow from these diagnostics.
