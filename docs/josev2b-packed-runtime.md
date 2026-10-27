# Josev packed CPU/F32 runtime

Prepared source only. No public ABI, model alias, actual packed inference or
release qualification follows from a model-free build. The original three-file
composition is a separate frozen producer; retain its failed artifact adapter
and corrected actual execution before accepting packed equivalence.

The distinct `josev2b_systemone` architecture contains exactly442 F32 tensors:
320 `l.*` backbone and122 `head.*` tensors, plus embedded tokenizer metadata.
`josev2b.packed.format=crispembed-josev-packed-f32-v1` and profile
`official-pytorch-text-batch1-512-no-extra-temperature` pin the source and all
three reviewed component-file hashes. The original standalone head122/tokenizer0
loaders keep their strict architectures and counts.

`composed_context::load_packed()` requires the packed gate and existing component
gates. It checks bounded8GiB file/64MiB header, metadata/profile, exact shapes,
F32 storage and320/122 ownership before inference. One mapped owner holds all
weights. The head borrows contiguous host tensors and owns only its CPU backend;
head destruction precedes parent tensor/buffer destruction. The frontend copies
its token metadata. Packed disablement clears output through the existing
transactional response path. No input IDs/hidden vectors are supplied externally.

The new metadata phase `josev-packed-build` compiles portable ordinary/observed
standalone libraries with two workers. It checks four malformed header classes,
independent122 tensor shapes, missing/wrong-shape rejection, refusal to borrow
metadata-only pointers, retained legacy loader rejection and production observer
exclusion. These controls use only small owned metadata fixtures, no weights.
Existing tokenizer/inference and head graph bodies stay byte-identical.

Next, only after independently reviewed actual composition and streaming
converter proof, declare a new explicit packed audit. Repack immutable components
remotely, rehash all442 tensors/readback, run the same227 original stages and
complete raw/plain/observed/repeat typed-output gates, then actual dedicated
C/Python/CLI/HTTP/Rust/Dart surfaces. Keep current producers and source guards
unchanged; no broad model/default/media/quality claim. No builds, model payloads,
frameworks or inference belong on the shared source host.
