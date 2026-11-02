# UltraBERT retained F32 freeze reviewed

[Run40834743025](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40834743025)
completed at source08b0a5727aa25412a1181db4695efb01be578b24 with **zero original
or F32 forwards**. The [independent review receipt](../tests/results/ultrabert/retained-freeze-review.json)
binds the154,743-byte small ZIP, all13 member identities and49,048-byte freeze
receipt. The132,645,918-byte payload ZIP and all77 member identities remain
remote. Runtime018, the old37/08 sources and frozen original references remain
unchanged.

All5,483 saved F32 rows, their original IDs and48 stream/adapter identities were
authenticated in the hosted freeze. Eight actual Ref comparisons cover raw and
normalized outputs over the original eight strided shards. Both fixed0.002/
0.99999 gates pass: raw maximum0.001098036766052246/minimum cosine
0.9999999968249437; normalized maximum1.0911375284194946e-5/minimum cosine
0.9999999968248703. Positive, numeric-negative and wrong-shape controls retain
exit0/1/2. The actual Ultra object symbol/string exclusion passes; the full
historical library symbol inventory remains available separately.

The hosted full300-query SciFact rankings match the original: NDCG@10
0.7667047800710401, MRR@10 0.7290066137566137, Recall@100 0.975 and top1 agreement
1.0. Independent review checked reported scalars, all eight Ref/independent
records, source/runtime objects, compiler flags, package versions and small
inventories. It did not download vectors, rerank locally or run a model.

This is F32 freeze evidence. **No quantized row or quantized retrieval result**
exists here. The next conservative recipe remains twelve genuine Q4_K FFN
tensors in layers20–23 plus206 unchanged F32 tensors. Conversion must first
produce an exact reviewed model SHA and artifact receipt; only then may a
separately activated shard0 run. The old018 quantizer's legacy file-type header
is explicitly unsuitable for standard-header publication. Q8 needs its own
metadata-aware source/protocol. A first-shard result cannot qualify full-corpus
Q4, Q8, a release, an application pin or a default change.
