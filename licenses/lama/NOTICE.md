# Big-LaMa source adaptation and weight evidence

The native files `src/lama.cpp`, `src/lama_fft.cpp` and their headers implement
the inference architecture and Fourier semantics inspected in
[`advimman/lama`](https://github.com/advimman/lama/tree/786f5936b27fb3dacd2b1ad799e4de968ea697e7),
revision `786f5936b27fb3dacd2b1ad799e4de968ea697e7`.

The original Apache-2.0 licence is retained unchanged in
`upstream-Apache-2.0.txt`, including the original notice:

```text
Copyright [2021] Samsung Research
```

The C++ adaptation changes execution to CrispEmbed CPU/ggml operations and
native real FFTs, adds GGUF loading and bounded inference input validation,
and excludes training, evaluation networks and Python framework integration.
Keep this notice and the upstream licence with source distributions and the
notices accompanying binaries containing this adaptation. CrispEmbed's own
original contributions remain subject to the root repository licence.

No model weights are included here. The official README links to the
`smartywu/big-lama` archive. In the
[upstream weight-licence discussion](https://github.com/advimman/lama/issues/230#issuecomment-1817604197),
a collaborator points to the repository's Apache-2.0 licence in response to
a question explicitly asking whether the weights use it. The pinned archive
does not include a dedicated weight-licence text. The source contract and
conversion sidecars preserve this evidence and its limitation; they do not
assert an independently documented weight grant.

The exact source files, archive digest, original configuration and generator
census are recorded in `docs/lama-source-contract.json`. A hosted conversion
also emits a licence/evidence directory and a receipt for its actual GGUF.
