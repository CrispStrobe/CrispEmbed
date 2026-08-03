# Intended purpose, limits, and acceptable use

CrispEmbed is a **software component**, not a finished AI system. This document
states what it is intended for, what it must not be used for, and which
obligations transfer to you when you build something with it.

It is written so that people integrating CrispEmbed can meet their own
regulatory duties. It is not legal advice, and it is not a compliance
certification — for any regulated deployment, get advice on your specific use.

---

## 1. What CrispEmbed is

A C++/ggml inference library plus a CLI, an HTTP server, and language bindings.
It runs pre-trained models to produce **vectors and text**: embeddings,
retrieval scores, OCR transcriptions, layout boxes, entity spans, cleaned-up
images, and face templates.

**Intended purpose:** retrieval, document understanding, and document
preprocessing — semantic search, RAG, reranking, OCR of documents and formulae
and musical scores, layout analysis, key-information extraction, language
identification, and image cleanup ahead of any of the above.

It ships as free and open-source software under the MIT licence. Under
Art. 2(12) of the EU AI Act, free-and-open-source AI is outside the Act's scope
unless it is placed on the market as a high-risk system, used for a prohibited
practice, or falls under the Art. 50 transparency rules. CrispEmbed is released
as a general-purpose component and is not placed on the market as a high-risk
system.

**The project also operates two things that are not components.** The
[HuggingFace Space](https://huggingface.co/spaces/cstr/CrispEmbed) and the
[WASM demo](https://crispstrobe.github.io/CrispEmbed/) are running AI systems
made available to the public, and for those this project is the deployer —
"we only ship a library" does not describe them. Both are deliberately scoped
to keep that surface small: neither exposes face detection or recognition, and
the WASM demo runs entirely client-side, so no uploaded image reaches a server
we operate. The Space transcribes a math image and deletes the temporary file
immediately; the Gradio/HuggingFace platform layer may cache uploads
independently of the application, which is outside our control and stated on
the Space itself. §8 covers the AI-literacy duty that follows from being a
deployer at all.

## 2. What CrispEmbed is not

It is **not** a system that makes decisions about people. It has no notion of
identity, no user record, no database, no persistence, and no decision
threshold. It returns numbers. Every consequential decision — matched / not
matched, hire / reject, flag / ignore — happens in *your* code, and the
responsibility for it is yours.

There is deliberately **no gallery, enrolment, index, watchlist, or 1:N search
primitive** anywhere in the library or the C ABI. The face API stops at:
detect a face → align it → return one L2-normalized embedding.

## 3. Prohibited uses

Do not use CrispEmbed, in whole or in part, to build or operate:

- **Untargeted scraping of facial images** from the internet or CCTV to create
  or expand a facial-recognition database (EU AI Act Art. 5(1)(e)).
- **Emotion inference** in the workplace or in education (Art. 5(1)(f)).
- **Biometric categorisation** to deduce race, political opinions, trade union
  membership, religion, sex life or sexual orientation (Art. 5(1)(g)).
- **Real-time remote biometric identification** in publicly accessible spaces
  for law-enforcement purposes, outside the narrow exceptions in Art. 5(1)(h).
- **Social scoring**, predictive policing based on profiling, or exploitation of
  the vulnerabilities of a specific group (Art. 5(1)(c), (d), (b)).
- **Generating or manipulating non-consensual intimate imagery, or child sexual
  abuse material.** Added to Art. 5 by the Digital Omnibus on AI
  (Regulation (EU) 2026/1744), with a transitional period ending
  **2 December 2026**. The test is whether such output is a reasonably
  foreseeable and reproducible outcome without significant technical
  modification. This bears directly on the image stack in §5: the restoration
  and super-resolution engines take arbitrary images, and repurposing them
  toward this end is prohibited regardless of what the code makes convenient.
- Covert surveillance, stalking, or identification of people who have not
  consented and have no reasonable expectation of being identified.

CrispEmbed ships no emotion-recognition, biometric-categorisation, age, gender,
or ethnicity model, no nudification or face-swap model, and no scraping tooling.
Those capabilities are absent by design, not by oversight, and pull requests
adding them will be declined.

**That is a statement about the models, not about what the code can be pointed
at.** CLIP and SigLIP score an image against arbitrary text, so a caller who
supplies the labels supplies the classifier — the same primitive that finds "an
invoice" finds a protected attribute, and nothing in the API can tell the two
apart. That makes the deployment a biometric categorisation system under Art.
5(1)(g) even though we ship no such model, and the prohibition lands on whoever
wrote the labels. Zero-shot generality is the feature; this is its cost. Do not
read the paragraph above as a guarantee that the software cannot be misused.

The Art. 5 prohibitions have applied since **2 February 2025** — the NCII/CSAM
one from **2 December 2026** — and none of them are waived by the open-source
exemption. Art. 2(12) puts free-and-open-source AI outside most of the Act;
it does not put it outside Art. 5.

## 4. Face recognition: obligations that transfer to you

The face pipeline (YuNet / SCRFD detection → ArcFace / SFace / AuraFace
recognition) is the highest-risk surface in this project. If you use it:

**A face template is biometric data.** Under GDPR Art. 9 it is special-category
personal data. You generally need an Art. 9(2) condition — usually explicit
consent — *plus* an Art. 6 lawful basis. A DPIA (Art. 35) is likely mandatory.
This applies today, independently of the AI Act.

**1:N identification is regulated.** Building a gallery and searching it is a
biometric identification system: Annex III §1 high-risk under the EU AI Act,
with provider and deployer obligations (risk management, data governance,
logging, human oversight, accuracy and robustness testing, registration,
conformity assessment). The Digital Omnibus on AI —
[Regulation (EU) 2026/1744](https://eur-lex.europa.eu/eli/reg/2026/1744/oj),
published in the Official Journal on 24 July 2026 and in force since 27 July
2026 — defers those obligations to **2 December 2027** for Annex III systems
under Art. 6(2), and to **2 August 2028** for Art. 6(1) systems. 1:1
verification that a person initiates about themselves is treated less strictly,
but is still Art. 9 data.

**Thresholds are yours to set and to justify.** CrispEmbed prints cosine
similarity and no verdict. Face-recognition error rates vary sharply across
demographic groups; a threshold that looks fine on your test set can have a very
different false-match rate on a population you did not measure. Calibrate on
representative data, measure error rates per subgroup, and document both.

To reduce accidental use, loading a face **recognition** model requires a
one-time acknowledgement. It sits in `crispembed_face_init()`, which every
binding funnels through — Python, Rust, Dart FFI — and at every equivalent
point in the CLI, which calls the internal loader directly. All of them key off
the model's own declared `cnn.model_type` rather than its filename, so a
recognition model is caught however it was named; a model that declares no type
at all is treated as a recognition model, so the check fails closed. The
acknowledgement is shared: the CLI's interactive prompt also satisfies the
library. It is satisfied by any of:

| Surface | How to acknowledge |
|---|---|
| CLI / server | `--accept-biometric`, or the interactive prompt on a TTY |
| any process | `CRISPEMBED_ACCEPT_BIOMETRIC=1` |
| C ABI | `crispembed_accept_biometric_use()` |
| Python | `crispembed.accept_biometric_use()` |
| Rust | `crispembed::accept_biometric_use()` |
| Dart / Flutter | `acceptBiometricUse()` |

Without one of these, loading a recognition model fails and prints why —
including for a bare `--dim`, because reading a property off a recognition
model is still loading one. The library never prompts — a library must not read
stdin — so callers that want to ask a human do it themselves and then
acknowledge. Detection alone (bounding boxes, no template) is not gated: a box
is not a template.

This is a speed bump and an audit trail, not a security control — the code is
MIT-licensed and the check is trivially removable. It exists so nobody starts
processing biometric data without noticing.

**Serving face recognition over HTTP.** `crispembed-server` has no
authentication of any kind, and its endpoints read images by *server-side
path* — the client sends `{"image": "/path/on/the/server"}`. On loopback that
is a local tool. On a routable address it is an unauthenticated biometric
endpoint that will turn any file the process can read into a template, so:

- The acknowledgement above is made once, by whoever starts the process. Every
  client that can then reach the port inherits it.
- Starting `--rec` on a non-loopback bind prints a warning naming the address.
  It warns rather than refuses, because containers legitimately bind `0.0.0.0`
  behind a proxy that does the authenticating.
- **`--image-root DIR` confines client-supplied *data* paths to one subtree**:
  the `image` field every endpoint reads, `/preprocess/dewarp`'s `output` — a
  **write**, so unconfined it makes any file this process can write creatable
  or truncatable — and `/pdf/dpi`'s `file`. Paths resolve through `..` and
  symlinks first and compare component-wise, so `/srv/scansEVIL` does not pass
  for a root of `/srv/scans`.
- **`--model-root DIR` confines client-supplied *model* paths**, currently
  `/preprocess/tps-dewarp`'s `model`. Separate on purpose: a model legitimately
  lives outside an image directory, and a GGUF is a graph this process then
  executes, so an unconfined model path is a code-execution surface rather than
  a data one.

Set both whenever the port is not loopback-only. Unset, any readable path is
accepted — the historical behaviour.

None of this substitutes for an authenticating proxy. If you serve /face to
anything other than localhost, the access control is yours to build.

**The acknowledgement is per process, not per request, and `crispembed-server`
has no authentication.** Whoever starts the server acknowledges once; every
client that can then reach the port inherits it. The image endpoints — `/face`
and `/detect` included — take a **server-side file path**, not an upload, so a
client that can reach the port can have the server read any file the process
can. Bound to loopback, which is the default, that is a local tool. Bound to a
routable address with `--host`, it is an open biometric endpoint: put an
authenticating reverse proxy in front of it, restrict what the process can read,
and do not expose it directly. The server warns at startup when a recognition
model is loaded on a non-loopback bind. That warning is a reminder, not a
control — it does not authenticate anyone.

If you deploy that way you are processing special-category data on behalf of
whoever's faces reach it, with the GDPR duties in this section attached, and the
access log of the proxy you put in front is likely the only record that any of
it happened.

## 5. Generated and modified content

CrispEmbed's preprocessing stack runs from classical operations (deskew,
binarize, dewarp, crop) through neural ones (ESRGAN, SwinIR, HAT, DAT, SAFMN,
TBSRN super-resolution; NAFNet, SCUNet, Restormer, InstructIR denoise and
restore). All of them change pixels. The neural ones additionally synthesise
detail that was not in the input.

**We treat the whole stack the same way, and the engine is not the unit of
analysis.** Art. 50(2) does not apply where a system performs an assistive
function for standard editing *or* does not substantially alter the input data
or its semantics (Recital 134) — the test is disjunctive. Deskewing a scan and
upscaling it are both standard editing in service of legibility, and neither
changes what the document says. A rotation resamples every pixel too; the line
between "resampling" and "synthesising" does not track the line the Article
draws. So there is no coherent reading on which deskew is exempt and
super-resolution is not, when both are applied to make a document readable.

**Read that as a reasoned position, not a settled exemption.** It has not been
tested by a regulator or a court, and it is strongest exactly where this project
aims — a scanned document, restored to be read. It gets weaker as you move away
from that. Nothing in the code enforces the document framing: `/esrgan/sr`,
`/swinir/sr`, `/hat/sr` and the rest accept any image, and a restoration model
run over a photograph of a person and then published is the case a regulator
would look at first. "Document preprocessing" is our declared intended purpose;
it is not a constraint the software imposes on you.

What *does* matter is the use, and that is a deployer question rather than a
property of the code. If you publish content depicting real people, places, or
events, the Art. 50(2) marking duty and the Art. 50(4) deep-fake disclosure duty
may fall on you regardless of which engine produced it. **Art. 50 is in force —
it applied from 2 August 2026, which is in the past.** This is no longer a
deadline to plan for. The only thing still running is the Digital Omnibus grace
period for the Art. 50(2) *machine-readable marking* on systems already on the
market before that date, which ends **2 December 2026**.

CrispEmbed adds no marking **by default**, and the argument above is why: for
the document case we do not think Art. 50(2) engages. That remains a reasoned
position, not a resolved question, and it is exactly as untested as this
section says.

What has changed is that every image CrispEmbed **returns to you** now carries
provenance by default. (Internal temporaries — a cleaned page handed between
pipeline stages — are unmarked and deleted; they are not output.) Output is
PNG rather than raw Netpbm — Netpbm has no metadata container and C2PA has no
PPM binding — and every image gets a PNG `iTXt` chunk naming the engine that
touched the pixels. `CRISPEMBED_IMAGE_FORMAT=ppm` restores the old raw output
(with only the header-comment marking of the previous scheme):

```
generated=true
software=CrispEmbed
engine=esrgan-sr
digitalSourceType=http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicallyEnhanced
note=AI-processed image. Not an authentic record of the original; restored or
  upscaled detail is a plausible completion, not recovered information.
policy=https://github.com/CrispStrobe/CrispEmbed/blob/main/POLICY.md
```

The IPTC term is **`algorithmicallyEnhanced`**, not `trainedAlgorithmicMedia`.
The input is a real capture that we enhanced; asserting wholly-synthetic media
would be false, and a false provenance claim is a worse position than none.

**Content Credentials (C2PA).** When a signing identity is configured —
`CRISPEMBED_C2PA_CERT` and `CRISPEMBED_C2PA_KEY` — the PNG additionally carries
a signed C2PA manifest asserting `c2pa.edited` with the same source type. Build
with `-DCRISPEMBED_C2PA_FETCH=ON` to pull the c2pa-rs native library; without it
images are still PNG with `iTXt`, just unsigned.

**We ship no signing key, deliberately.** A private key published in an MIT
repository would let anyone mint a manifest naming CrispEmbed as the software
agent for an image it never touched, and re-sign after altering the pixels —
destroying both jobs a C2PA signature exists to do while looking like it does
them. That is a worse outcome than an unsigned image. `scripts/make-c2pa-cert.sh`
generates a per-installation chain if you want signing without sourcing a
certificate; note that a **self-signed certificate is rejected by c2pa-rs**, so
it builds a leaf + local CA. Verifiers will show *unverified signer* either way:
such a manifest attests what was done, not who did it. For attributable
provenance you need a certificate from a CA on the C2PA trust list.

The engine is named rather than a bare "AI-processed", because that flag alone
does not tell a reader whether detail was **synthesised** (ESRGAN, NAFNet,
SCUNet) or merely **resampled** (deskew, dewarp) — a distinction this section
argues matters and that nobody can recover from the pixels afterwards.

**Know what the unsigned level is not.** An `iTXt` chunk is metadata, not a
signature: strippable, with no cryptographic binding to the pixels, and lost by
any conversion that drops ancillary chunks. It satisfies the *marking* Art.
50(2) asks for; it proves nothing. The signed level adds tamper-evidence over
the pixels, but only identity you can trace to a trusted CA makes it
attributable. Do not present either as proof that an image is what it claims.

Independently of the AI Act: do not present restored or upscaled imagery as an
authentic record of the original. Upscaling a licence plate or a face does not
recover information that was never captured — it invents a plausible completion.
Treating that output as evidence is unsound whatever the disclosure rules say.

## 6. OCR accuracy

OCR output is a probabilistic reconstruction, not a faithful copy. VLM-based
engines can hallucinate plausible text. Do not use CrispEmbed output as the sole
basis for a decision about a person — benefits, credit, employment, immigration,
medical or legal outcomes — without human review of the source document. Several
of those uses are Annex III high-risk in their own right.

**Transcription is not generation, but the VLM engines blur that.** We read
transcription as outside Art. 50(2): the output is meant to reproduce what the
document already says, which is the Recital 134 case of not substantially
altering the input or its semantics. That reading is comfortable for the CTC and
attention recognisers, and thinner for the VLM-based engines, which are language
models and will confabulate through a smudge rather than leave it blank. If you
publish OCR output as text in its own right — rather than using it as an index
over a source document a reader can still consult — treat it as model output and
say so. Art. 50(4) additionally requires disclosure for AI-generated text
published to inform the public on matters of public interest. As in §5,
CrispEmbed marks nothing for you.

## 7. Models and licences

Converting a checkpoint to GGUF does not relicense it. Each model remains under
its upstream licence, some non-commercial or vendor-restricted. See the
[Model licences](README.md#model-licenses) table, `crispembed --list-models`,
and `tests/check_registry_licenses.py`.

**A fine-tune's declared licence is not evidence about its base.** Whoever
re-hosts is relying on a chain, and the chain is where it breaks. Two failure
modes we have actually hit in this registry:

- The tag is in the wrong field. HuggingFace records custom licences in
  `license_name`, not `license`, so a checker that reads only `license` sees
  nothing and reports a permissive default. `Qwen2.5-VL-3B-Instruct` is
  `qwen-research` — research-only — while the 7B and Qwen2-VL-2B are
  Apache-2.0. A "Qwen2.5-VL fine-tune" tagged Apache-2.0 is therefore worth
  checking rather than believing.
- The card is simply wrong. `german-ocr-3.1` credits a nonexistent "Qwen3.5",
  and our own registry described it as a Qwen2.5-VL fine-tune. Neither was
  reliable. Its GGUF tensor shapes — hidden 1536, 28 layers — identify
  Qwen2-VL-2B-Instruct (Apache-2.0), so the Apache-2.0 claim does hold; but it
  held by luck, not because anyone had checked.

Read the weights, not the prose, when the answer decides whether commercial use
is permitted.

Re-hosted GGUFs at `huggingface.co/cstr` are quantized derivatives. Quantization
is not a substantial modification, so the EU AI Act Art. 53 obligations for
general-purpose AI models remain with the original model providers; consult the
upstream model card for training data, capabilities, and limitations.

The prior argument is the fallback. The first one is that Art. 53 does not
engage at all: almost everything in the registry is a task-specific model — an
embedder, a detector, a recogniser — and a model that does one task is not a
general-purpose AI model however it was trained. Where a re-host *is* a
quantization of a general-purpose model, the compute test settles it, since
quantization adds no training compute and cannot cross the threshold at which a
downstream modifier becomes the provider.

Worth knowing if you re-host models yourself: the open-source route out of Art.
53 is narrower than the one in Art. 2(12). Art. 53(2) relieves a free and
open-source model provider of the technical-documentation duties, but **not** of
the copyright policy or the public training-data summary, and none of it applies
to a model with systemic risk. Publishing weights under a permissive licence is
therefore not by itself a complete answer.

**Model payloads are pinned.** A GGUF is a graph plus weights that the process
then executes, so "the download succeeded" is not an integrity statement. Every
auto-download URL in the registry carries a SHA-256 pin in
`examples/cli/model_hashes.h`, generated by `tools/fetch_model_hashes.py` from
HuggingFace's LFS object IDs. A payload whose digest does not match its pin is
deleted rather than installed, so a swapped or tampered re-host fails closed
instead of being loaded. Downloads over plain HTTP are refused outright. A URL
with no pin is also refused, overridable for a one-off with
`CRISPEMBED_ALLOW_UNPINNED_MODEL=1`, which prints that the payload is
unverified. This is a supply-chain control, not an AI Act obligation in itself,
but it is a precondition for any Art. 15 accuracy/robustness claim a downstream
integrator wants to make: you cannot attest to the behaviour of weights you did
not verify you received.

## 8. AI literacy (Art. 4)

Art. 4 has applied since **2 February 2025** and binds providers *and*
deployers: staff and others operating an AI system on your behalf need a level
of AI literacy appropriate to the system, the context, and the people affected.

The Art. 2(12) open-source exemption does not reach this for the two systems in
§1 that this project actually deploys. It also does not reach *you*. If you
integrate CrispEmbed, the duty attaches to your deployment, not to our library,
and nothing we ship discharges it.

The unglamorous reading of Art. 4 for a component like this: the people running
it should understand that OCR output is a reconstruction and not a copy (§6),
that a cosine similarity is not a match decision (§4), that face-recognition
error rates vary by demographic group and a threshold calibrated on one
population does not transfer to another (§4), and that a restored image is a
plausible completion rather than recovered evidence (§5). Those are the
misunderstandings that turn this software into a harm. They are stated
throughout this document for that reason, and repeated in the README, the
Python and Dart package pages, and the runtime warning printed when a
recognition model is loaded — so the people operating the thing meet them
without having to find this file.

## 9. Reporting misuse

If you find CrispEmbed being used for anything in §3, or you believe a shipped
model or example makes such use easier than it should be, please open an issue
at <https://github.com/CrispStrobe/CrispEmbed/issues>.

---

*Regulatory dates reflect Regulation (EU) 2024/1689 (the AI Act) as amended by
the Digital Omnibus on AI, [Regulation (EU)
2026/1744](https://eur-lex.europa.eu/eli/reg/2026/1744/oj) — published in the
Official Journal on 24 July 2026, in force since 27 July 2026. Last checked
against the OJ text on 1 August 2026:*

| Date | What applies |
|---|---|
| 2 February 2025 | Art. 5 prohibitions, and Art. 4 AI literacy (both in force) |
| 2 August 2025 | General-purpose AI model obligations (in force) |
| 2 August 2026 | Art. 50 transparency obligations (**in force — this date has passed**) |
| 2 December 2026 | End of the Art. 50(2) marking grace period for systems already on the market; end of the NCII/CSAM prohibition transitional period |
| 2 December 2027 | Annex III high-risk obligations (Art. 6(2) systems) |
| 2 August 2028 | High-risk obligations for Art. 6(1) systems (AI in regulated products) |

*Verify against the current OJ text before relying on any of this. Dates have
moved once already and the amending regulation is recent.*
