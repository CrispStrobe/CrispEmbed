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
- Covert surveillance, stalking, or identification of people who have not
  consented and have no reasonable expectation of being identified.

CrispEmbed ships no emotion-recognition, biometric-categorisation, age, gender,
or ethnicity model, and no scraping tooling. Those capabilities are absent by
design, not by oversight, and pull requests adding them will be declined.

The Art. 5 prohibitions have applied since **2 February 2025** and are not
waived by the open-source exemption.

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
conformity assessment). Following the Digital Omnibus adopted in June 2026,
those obligations apply from **2 December 2027**. 1:1 verification that a person
initiates about themselves is treated less strictly, but is still Art. 9 data.

**Thresholds are yours to set and to justify.** CrispEmbed prints cosine
similarity and no verdict. Face-recognition error rates vary sharply across
demographic groups; a threshold that looks fine on your test set can have a very
different false-match rate on a population you did not measure. Calibrate on
representative data, measure error rates per subgroup, and document both.

To reduce accidental use, running a face **recognition** model requires a
one-time acknowledgement: `--accept-biometric`, `CRISPEMBED_ACCEPT_BIOMETRIC=1`,
or an interactive prompt. Detection alone (bounding boxes, no template) is not
gated. This is a speed bump and an audit trail, not a security control — the
code is MIT-licensed and the check is trivially removable. It exists so nobody
starts processing biometric data without noticing.

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

What *does* matter is the use, and that is a deployer question rather than a
property of the code. If you publish content depicting real people, places, or
events, the Art. 50(2) marking duty and the Art. 50(4) deep-fake disclosure duty
may fall on you regardless of which engine produced it — those rules apply from
**2 August 2026**. CrispEmbed adds **no watermark or C2PA provenance marking**
to any output, so if you need marking you must add it yourself.

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

## 7. Models and licences

Converting a checkpoint to GGUF does not relicense it. Each model remains under
its upstream licence, some non-commercial or vendor-restricted. See the
[Model licences](README.md#model-licenses) table, `crispembed --list-models`,
and `tests/check_registry_licenses.py`.

Re-hosted GGUFs at `huggingface.co/cstr` are quantized derivatives. Quantization
is not a substantial modification, so the EU AI Act Art. 53 obligations for
general-purpose AI models remain with the original model providers; consult the
upstream model card for training data, capabilities, and limitations.

## 8. Reporting misuse

If you find CrispEmbed being used for anything in §3, or you believe a shipped
model or example makes such use easier than it should be, please open an issue
at <https://github.com/CrispStrobe/CrispEmbed/issues>.

---

*Regulatory dates reflect the EU AI Act as amended by the Digital Omnibus on AI
adopted in June 2026. Verify current deadlines before relying on them.*
