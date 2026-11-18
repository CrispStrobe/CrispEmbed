# W6 first guarded grammar qualification

The first sealed W6 qualification passes on Linux x86-64/AVX2, German `de-DE`,
CPU/F32. Against the updated deterministic Quds baseline, exact corrections
increase from **5/30 to 22/30**, adding **17**. The
[complete first result](https://github.com/CrispStrobe/quds_office_suite/blob/3a8bca4f8f8744bbb0fbd6809a700b5768d5ceba/docs/validation/native-spacy-w6-20261117/qualification-first.json)
is preserved with all eight misses. The optional manual grammar provider API
defaults off, and Studio does not register it automatically.

## Frozen scope and source order

The [original protocol and panel](https://github.com/CrispStrobe/quds_office_suite/tree/846fc0371b23bf4a84542db7a3f24eee2d8609dd/docs/validation/native-spacy-w6-20261117)
contain 30 error inputs, 42 valid inputs, ten protection/adverse controls and
four unsupported locales. This is an assistant-authored finite panel, not an
external corpus or a population-level quality estimate. Its expected outputs
were published before candidate inspection. Development uses different inputs.

The latest baseline is Quds main
[`6245a6956a76a38f925e1eb18851c0086730112d`](https://github.com/CrispStrobe/quds_office_suite/commit/6245a6956a76a38f925e1eb18851c0086730112d),
including German phrase completion and formatting-preserving correction.
The [baseline amendment](https://github.com/CrispStrobe/quds_office_suite/blob/3a8bca4f8f8744bbb0fbd6809a700b5768d5ceba/docs/validation/native-spacy-w6-20261117/baseline-amendment.json)
updates the protocol before qualification; the original panel and thresholds
remain unchanged. Candidate source
[`da18e29c20e2d42036cee911e85ff8bb22c328ae`](https://github.com/CrispStrobe/quds_office_suite/commit/da18e29c20e2d42036cee911e85ff8bb22c328ae)
and its [remotely saved seal at `84b4120c`](https://github.com/CrispStrobe/quds_office_suite/blob/84b4120ca29715d9b6f6555d1393f5fcb97e47cf/docs/validation/native-spacy-w6-20261117/candidate-seal.json)
precede the first result at `3a8bca4f`. The qualification is now consumed.

## First result and limits

| Measure | Observed result |
| --- | --- |
| Deterministic baseline exact error corrections | **5/30** |
| Native planner exact error corrections | **22/30** |
| Combined exact error corrections | **22/30**, **17** beyond baseline |
| Prepositional-phrase group | **11/12** exact |
| Inversion group | **6/12** exact |
| Overlap group | **5/6** exact |
| Valid inputs / protection and adverse controls | **42 / 10**, zero findings |
| Unsupported locales | **4**, explicit unsupported status |
| Corrected rechecks | **22**, no further finding |
| Missed error cases | **8**, retained in the original report |

Of the 22 corrected rechecks, 18 explicitly preserve a compatible lexical
reading. Four conservatively abstain: three on finite-analysis guards and
one on a clause/reference guard. A no-further-finding result does not mean
every corrected parse was affirmatively accepted. Existing deterministic
findings keep precedence. Repeated executions, rechecks and lifecycle controls
are not additional independent linguistic cases. The finite result does not
establish broad German grammar coverage or target-device performance.

## Local regression and editor controls

The latest-source receipts retain complete logs and source identities:

| Check | Result | Receipt |
| --- | --- | --- |
| spaCy package | **73 passed** | [Package controls](https://github.com/CrispStrobe/quds_office_suite/blob/3a8bca4f8f8744bbb0fbd6809a700b5768d5ceba/docs/validation/native-spacy-w6-20261117/latest-package-controls/completion.json) |
| German deterministic/shared-facts regressions | **869 passed** | [Baseline controls](https://github.com/CrispStrobe/quds_office_suite/blob/3a8bca4f8f8744bbb0fbd6809a700b5768d5ceba/docs/validation/native-spacy-w6-20261117/latest-baseline-controls/receipt.json) |
| Existing editor proofing | **75 passed** | Same baseline receipt |
| German editor integration | **36 passed** | Same baseline receipt |
| Word run editing | **1 passed** | Same baseline receipt |
| Actual-native manual editor integration | **10 passed** | [Native editor controls](https://github.com/CrispStrobe/quds_office_suite/blob/3a8bca4f8f8744bbb0fbd6809a700b5768d5ceba/docs/validation/native-spacy-w6-20261117/editor-controls/latest-grammar-main.json) |
| Incoming icon rendering | **9 passed** | [Icon controls](https://github.com/CrispStrobe/quds_office_suite/blob/3a8bca4f8f8744bbb0fbd6809a700b5768d5ceba/docs/validation/native-spacy-w6-20261117/main-refresh-controls/latest-icon-controls.json) |

Three [earlier browser import controls](https://github.com/CrispStrobe/quds_office_suite/blob/3a8bca4f8f8744bbb0fbd6809a700b5768d5ceba/docs/validation/native-spacy-w6-20261117/package-refresh-controls/completion.json)
also pass and report native analysis as unavailable. Their receipt retains
their earlier source boundary. The real editor controls use the actual worker
and an authored development sentence, including original UTF-16 mapping,
correction/recheck/undo/redo, ignores, vocabulary, protected content and stale
replies. Delayed actual replies are labeled lifecycle harness controls.

## Preserved runtime and reconciliation history

The paired continuation starts from Quds `e7c67f29` and CrispEmbed `584dd9e5`.
The owner's native PR #236 merged at
[`04e8445823d82591120203e953365582d7ba33be`](https://github.com/CrispStrobe/CrispEmbed/commit/04e8445823d82591120203e953365582d7ba33be),
including review `05fe9a0c`; this branch includes that owner work. Worksheet
PR #235 and the separate GECToR lane remain independent.

Quds preserves the late Office union repairs at
[`ec4aeefe`](https://github.com/CrispStrobe/quds_office_suite/commit/ec4aeefe1be6ce1868ee7eebf004c07b25d4108a)
and main `23f3b420` at
[`92d8ec1e`](https://github.com/CrispStrobe/quds_office_suite/commit/92d8ec1ecf1e7aea0bb68a06cf96aa67eda4a2d1).
The common-parent frame repair already matched. A focused incoming calendar
boundary repair preserves its initial UTC/Berlin/Los Angeles probe evidence
and [successful civil-date checks](https://github.com/CrispStrobe/quds_office_suite/blob/3a8bca4f8f8744bbb0fbd6809a700b5768d5ceba/docs/validation/native-spacy-w6-20261117/calendar-repair/README.md).
These Office changes precede the final source seal.

The [first development comparison](https://github.com/CrispStrobe/quds_office_suite/tree/8eb3c03aaaeb6c676f44195e89deaef1535d541e/docs/validation/native-spacy-w6-20261117)
retains its earlier baseline 2/10 versus combined 7/10, with zero findings on
18 valid and three guard inputs, and three explicit unsupported locales.
Those earlier results remain bound to their original baseline. Independent
source, provider, harness and editor reviews are retained with subsequent
baseline refreshes; they do not replace the first qualification result.

W6 reuses the exact accepted native library SHA-256
`ea89842ebf22c08508aff1cfbf18d78a107118af320523009616ff62b6e42977`
and bundle manifest SHA-256
`d599768375de3dc56abcca5cf1d49da0015857f9d3a10640aebb5317ef5d427e`.
No native source, precision, budget, model data or native acceptance fixture
changed. The first eight-case native acceptance at `2176c89` remains consumed
and was not rerun. No Actions/status endpoint was queried, awaited or rerun.

The [audited Quds completion](https://github.com/CrispStrobe/quds_office_suite/blob/20026ff49ac37391e2a1fd0947560a587f2f7332/docs/validation/native-spacy-w6-20261117/completion.json)
and [147-artifact evidence index](https://github.com/CrispStrobe/quds_office_suite/blob/20026ff49ac37391e2a1fd0947560a587f2f7332/docs/validation/native-spacy-w6-20261117/evidence-index.json)
bind the final handoff. Independent audit resolves every score, merge, original
coordinate and source/runtime identity; all consumed qualification inputs and
initial failures remain retained.

## Next: W7 packaging, devices and distribution

Continue from the [Quds W6 handoff](https://github.com/CrispStrobe/quds_office_suite/blob/20026ff49ac37391e2a1fd0947560a587f2f7332/docs/validation/native-spacy-w6-20261117/README.md).
Qualify target-specific packages, native dependencies and notices, actual
device loading and cleanup/resource behavior, and immutable model/native
distribution with consumer readback. Production activation remains separate.
The manual provider API stays default off; no automatic Studio registration,
runtime release, model alias or application pin follows from this finite panel.
