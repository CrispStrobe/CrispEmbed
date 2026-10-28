# Josev owned task-quality v1 label review

These20 new requests form10 EN/DE scenario pairs and60 field labels. Labels were authored before model outputs. They do not replace the old four smoke references. The panel is CC0-1.0 and is a bounded owned holdout, not a broad representative benchmark.

| Scenario (EN and DE) | Type of fact | NOUL gold | Choice gold | Completed checks / ordinal gold |
| --- | --- | --- | --- | --- |
| fact-key | factual | true | a | 1 |
| fact-badge | factual | false | z | 0 |
| negation-depot | negation | true | m | 2 |
| negation-letter | negation | false | a | 1 |
| correction-time | temporal-correction | true | z | 2 |
| correction-shelf | temporal-correction | false | m | 0 |
| order-finish | relative-order | true | a | 1 |
| order-visits | relative-order | false | z | 1 |
| count-lamps | explicit-count | true | m | 2 |
| count-room | explicit-count | false | a | 0 |

Each request includes literal state facts, one truth question, one three-way selection and an ordered0/1/2 count question. A state rationale is included with each label in the JSON. Option and question insertion orders vary before outputs; no row is dropped for difficulty or model disagreement.

Review every English/German state, question, option description and label in [the frozen panel](../tests/regression/josev2b/task-quality-v1.json) before authorizing the new original-only GPU capture. No execution is authorized by this draft.
