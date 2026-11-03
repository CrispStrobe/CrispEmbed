---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.5-0.8B
tags:
- ocr
- document-parsing
- multimodal
- markdown
- tables
- formulas
- key-information-extraction
- figure-captioning
- vllm
- qwen3_5
---

# OvisOCR3

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/658a8a837959448ef5500ce5/vRCIu5QD8VuIJolkC_ZHQ.png" alt="Ovis" width="30%" />
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/ATH-MaaS/OvisOCR3">Online Demo</a>
</p>

## Introduction

OvisOCR3 is a **unified 0.8B multimodal model for OCR and document understanding**. It integrates **document parsing, general OCR, key information extraction, document figure captioning, and cross-page table reconstruction** into a single compact model. These capabilities are selected through task-specific prompts, bringing recognition, extraction, and understanding together without requiring separate model deployments for each task.

Built on Qwen3.5-0.8B, OvisOCR3 is trained in four stages: continued pretraining, mid-training, on-policy distillation, and reinforcement learning. This training process builds a strong foundation in text recognition and document structure understanding, then refines these capabilities through domain-specific supervision and rewards.

OvisOCR3 achieves **state-of-the-art performance among models of comparable size across all five domains**, demonstrating that a single compact model can combine strong document parsing with a broad range of OCR and document-understanding capabilities.

![OvisOCR3 performance overview across five task domains.](./assets/ovisocr3_performance_overview.png)

## Capabilities

| Capability | Input | Output |
| --- | --- | --- |
| Document parsing | A page image and a parsing-mode prompt | Structured JSON, layout JSON, or Markdown with LaTeX formulas and HTML tables |
| General OCR | An image, optionally with a region, references, a question, a target language, or candidate labels | Text, bounding boxes, LaTeX, chart data, a label, or an answer |
| Key information extraction | An image, optionally with specified fields or a JSON output template | Key-value records or JSON following the supplied template |
| Document figure captioning | A page image and a document-figure-caption-mode prompt | Structured page content with localized visual captions and Mermaid representations for flowcharts |
| Cross-page table reconstruction | Consecutive page images and their parsed Markdown | Complete HTML tables with source-page associations, or a no-continuation prediction |

The [task prompts](./prompts/) specify the output contracts. The examples below use these files directly.

## Performance

Performance comparisons across five task domains are reported below. `--` denotes an unreported value, not zero. Bold scores indicate the best result in each column.

### Document Parsing

| Model | OmniDocBench1.6 | MDPBench | PureDocBench | Real5-OmniDocBench |
|---|---:|---:|---:|---:|
| Qwen3.5-OCR | 91.27 | 60.60 | 68.38 | 82.48 |
| Qwen3.8-Max | 93.70 | 86.40 | 80.56 | 86.86 |
| GLM-OCR | 95.22 | 67.30 | 63.34 | 90.32 |
| PaddleOCR-VL-1.6 | 96.34 | 78.90 | 65.44 | 93.19 |
| MonkeyOCRv2-B-Parsing | 91.57 | 83.30 | 64.91 | 89.22 |
| OvisOCR2 | 96.58 | 69.00 | 75.10 | 92.29 |
| HunyuanOCR-1.5 | 94.74 | 76.80 | 67.58 | 88.53 |
| Xiaomi-OCR-0 | **96.83** | 73.00 | 74.60 | 92.73 |
| **OvisOCR3** | 96.74 | **88.80** | **82.31** | **93.63** |

### General OCR

| Model | OCRBench | OCRBench v2 EN | OCRBench v2 ZH |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B | 79.1 | 41.5 | 43.9 |
| Qwen3.5-2B | 85.4 | 48.1 | 55.4 |
| Qwen3.5-4B | 85.0 | 54.1 | 61.0 |
| Qwen3.5-OCR | 76.8 | 44.5 | 43.8 |
| PaddleOCR-VL-1.6 | 55.9 | 23.8 | 30.4 |
| HunyuanOCR-1.5 | 86.1 | 50.7 | 57.7 |
| GLM-OCR | 83.8 | 32.9 | 48.1 |
| Xiaomi-OCR-0 | 84.6 | 40.1 | 48.4 |
| **OvisOCR3** | **88.8** | **65.9** | **61.3** |

### Key Information Extraction

| Model | OCRBench KIE | OCRBench v2 KIE | CC-OCR KIE | In-house KIE Bench |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-OCR | 78.50 | 72.90 | 88.43 | 59.90 |
| GLM-OCR | 79.00 | 74.22 | 68.36 | 75.25 |
| HunyuanOCR-1.5 | 81.00 | 68.63 | 66.08 | 75.96 |
| Xiaomi-OCR-0 | 76.50 | 74.66 | 79.53 | 75.68 |
| **OvisOCR3** | **90.50** | **83.01** | **88.69** | **77.73** |

### Document Figure Captioning

| Model | DocFigCapBench Localization F1 | DocFigCapBench Caption Score |
| --- | ---: | ---: |
| HunyuanOCR-1.5 | 48.24 | -- |
| Qwen3.5-OCR | 78.47 | 4.74 |
| OvisOCR2 | 82.18 | -- |
| MinerU2.5-Pro | 85.00 | 4.61 |
| PaddleOCR-VL-1.6 | 85.32 | 4.29 |
| Qwen3.8-Max | 86.67 | **7.30** |
| **OvisOCR3** | **90.97** | 6.93 |

### Cross-Page Table Reconstruction

| Model | CrossTabBench Relation F1 (P/R) | CrossTabBench Reconstruction TEDS | MPDocBench-Parse Truncated Table TEDS |
| :--- | ---: | ---: | ---: |
| PaddleOCR-VL-1.6 | 42.61 (90.13/27.90) | 24.24 | 83.39 |
| Logics-Parsing-V3 | 44.55 (90.63/29.53) | 26.10 | 88.05 |
| MinerU2.5-Pro | 59.14 (93.81/43.18) | 41.02 | 91.44 |
| MinerU2.5-Pro + MinerU-PoPo | 54.76 (97.40/38.09) | 36.42 | 91.87 |
| Qwen3.8-Max | 94.47 (90.19/99.19) | **91.71** | 89.57 |
| **OvisOCR3** | **97.07** (96.20/97.96) | 89.66 | **92.04** |

## Inference

The example uses a local model directory containing the weights, configuration, tokenizer, processor files, and `prompts/`. Run it from that directory, or set `MODEL_PATH` to its location. A GPU environment supported by vLLM is required.

```bash
pip install "vllm==0.22.1" pillow
```

```python
import warnings
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams


MODEL_PATH = Path(".")

model = LLM(
    model=str(MODEL_PATH.resolve()),
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    gdn_prefill_backend="triton",
    limit_mm_per_prompt={"image": 2},
)
tokenizer = model.get_tokenizer()
sampling_params = SamplingParams(max_tokens=16384, temperature=0.0)


def load_prompt(name: str) -> str:
    return (MODEL_PATH / "prompts" / f"{name}.txt").read_text(encoding="utf-8")


def infer(
    images: list[Image.Image], task_prompt: str, *,
    return_metadata=False,
):
    if not 1 <= len(images) <= 2:
        raise ValueError("This example accepts one or two images per request.")

    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": task_prompt})
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    request = {
        "prompt": prompt,
        "multi_modal_data": {"image": [image.convert("RGB") for image in images]},
        "mm_processor_kwargs": {
            "images_kwargs": {
                "min_pixels": 448 * 448,
                "max_pixels": 2880 * 2880,
            }
        },
    }
    output = model.generate([request], sampling_params)[0].outputs[0]
    if output.finish_reason == "length":
        warnings.warn("Generation reached a token limit; the output may be incomplete.")
    return (output.text, output.finish_reason) if return_metadata else output.text


page = Image.open("test1.jpg").convert("RGB")
text = infer([page], load_prompt("plain_ocr"))
Path("output.txt").write_text(text, encoding="utf-8")
print(text)
```

`infer` returns the raw generated text by default; `return_metadata=True` also returns the finish reason. It does not remove image tags, trim repetitions, repair JSON, or otherwise rewrite the response. All task examples share the inference settings above. These are usage settings, not benchmark reproduction settings.

An image list passed to `infer` belongs to **one request**. For independent pages, call `infer([page], prompt)` for each page. The two-image form is used for cross-page reconstruction below.

## Task-Specific Usage

The following snippets reuse `page`, `infer`, and `load_prompt` from the inference example. Replace placeholder values with the fields, regions, or questions needed for your document. Use literal placeholder replacement rather than Python `str.format`, since prompts also contain JSON and LaTeX braces.

### Document Parsing

Compared with OvisOCR2's Markdown parsing, OvisOCR3 adds structured document parsing and document layout detection. Structured parsing returns region-level records with bounding boxes, categories, and content; Markdown parsing returns a continuous document without explicit bounding boxes or category labels for individual text regions.

| Mode | Prompt | Output |
| --- | --- | --- |
| Markdown | [document_markdown.txt](./prompts/document_markdown.txt) | A document in natural reading order, with LaTeX formulas, HTML tables, and image-region references |
| Structured | [document_structured.txt](./prompts/document_structured.txt) | A JSON array with `bbox`, `category`, and `content` for each element |
| Layout only | [document_layout.txt](./prompts/document_layout.txt) | A JSON array with `bbox` and `category`, without transcription |

For `document_markdown`, follow the [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) postprocessing order: strip leading and trailing whitespace, optionally filter image-tag blocks (enabled by default), and then apply repeated-tail cleanup to every response regardless of the finish reason. Set `filter_imgtags=False` to retain visual-region references for rendering. Keep `raw_markdown` for inspection; structured and layout JSON outputs are left unchanged. The helper checks its own length and repetition thresholds. It does not recover missing content or make a truncated response complete.

```python
def _clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    n = len(text)
    if n < min_text_len:
        return text

    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue

        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1

        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len

        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len :]

    return text


# Format 1: Markdown, with optional image-tag filtering and repeated-tail cleanup.
raw_markdown = infer([page], load_prompt("document_markdown"))
filter_imgtags = True
text = raw_markdown.strip()
if filter_imgtags:
    text = "\n\n".join(
        block
        for block in text.split("\n\n")
        if not block.strip().startswith('<img src="images/bbox_')
    )
markdown = _clean_truncated_repeats(text)

# Format 2: Structured JSON with bbox, category, and content.
structured_json = infer([page], load_prompt("document_structured"))

# Format 3: Layout JSON with bbox and category only, without transcription.
layout_json = infer([page], load_prompt("document_layout"))
```

Structured and layout outputs follow natural reading order. The allowed category lists and category-specific content formats are defined in their respective prompts.

#### Coordinates and Visual Regions

Bounding boxes use `[left, top, right, bottom]` order with integer coordinates in `[0, 1000)`, relative to the image supplied to the model. The origin is the top-left corner. Convert to pixels with `x * image_width / 1000` and `y * image_height / 1000`. For cropped inputs, coordinates are relative to that crop, not the original page.

Markdown and structured parsing can reference visual regions using `<img src="images/bbox_left_top_right_bottom.jpg" />`. These paths describe crops to create from the input image; the model does not write image files. Preserve the image orientation used for inference when extracting crops.

<details>
<summary>Save Markdown together with its referenced image crops</summary>

```python
import re


BBOX_IMAGE_PATTERN = re.compile(r'images/bbox_(\d+)_(\d+)_(\d+)_(\d+)\.jpg')


def save_markdown_with_crops(markdown: str, page_image: Image.Image, output_dir: str):
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    width, height = page_image.size

    for left, top, right, bottom in set(BBOX_IMAGE_PATTERN.findall(markdown)):
        x1 = max(0, min(width, round(int(left) * width / 1000)))
        y1 = max(0, min(height, round(int(top) * height / 1000)))
        x2 = max(0, min(width, round(int(right) * width / 1000)))
        y2 = max(0, min(height, round(int(bottom) * height / 1000)))
        if x2 <= x1 or y2 <= y1:
            warnings.warn(f"Skipping invalid crop: {(left, top, right, bottom)}")
            continue
        crop_path = images_dir / f"bbox_{left}_{top}_{right}_{bottom}.jpg"
        page_image.crop((x1, y1, x2, y2)).convert("RGB").save(crop_path)

    (output_dir / "output.md").write_text(markdown, encoding="utf-8")


save_markdown_with_crops(markdown, page, "output")
```

This helper handles single-page Markdown references. For reconstructed tables, resolve each `page_N/` prefix against its corresponding source page and save crops under that page's directory.

</details>

### General OCR

<a id="general-ocr-prompts"></a>

| Task | Prompt | Output |
| --- | --- | --- |
| Plain OCR | [plain_ocr.txt](./prompts/plain_ocr.txt) | Text without markup or bounding boxes |
| Word-level text spotting | [word_spotting.txt](./prompts/word_spotting.txt) | JSON records with `bbox` and `text`, one per word or text token |
| Reference grounding | [reference_grounding.txt](./prompts/reference_grounding.txt) | JSON records with `bbox` and the requested `reference` |
| Region OCR | [region_ocr.txt](./prompts/region_ocr.txt) | Text within a specified region |
| Formula recognition | [formula_recognition.txt](./prompts/formula_recognition.txt) | LaTeX expressions enclosed in `$$ ... $$` |
| Chart parsing | [chart_parsing.txt](./prompts/chart_parsing.txt) | JSON with `title`, `source`, `x_title`, `y_title`, and `values` |
| Text translation | [text_translation.txt](./prompts/text_translation.txt) | Text from a specified region translated into the target language |
| Image classification | [image_classification.txt](./prompts/image_classification.txt) | Exactly one label from the supplied candidates |
| Visual question answering | [visual_question_answering.txt](./prompts/visual_question_answering.txt) | Reasoning followed by a final answer inside `\boxed{}` |
| Position-aware VQA | [position_aware_vqa.txt](./prompts/position_aware_vqa.txt) | JSON with `answer` and a string-valued `bbox` |

All ten tasks read the linked `.txt` prompt files. Image classification takes a target description and candidate labels, and position-aware VQA uses its own answer-and-box instruction. Fill only the requested task inputs in each template.

```python
import re


def general_ocr_prompt(name: str) -> str:
    prompt = load_prompt(name)
    return prompt if name == "position_aware_vqa" else prompt.strip()


def general_vqa_prompt(question: str) -> str:
    prompt = general_ocr_prompt("visual_question_answering").replace("{question}", question)
    return re.sub(r"\n{3,}", "\n\n", prompt).strip()

```

```python
import json

text = infer([page], general_ocr_prompt("plain_ocr"))
words_json = infer([page], general_ocr_prompt("word_spotting"))

references = json.dumps(["Invoice number", "Total"], ensure_ascii=False, separators=(",", ":"))
grounding_prompt = general_ocr_prompt("reference_grounding").replace("{references}", references)
grounding_json = infer([page], grounding_prompt)

region = json.dumps([100, 100, 900, 300], separators=(",", ":"))
region_prompt = general_ocr_prompt("region_ocr").replace("{target_region}", region)
region_text = infer([page], region_prompt)

formula_latex = infer([page], general_ocr_prompt("formula_recognition"))
chart_json = infer([page], general_ocr_prompt("chart_parsing"))

translation_prompt = general_ocr_prompt("text_translation").replace("{target_region}", region)
translation_prompt = translation_prompt.replace("{target_language}", "English")
translated_text = infer([page], translation_prompt)

target_description = "the main subject"
candidate_labels = json.dumps(["cat", "dog", "bird", "fish"], ensure_ascii=False)
classification_prompt = general_ocr_prompt("image_classification").replace(
    "{target description}", target_description
).replace("{candidate_labels}", candidate_labels)
selected_label = infer([page], classification_prompt)

question = "What is the invoice number?"
vqa_prompt = general_vqa_prompt(question)
answer_with_reasoning = infer([page], vqa_prompt)

position_prompt = general_ocr_prompt("position_aware_vqa").replace("{question}", question)
answer_json = infer([page], position_prompt)
```

References and regions use compact JSON. Supply references expected to be visible in the image. For classification, `{target description}` specifies what to classify, and `{candidate_labels}` is a JSON array of candidate labels. VQA accepts the complete question, including answer choices when applicable; its visible reasoning ends with a boxed answer. Position-aware VQA accepts the question text and uses its own answer-and-box prompt. Formula recognition and chart parsing need no template substitutions.

Spotting and grounding boxes, as well as input regions, use `[x_min,y_min,x_max,y_max]` with integer coordinates in `[0,1000)`, normalized independently by image width and height. The origin is the top-left corner, and coordinates refer to the supplied image, including when it is a crop. Use `[0,0,999,999]` for whole-image region OCR or translation.

Chart parsing represents scalar data values as strings and uses the literal string `"None"` for missing title, source, or axis metadata. Position-aware VQA uses `"bbox": "(x1,y1,x2,y2)"`, unlike the numeric arrays used by spotting and grounding.

### Key Information Extraction

| Mode | Prompt | Additional input |
| --- | --- | --- |
| Open extraction | [kie_open.txt](./prompts/kie_open.txt) | None; extract all visible key-value pairs |
| Specified-field extraction | [kie_fields.txt](./prompts/kie_fields.txt) | `{keys_json}`: a JSON array of field names |
| Schema-based extraction | [kie_schema.txt](./prompts/kie_schema.txt) | `{schema_json}`: a JSON template specifying the desired output structure |

Schema-based extraction uses an example of the desired JSON output, not a formal JSON Schema declaration. Empty strings serve as value placeholders, while nested objects and arrays define the output structure.

```python
import json

open_json = infer([page], load_prompt("kie_open"))

keys = json.dumps(["invoice_number", "date", "total"], ensure_ascii=False)
fields_prompt = load_prompt("kie_fields").replace("{keys_json}", keys)
fields_json = infer([page], fields_prompt)

schema = {
    "invoice_number": "",
    "date": "",
    "seller": {
        "name": "",
        "address": "",
    },
    "items": [
        {
            "name": "",
            "quantity": "",
            "unit_price": "",
            "amount": "",
        }
    ],
    "total": "",
}
schema_prompt = load_prompt("kie_schema").replace(
    "{schema_json}", json.dumps(schema, ensure_ascii=False)
)
schema_json = infer([page], schema_prompt)
```

Values should retain the visible source text. Open and specified-field extraction use arrays for repeated values and empty strings for empty values; specified-field extraction also uses empty strings for missing fields. Schema-based extraction follows the supplied template and uses empty strings for missing values. Adherence to the template is prompted, not enforced by constrained decoding in this example; validate generated JSON before downstream use.

### Document Figure Captioning

Use the **full page image** with [figure_captioning.txt](./prompts/figure_captioning.txt):

```python
caption_json = infer([page], load_prompt("figure_captioning"))
```

The model returns a structured page JSON array with `bbox`, `category`, and `content`. For visual categories such as `image` and `chart`, `content` contains a concise description inside `<caption>...</caption>`, with a maximum of 100 words requested by the prompt. These captions replace the image references used for visual regions in structured document parsing. Flowcharts additionally include a Mermaid representation inside `<mermaid>...</mermaid>`. Other page elements retain their task-specific text, formula, or table content.

This is localized figure description within a document page, not just transcription of a printed figure caption.

### Cross-Page Table Reconstruction

OvisOCR3 reconstructs cross-page tables in two stages using the same checkpoint:

1. **Stage 1: page parsing.** Parse each page independently with [document_markdown.txt](./prompts/document_markdown.txt) or [document_structured.txt](./prompts/document_structured.txt). For structured JSON output, concatenate each element’s `content` in reading order to obtain Markdown, preserving HTML tables, formulas, and image references.
2. **Stage 2: table reconstruction.** Supply two adjacent page images and their parsed Markdown with [crosspage_table.txt](./prompts/crosspage_table.txt). The model identifies whether a table continues across the page boundary and reconstructs continued fragments as a single HTML table; otherwise, it returns a negative prediction.

```python
import re

pages = [
    Image.open("page1.jpg").convert("RGB"),
    Image.open("page2.jpg").convert("RGB"),
]
page_results = [
    infer([image], load_prompt("document_markdown"), return_metadata=True)
    for image in pages
]
page_markdowns = []
for index, (raw, reason) in enumerate(page_results, start=1):
    if reason not in ("stop", "length"):
        raise ValueError(f"Page {index}: finish_reason={reason}")
    markdown = _clean_truncated_repeats(raw.strip())
    markdown = re.sub(r'^```(?:markdown|html)?\s*\n|\n```\s*$', '', markdown.strip(), flags=re.I)
    if not markdown:
        raise ValueError(f"Page {index}: empty Markdown")
    page_markdowns.append(re.sub(
        r'(?<![\w/])(?:page_\d+/)?(images/bbox_\d+_\d+_\d+_\d+\.jpg)',
        rf'page_{index}/\1', markdown,
    ))

if all(re.search(r'<table\b[^>]*>.*?</table\s*>', md, re.I | re.S) for md in page_markdowns):
    page_context = "\n\n".join(
        f'<INPUT_PAGE index="{index}">\n{markdown}\n</INPUT_PAGE>'
        for index, markdown in enumerate(page_markdowns, start=1)
    )
    prompt = load_prompt("crosspage_table").removesuffix("\n") + "\n\nInput Markdown:\n\n" + page_context
    tables_json, reason = infer(pages, prompt, return_metadata=True)
    if reason != "stop":
        raise ValueError(f"Cross-page output is incomplete: finish_reason={reason}")
else:
    tables_json = '[{"source_pages": [], "html": "No cross-page table exists between these pages."}]'
print(tables_json)
```

The result is a JSON array. Each record has `source_pages`, using **input-local, 1-based page numbers**, and `html`, containing one complete HTML table. Images within tables retain page-qualified references such as `page_1/images/bbox_100_200_300_400.jpg`.

When no cross-page table exists, the specified response is:

```json
[{"source_pages": [], "html": "No cross-page table exists between these pages."}]
```

For tables spanning more than two pages, **Stage 2 is applied recursively**, processing adjacent pairs from the end of each consecutive sequence of table-bearing pages toward its beginning. After each merge, the reconstructed table is written into the earlier page and its updated Markdown is re-rendered before the next step, keeping the image and text consistent. For example, the model first merges pages 2 and 3, then pairs the updated page 2 with page 1. Each step uses only two page images while preserving the original source-page associations. A ready-to-run document-level implementation is available in [document_crosspage.py](./document_crosspage.py), supporting either `document_markdown.txt` or `document_structured.txt` for Stage 1 and converting structured JSON to Markdown before reconstruction.

![Recursive reconstruction of a table spanning three pages.](./assets/crosspage_recursive.png)

*Stage 2 first reconstructs the table across pages 2 and 3, then pairs the re-rendered page 2 with page 1. Each step uses two page images and their current Markdown.*

The script uses the same vLLM version and inference settings as the example above. Install the additional dependencies and run the command below from the model repository root, supplying page images in reading order.

```bash
python -m pip install beautifulsoup4 markdown-it-py playwright
python -m playwright install --with-deps chromium

python document_crosspage.py \
    --model . \
    --stage1-prompt document_structured.txt \
    --output crosspage_output \
    page1.jpg page2.jpg page3.jpg
```

Use a new output directory for each run; it contains `document.md`, per-page Markdown, raw responses, and `result.json` with table provenance and any processing errors.

## Output Handling

### Validation and Rendering

- Preserve raw outputs before parsing, rendering, or applying cleanup. JSON, HTML, LaTeX, and Mermaid may need validation, and generation can stop before an output is complete.
- Render Markdown with support for HTML tables and LaTeX. Figure captions and Mermaid content require their own handling inside structured JSON.
- Treat generated HTML and Mermaid as untrusted content. Sanitize or isolate them before rendering in a browser.
- For evaluation, freeze prompts, preprocessing, generation parameters, and postprocessing. These usage examples do not define a benchmark protocol.

## Limitations

OvisOCR3 can produce incorrect, incomplete, repeated, or malformed outputs. Small or blurred text, severe perspective distortion, dense layouts, complex formulas, and long tables may affect recognition and structure recovery. Captions and answers can include unsupported details; verify them against the source image.

Cross-page reconstruction also depends on the quality of the first-stage page parsing and correct page order. Missing table fragments, inconsistent page context, or truncated generations can lead to incorrect merges. Review results before using them in critical applications.

## Technical Report and Citation

If you find OvisOCR3 useful, please consider citing our technical report:

```bibtex
@misc{ovisocr3,
  title  = {{OvisOCR3 Technical Report}},
  author = {{ATH-MaaS}},
  year   = {2026}
}
```

The technical report link will be added when available.

## License

This project is licensed under the [Apache License, Version 2.0](./LICENSE) (SPDX-License-Identifier: Apache-2.0).
