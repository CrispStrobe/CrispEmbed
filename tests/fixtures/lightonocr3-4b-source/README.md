---
license: apache-2.0
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- ocr
- document-understanding
- visual-grounding
- vision-language
- tables
- forms
---

<div align="center">

<img src="https://huggingface.co/lightonai/LightOnOCR-3-4B/resolve/main/lightonocr3_logo.png" alt="LightOnOCR-3 logo" width="600">

[![Website](https://img.shields.io/badge/LightOn-Website-blue?logo=google-chrome)](https://lighton.ai)
[![LinkedIn](https://img.shields.io/badge/LightOn-LinkedIn-0A66C2?logo=linkedin)](https://www.linkedin.com/company/lighton/)
[![X](https://img.shields.io/badge/@LightOnIO-X-black?logo=x)](https://x.com/LightOnIO)

📄 [Paper](https://arxiv.org/pdf/2601.14251) | 📝 [Blog](https://huggingface.co/blog/lightonai/lightonocr-3) | 🚀 [Demo](https://huggingface.co/spaces/lightonai/LightOnOCR-3-Demo) | 💻 [GitHub](https://github.com/lightonai/LightOnOCR) | 🤗 [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B)

</div>

# LightOnOCR-3-4B

**Best OCR model.** LightOnOCR-3-4B is the largest and most accurate model of the LightOnOCR-3 family, **[LightOn's](https://lighton.ai)** new generation of lightweight end-to-end OCR models with layout and visual understanding. We recommend it for most OCR tasks.

## About LightOnOCR-3

LightOnOCR-3 is a new family of highly performant lightweight OCR models. Compared to the previous generation, they bring significant improvements in speed and transcription quality and introduce new visual understanding features: the models now output bounding box coordinates with labels for all visual elements of a document, short descriptions of images, and the numerical data of figures and charts. With these capabilities, they offer a ready-to-use, easier-to-maintain alternative to complex document understanding pipelines.

The models come in three sizes. The 1B keeps the LightOnOCR-2-1B architecture, while the 0.8B and 4B adopt the Qwen3.5 vision-language architecture, which simplifies integration with existing tools and brings a significant speed-up. All models are released under the Apache 2.0 license for research and commercial use.

## Highlights

* 📝 **Transcription mode:** call the model with an empty prompt and get the full page text, as before. Switching from LightOnOCR-2 requires no change
* 📍 **Grounding mode:** call it with the `grounding` prompt and every block comes back with a label and a bounding box
* 🖼️ **Visual understanding:** images get a short description, charts become a table of their data points
* 🧠 **End-to-End:** one model instead of a document understanding pipeline, easier to deploy and maintain
* 🧾 **Versatile:** handles tables, receipts, forms, charts, multi-column layouts, and math notation

---

## Model Variants

| Variant | Description |
|---------|-------------|
| **[LightOnOCR-3-4B](https://huggingface.co/lightonai/LightOnOCR-3-4B)** | Best OCR model, recommended for most tasks |
| **[LightOnOCR-3-1B](https://huggingface.co/lightonai/LightOnOCR-3-1B)** | LightOnOCR-2 architecture, drop-in upgrade for existing deployments |
| **[LightOnOCR-3-0.8B](https://huggingface.co/lightonai/LightOnOCR-3-0.8B)** | Fast and Efficient model |

---

## Prompt Modes

The models can be used as before in a **transcription-only mode**: called with an empty prompt (`""`; the image only), they output all textual elements of the page as markdown. If you are currently using our previous models, switching to LightOnOCR-3 requires no change, as this is the default behavior.

The new usage mode is to call the models with the `grounding` prompt. They then output the new vision features alongside the transcribed text. Each block of content is prepended with a placeholder giving its type and its bounding box, in page coordinates normalized to 0–1000:

```
![title](80,70,920,140)
# Company report 2025

![text](80,175,920,225)
Revenue rose 20%.

![image](80,285,460,545)
A solar-powered factory.

![chart](540,285,920,545)
<table>
  <tr><th>Year</th><th>Revenue</th></tr>
  <tr><td>2024</td><td>€10M</td></tr>
  <tr><td>2025</td><td>€12M</td></tr>
</table>
```

* **Text blocks** contain the transcribed content of paragraphs, titles and other text elements.
* **Image blocks** pair a bounding box with a short description, making visual content accessible to retrieval and question-answering pipelines.
* **Chart blocks** contain an HTML table of the data points extracted from the figure, turning visual information into structured data.

The placeholder label is one of:

| Label | Definition |
|---|---|
| `text` | Ordinary body text and paragraphs. |
| `title` | Document, section, or paragraph headings. |
| `list` | Bulleted or numbered-list content. |
| `header` | Text in the page header. |
| `footer` | Text in the page footer. |
| `page_number` | Page-number regions. |
| `footnote` | Footnote text. |
| `caption` | Captions associated with figures, tables, or other visual elements. |
| `formula` | Mathematical formulas and equations. |
| `code` | Code blocks or code-like text. |
| `table` | Table regions and their content. |
| `image` | Illustrations, photographs, and other image regions. |
| `chart` | Charts, graphs, and plotted visual data. |
| `header_image` | Image regions in page headers, such as logos. |
| `footer_image` | Image regions in page footers. |
| `aside_text` | Marginal or side text outside the main text flow. |
| `+` suffix | Marks a continuation block, not a separate semantic class. |

Grounding adds about 25% output tokens over plain transcription (1,433 vs 1,158 tokens per page on average for the 4B). Other instructions are out of distribution: use the empty prompt (`""`) or `grounding`.

---

## Usage with Transformers

> **Note:** LightOnOCR-3-4B loads with the Qwen3.5 classes (Transformers ≥ 5.2, saved with 5.5.4). It is trained and evaluated with thinking disabled, which is the chat template default.

```bash
uv pip install "transformers>=5.5.4" pillow pypdfium2
```

```python
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

model_id = "lightonai/LightOnOCR-3-4B"
model = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/SROIE-receipt.jpeg"
prompt = "grounding"  # or "" for transcription-only

content = [{"type": "image", "url": url}]
if prompt:
    content.append({"type": "text", "text": prompt})
conversation = [{"role": "user", "content": content}]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.2)
generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
print(processor.decode(generated_ids, skip_special_tokens=True))
```

---

## Usage with vLLM

```bash
vllm serve lightonai/LightOnOCR-3-4B \
  --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

```python
import base64
import io

import pypdfium2 as pdfium
import requests

ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL = "lightonai/LightOnOCR-3-4B"
PROMPT = "grounding"  # or "" for transcription-only

# Render the first page of a PDF at 400 DPI, then resize to a longest edge of 2048 px (aspect ratio preserved)
pdf = pdfium.PdfDocument(requests.get("https://arxiv.org/pdf/2412.13663").content)
pil_image = pdf[0].render(scale=400 / 72).to_pil()
pil_image.thumbnail((2048, 2048))

buffer = io.BytesIO()
pil_image.save(buffer, format="PNG")
image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]
if PROMPT:
    content.append({"type": "text", "text": PROMPT})

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": content}],
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.9,
}

response = requests.post(ENDPOINT, json=payload)
print(response.json()["choices"][0]["message"]["content"])
```

The [LightOnOCR repository](https://github.com/lightonai/LightOnOCR) on GitHub provides a minimal client, CLI and viewer for the models served with vLLM, plus the code to reproduce our benchmarks.

---

## Rendering and Preprocessing Tips

* Render PDFs at 400 DPI to images using a target longest dimension of **2048px**
* Maintain aspect ratio to preserve text geometry

---

## License

Apache License 2.0

---

## Citation

```bibtex
@misc{lightonocr2_2026,
  title        = {LightOnOCR: A 1B End-to-End Multilingual Vision-Language Model for State-of-the-Art OCR},
  author       = {Said Taghadouini and Adrien Cavaill\`{e}s and Baptiste Aubertin},
  year         = {2026},
  howpublished = {\url{https://arxiv.org/abs/2601.14251}}
}
```

[![Downloads](https://img.shields.io/badge/dynamic/json?url=https://huggingface.co/api/models/lightonai/LightOnOCR-3-4B&query=downloads&label=Downloads&color=blue)](https://huggingface.co/lightonai/LightOnOCR-3-4B)
[![EU](https://img.shields.io/badge/🇪🇺%20Made%20in-Europe-blue)](https://huggingface.co/lightonai)