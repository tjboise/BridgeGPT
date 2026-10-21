# BridgeGPT 🌉

**BridgeGPT** is an AI-powered bridge inspection assistant developed at Rutgers University. It combines a Vision-Language Model (VLM) with CNN-based perception to answer natural language questions about bridge images — segmenting structural elements, detecting corrosion, and performing multi-step comparative reasoning.

## Live Demo

**[bridgegpt.streamlit.app](https://bridgegpt-b6sypfddyaavyt7nevycav.streamlit.app/)**

Upload a bridge photo and ask questions in plain English. The app returns a segmentation overlay, a color legend, a professional condition assessment, and a collapsible reasoning trace.

---

## How It Works

BridgeGPT uses an **agentic few-shot pipeline** with real multi-turn tool calling:

1. A user submits a bridge photo and a natural-language question
2. **Gemini 2.5 Flash** decomposes the question into a sequence of tool calls, one at a time
3. Two CNN models execute each tool call in real time:
   - **DAM2F** (Domain-Adapted Mask2Former, Swin-Tiny) — segments 6 structural element classes using median-frequency class weighting
   - **CorrFormer** (SegFormer MiT-B4, 768 px + TTA) — detects rust/corrosion with photometric augmentation
4. The model reads the **real tool results** (e.g. actual pixel counts) before deciding its next call — no guessing
5. After calling `finish()`, the VLM synthesizes the results into a final segmentation mask and text response

This agentic loop is what enables reliable Class C reasoning (e.g. "which element has the most rust?") — the model compares real pixel counts, not visual impressions.

---

## Question Types (SBIRS Benchmark)

| Class | Description | Example |
|-------|-------------|---------|
| **A** | Single or multi-element segmentation | *"Segment the girders."* |
| **B** | Rust on a specific element | *"Show me the rust on the floor beam."* |
| **C** | Multi-step comparative reasoning | *"Which element has the most rust? Segment it."* |
| **C** | All corroded elements | *"Which members have corrosion? Mask them."* |
| **C** | Non-corroded elements | *"Which structural members have no rust? Show them."* |

---

## Perception Models

### DAM2F — Element Segmentation

| Setting | Value |
|---------|-------|
| Architecture | Mask2Former (Swin-Tiny), pretrained on ADE20K |
| Classes | Background, Bearing, Bracing, Deck, Floor Beam, Girder, Substructure |
| Input size | 512 × 512 |
| Batch size | 2 |
| Epochs | 100 |
| Optimizer | AdamW (backbone lr=1×10⁻⁵, head lr=1×10⁻⁴, wd=0.05) |
| Scheduler | Cosine annealing (min lr=1×10⁻⁶) |
| Augmentation | Horizontal flip (p=0.5), random rotation ±15° |
| Loss | Weighted cross-entropy (median-frequency weighting) |

### CorrFormer — Corrosion Segmentation

| Setting | Value |
|---------|-------|
| Architecture | SegFormer MiT-B4, pretrained on ImageNet |
| Classes | Background, Rust |
| Input size | 768 × 768 (letterbox) |
| Batch size | 4 |
| Epochs | 100 |
| Optimizer | AdamW (lr=6×10⁻⁵, wd=1×10⁻⁴) |
| Scheduler | Cosine annealing (min lr=1×10⁻⁶) |
| Augmentation | HSV jitter (hue ±0.05, sat ×[0.3,1.7], val ×[0.6,1.4]) |
| TTA | 3 scales × 2 flips (averaged) |

---

## Element Classes & Colors

| Element | Color |
|---------|-------|
| Bearing | `(128, 0, 0)` Maroon |
| Bracing | `(0, 128, 0)` Dark Green |
| Deck | `(128, 128, 0)` Olive |
| Floor Beam | `(0, 0, 128)` Navy |
| Girder | `(128, 0, 128)` Purple |
| Substructure | `(0, 128, 128)` Teal |
| Rust | `(220, 50, 50)` Red |

---

## Prompts

### System Role Header

<details>
<summary>Click to expand</summary>

```
You are a bridge-inspection reasoning agent. You answer questions
about a bridge image. For questions that require segmentation or analysis
(e.g. "show the girders", "which element has the most rust?"), compose
atomic vision tools into a plan. For general questions that do not require
segmentation (e.g. "describe the image", "what do you see?"), answer
directly using your visual understanding without calling any tools — set
"steps": [] and "output": null, and put your answer in "reasoning".

The bridge elements are: [bearing, bracing, deck, floor_beam, girder, substructure].
Defects are limited to: rust.

Rules:
- Use ONLY the tools listed. Compose them; there are no shortcut tools.
- Every mask you use must come from a previous step (reference it by id).
- Name the final answer mask's step id in "output".
- Respond with JSON ONLY, no prose outside the JSON.
- Base your decisions on the values returned by the tools, not on visual
  impression of the image.
```

</details>

### Agentic Multi-Turn Guidance

<details>
<summary>Click to expand</summary>

```
You operate in a MULTI-TURN tool-calling session: call ONE tool at a time and
WAIT for its real result before deciding the next call. Never assume or guess
a count_pixels value -- always call it and read the actual number back before
comparing or deciding anything.

Before EVERY tool call, first write one short plain-text sentence stating what
you are about to do and why, THEN call the tool in that same turn. Do this
every single time, not just for the first call. This is your visible reasoning
trail -- without it, no one can audit why you made each decision, so never
skip it.

For comparison questions ("which element has the most rust", "which elements
have rust", "which elements have no rust"): first call select() on EVERY one
of the six elements and count_pixels each, to find out which elements actually
exist in this image (a count of 0 means that element is absent -- skip it for
the rest of the question, there is nothing there to check for rust). Only then
check rust on the elements that exist, by intersecting each one with the rust
mask and calling count_pixels again. Compare the REAL numbers returned before
deciding your final answer -- do not decide before you have checked all of
them.

When you are done, first write a short plain-text sentence summarizing the
REAL numbers you compared and why your chosen step is the answer, THEN call
finish(output=<step id>) naming the step that holds the final answer mask in
that same turn. Only call finish once you have actually checked the real
numbers via the tools -- never call it on a guess.
```

</details>

### Few-Shot Examples (Class A, B, C)

<details>
<summary>Click to expand</summary>

```
Worked examples (for illustration only -- the numbers below are MADE UP for an
imaginary image. For your REAL question below, you must call the real tools
and use whatever REAL numbers they return -- never reuse any number shown here.)

Question: "Segment the girder."
  call segment_elements() -> {"id": 1}
  call select(source=1, class_name="girder") -> {"id": 2}
  call finish(output=2)

Question: "Show me the rust on the bearing."
  call segment_elements() -> {"id": 1}
  call segment_rust() -> {"id": 2}
  call select(source=1, class_name="bearing") -> {"id": 3}
  call select(source=2, class_name="rust") -> {"id": 4}
  call intersect(mask_a=3, mask_b=4) -> {"id": 5}
  call finish(output=5)

Question: "Which element has the most rust? Segment it."
  [enumerates all 6 elements, intersects each with rust mask, calls
   count_pixels for each, compares real numbers, then calls finish()]

Question: "Which members have corrosion? Mask them."
  [enumerates all 6 elements, checks rust pixels for each, unions the
   whole-element masks of those with count > 0, then calls finish()]

Question: "Which structural members have no rust? Show them."
  [enumerates all 6 elements, checks rust pixels for each, unions the
   whole-element masks of those with count = 0, then calls finish()]
```

</details>

---

## Repository Structure

```
BridgeGPT/
├── web_app.py          # Streamlit web application
├── bridge_vlm.py       # Agentic VLM planner (Gemini, multi-turn tool calling)
├── bridge_tools.py     # CNN tool executor (ToolExecutor)
├── eval_sbirs.py       # Evaluation script for the SBIRS benchmark
├── requirements.txt    # Python dependencies
├── img/                # Logo and UI assets
└── standard/           # AASHTO bridge inspection manual (PDF)
```

---

## CNN Model Weights

Weights are hosted on Hugging Face Hub and downloaded automatically at startup:

- [`tjzhang123/bridgegpt-weights`](https://huggingface.co/tjzhang123/bridgegpt-weights) — DAM2F (Mask2Former) + CorrFormer (SegFormer) checkpoints

---

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). Required secrets:

```toml
GOOGLE_API_KEY = "..."   # Gemini API key
HF_TOKEN = "..."         # Hugging Face token (for private model weights)
```

---

## Affiliation

Developed at the **Department of Civil and Environmental Engineering, Rutgers University**.
