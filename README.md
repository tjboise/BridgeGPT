# BridgeGPT 🌉

**BridgeGPT** is an AI-powered bridge inspection assistant developed at Rutgers University. It combines a Vision-Language Model (VLM) with CNN-based segmentation to analyze bridge images and answer natural language questions about structural elements and corrosion conditions.

## How It Works

BridgeGPT uses an **agentic few-shot pipeline**:

1. A user uploads a bridge photo and asks a question in natural language
2. The VLM (Gemini 2.5 Flash) decomposes the question into a sequence of tool calls
3. Two CNN models execute the tool calls on the image:
   - **Mask2Former** (Swin-Tiny) — segments 6 structural element classes
   - **SegFormer-v3** (MiT-B4, 768px + TTA) — detects rust/corrosion
4. The VLM synthesizes the CNN outputs to produce a final answer with a segmentation overlay

## Popular Questions

Here are some examples to get you started:

| Type | Example |
|------|---------|
| Element segmentation | *"Show me the girders."* |
| Rust segmentation | *"Segment all the rust areas."* |
| Element + rust | *"Show me rust on the floor beams."* |
| Comparative | *"Which element has the most rust?"* |
| Multi-element | *"Which elements have rust? Segment them."* |
| Condition assessment | *"What is the condition state of the bearing?"* |

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

## Web App

The live demo is available at: **[bridgegpt.streamlit.app](https://bridgegpt-b6sypfddyaavyt7nevycav.streamlit.app/)**

Upload a bridge image and ask questions such as:
- *"Can you simply describe the figure?"*
- *"Which element has the most rust?"*
- *"Which elements have no rust? Segment them."*

The app displays:
- A **segmentation overlay** with per-element colors
- A **color legend** identifying each segmented region
- A **professional text response** with condition assessment
- A collapsible **reasoning trace** showing the VLM's step-by-step analysis

## Repository Structure

```
BridgeGPT/
├── web_app.py          # Streamlit web application
├── bridge_vlm.py       # Agentic VLM (Gemini + OpenAI backends)
├── bridge_tools.py     # CNN tool executor (ToolExecutor)
├── requirements.txt    # Python dependencies
├── img/                # Logo and UI assets
└── standard/           # AASHTO bridge inspection manual (PDF)
```

## CNN Model Weights

Model weights are hosted on Hugging Face Hub and downloaded automatically at startup:
- `tjzhang123/bridgegpt-weights` — Mask2Former + SegFormer checkpoints (private)

## Deployment

The app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). Required secrets:

```toml
GOOGLE_API_KEY = "..."   # Gemini API key
HF_TOKEN = "..."         # Hugging Face token (for private model weights)
```

## Citation

If you use BridgeGPT in your research, please cite:

```
@misc{bridgegpt2025,
  title   = {BridgeGPT: Agentic Vision-Language Model for Bridge Inspection},
  author  = {Zhang, Tianjie and others},
  year    = {2025},
  school  = {Rutgers University}
}
```

## Affiliation

Developed at the **Department of Civil and Environmental Engineering, Rutgers University**.
