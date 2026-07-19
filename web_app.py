import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import torch
import google.generativeai as genai
from huggingface_hub import hf_hub_download

# ==========================================
# 1. 页面配置 & 风格
# ==========================================
BRIDGEGPT_LOGO = "img/logo_1.png"
RUTGERS_LOGO = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Rutgers_Scarlet_Knights_logo.svg/1200px-Rutgers_Scarlet_Knights_logo.svg.png"

st.set_page_config(
    page_title="BridgeGPT",
    page_icon=RUTGERS_LOGO,
    layout="wide"
)

st.markdown("""
    <style>
        .stApp { height: 100vh; overflow: hidden; }
        .block-container { padding-top: 2rem; padding-bottom: 0rem; height: 100vh; }
        section.main { overflow: hidden; }
        [data-testid="stSidebarUserContent"] { overflow-y: auto; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. API & 模型配置
# ==========================================
GEMINI_MODEL      = "gemini-2.5-flash"
AASHTO_PDF_PATH   = "standard/AASHTO-bridge_element_guide_manual__05092010.pdf"

MASK2FORMER_PRETRAINED = "facebook/mask2former-swin-tiny-ade-semantic"
MASK2FORMER_NUM_LABELS = 7

SEGFORMER_MODEL_ID   = "nvidia/mit-b4"
SEGFORMER_NUM_LABELS = 2
SEGFORMER_INPUT_SIZE = 768

HF_CACHE = None   # use default HuggingFace cache

# Weights repo on HuggingFace Hub (private)
HF_WEIGHTS_REPO = "tjzhang123/bridgegpt-weights"

def _get_ckpt(filename):
    """Download weight file from HF Hub if not cached locally."""
    # Allow local override via env var (for local dev)
    local = os.environ.get(f"CKPT_{filename.upper().replace('.','_')}", "")
    if local and os.path.exists(local):
        return local
    hf_token = None
    try:
        hf_token = st.secrets.get("HF_TOKEN")
    except Exception:
        pass
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")
    return hf_hub_download(repo_id=HF_WEIGHTS_REPO, filename=filename,
                           token=hf_token, cache_dir=HF_CACHE)

ELEMENT_COLORS = {
    "bearing":      (128,   0,   0),
    "bracing":      (  0, 128,   0),
    "deck":         (128, 128,   0),
    "floor_beam":   (  0,   0, 128),
    "girder":       (128,   0, 128),
    "substructure": (  0, 128, 128),
    "rust":         (220,  50,  50),
}

try:
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        raise KeyError
except Exception:
    if os.path.exists("api_key.txt"):
        with open("api_key.txt") as f:
            genai.configure(api_key=f.read().strip())
    else:
        st.error("No Gemini API key found.")


# ==========================================
# 3. CNN 加载 (cached)
# ==========================================
@st.cache_resource
def load_cnn():
    import torch.nn.functional as F
    from transformers import (Mask2FormerForUniversalSegmentation,
                               AutoImageProcessor,
                               SegformerForSemanticSegmentation,
                               SegformerImageProcessor)

    # ── Mask2Former (elements) ──
    _id2label = {i: n for i, n in enumerate(
        ["Background","Bearing","Bracing","Deck","Floor_beam","Girder","Substructure"])}
    _label2id = {n: i for i, n in _id2label.items()}
    m2f_proc = AutoImageProcessor.from_pretrained(
        MASK2FORMER_PRETRAINED, cache_dir=HF_CACHE)
    m2f_net  = Mask2FormerForUniversalSegmentation.from_pretrained(
        MASK2FORMER_PRETRAINED,
        num_labels=MASK2FORMER_NUM_LABELS,
        id2label=_id2label, label2id=_label2id,
        ignore_mismatched_sizes=True, cache_dir=HF_CACHE)
    state = torch.load(_get_ckpt("mask2former_enhanced_seed44.pth"),
                       map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    m2f_net.load_state_dict(state, strict=False)
    m2f_net = m2f_net.eval()

    # ── SegFormer-v3 (rust, 768px, TTA) ──
    sf_proc = SegformerImageProcessor(
        do_resize=False, do_normalize=True,
        image_mean=[0.485,0.456,0.406],
        image_std=[0.229,0.224,0.225])
    sf_net  = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_MODEL_ID,
        num_labels=SEGFORMER_NUM_LABELS,
        id2label={0:"background",1:"rust"},
        label2id={"background":0,"rust":1},
        ignore_mismatched_sizes=True, cache_dir=HF_CACHE)
    sf_state = torch.load(_get_ckpt("segformer_v3_seed44.pth"),
                          map_location="cpu", weights_only=False)
    if isinstance(sf_state, dict) and "state_dict" in sf_state:
        sf_state = sf_state["state_dict"]
    sf_net.load_state_dict(sf_state, strict=False)
    sf_net = sf_net.eval()

    def letterbox(img, size=768):
        w, h = img.size
        scale = size / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img_r = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
        canvas.paste(img_r, (pad_x, pad_y))
        return canvas, scale, pad_x, pad_y

    def get_miou_png(pil_image):
        orig_w, orig_h = pil_image.size

        # ── Element segmentation ──
        inputs = m2f_proc(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            out = m2f_net(**inputs)
        pred = m2f_proc.post_process_semantic_segmentation(
            out, target_sizes=[(orig_h, orig_w)])[0].numpy()
        elem_pil = Image.fromarray(pred.astype(np.uint8))

        # ── Rust segmentation (letterbox + TTA) ──
        canvas, scale, pad_x, pad_y = letterbox(pil_image, SEGFORMER_INPUT_SIZE)
        flipped = canvas.transpose(Image.FLIP_LEFT_RIGHT)
        accum = None
        for img_in in [canvas, flipped]:
            enc = sf_proc(images=img_in, return_tensors="pt")
            with torch.no_grad():
                logits = sf_net(**enc).logits  # (1, 2, H/4, W/4)
            up = F.interpolate(logits, size=(SEGFORMER_INPUT_SIZE, SEGFORMER_INPUT_SIZE),
                               mode="bilinear", align_corners=False)
            prob = torch.softmax(up, dim=1)[:, 1]  # rust prob
            if accum is None:
                accum = prob
            else:
                accum = accum + prob.flip(-1)
        accum /= 2
        # Unpad
        canvas_np = accum[0].numpy()
        nw = int(orig_w * scale)
        nh = int(orig_h * scale)
        crop = canvas_np[pad_y:pad_y+nh, pad_x:pad_x+nw]
        crop_pil = Image.fromarray((crop * 255).astype(np.uint8))
        rust_prob = np.array(crop_pil.resize((orig_w, orig_h), Image.BILINEAR)) / 255.0
        rust_pred = (rust_prob > 0.5).astype(np.uint8)
        rust_pil = Image.fromarray(rust_pred)

        return elem_pil, rust_pil

    class CNNWrapper:
        def get_miou_png(self, image_pil):
            return get_miou_png(image_pil)

    return CNNWrapper()


# ==========================================
# 4. VLM 加载 (cached)
# ==========================================
@st.cache_resource
def load_vlm():
    from bridge_vlm import BridgeVLMAgentic
    return BridgeVLMAgentic(GEMINI_MODEL, variant="few_shot")


# ==========================================
# 5. AASHTO PDF 加载 (cached)
# ==========================================
@st.cache_resource
def load_aashto_pdf():
    if not os.path.exists(AASHTO_PDF_PATH):
        return None
    try:
        return genai.upload_file(path=AASHTO_PDF_PATH, display_name="AASHTO Manual")
    except Exception:
        return None


# ==========================================
# 6. 可视化
# ==========================================
def _apply_one_layer(img_np, mask_np, color_rgb, alpha=0.45):
    result = img_np.copy()
    mask_bool = (mask_np == 1)
    color = np.array(color_rgb, dtype=np.float32)
    result[mask_bool] = (img_np[mask_bool].astype(np.float32) * (1 - alpha)
                         + color * alpha).clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 255), 2)
    return result


def overlay_mask(image_pil, mask_np, color_rgb=(220, 50, 50), alpha=0.45):
    img = np.array(image_pil).copy()
    return Image.fromarray(_apply_one_layer(img, mask_np, color_rgb, alpha))


def _collect_deps(step_id, graph, visited=None):
    """Recursively collect all step ids in the dependency chain of step_id."""
    if visited is None:
        visited = set()
    if step_id not in graph or step_id in visited:
        return visited
    visited.add(step_id)
    args = graph[step_id].get("args", {})
    for k in ("source", "mask", "mask_a", "mask_b"):
        if k in args:
            _collect_deps(args[k], graph, visited)
    return visited


def multi_color_overlay(image_pil, plan, executor):
    """
    Render only the steps reachable from finish(output=N), each select
    colored by its class. Returns (result_np, legend_items).
    """
    steps     = plan.get("steps", [])
    output_id = plan.get("output")
    img       = np.array(image_pil).copy()
    legend    = []

    if output_id is None:
        return img, legend

    graph    = {s["id"]: s for s in steps}
    dep_ids  = _collect_deps(output_id, graph)

    # Gather select steps that are in the dependency chain.
    # Exclude "rust" — it's a counting tool, never the element we want to color.
    cls_to_sids = {}
    for s in steps:
        if s["id"] not in dep_ids:
            continue
        if s.get("tool") == "select":
            cls = s.get("args", {}).get("class_name", "").lower()
            if cls in ELEMENT_COLORS and cls != "rust":
                cls_to_sids.setdefault(cls, []).append(s["id"])

    # For each element select, prefer the intersect/subtract result that
    # directly uses it (element ∩ rust) — but NOT a union (which may be huge).
    def _best_mask_id(select_sid):
        for s in steps:
            if s["id"] not in dep_ids:
                continue
            if s.get("tool") in ("intersect", "subtract"):
                args = s.get("args", {})
                if select_sid in (args.get("mask_a"), args.get("mask_b")):
                    return s["id"]
        return select_sid

    painted_any = False
    seen_cls = set()
    for cls, sids in cls_to_sids.items():
        if cls in seen_cls:
            continue
        color = ELEMENT_COLORS[cls]
        combined = None
        for sid in sids:
            mid = _best_mask_id(sid)
            mask = executor.results.get(mid)
            if not isinstance(mask, np.ndarray):
                mask = executor.results.get(sid)
            if isinstance(mask, np.ndarray) and mask.sum() > 0:
                binary = (mask > 0).astype(np.uint8)
                combined = binary if combined is None else np.logical_or(combined, binary).astype(np.uint8)
        if combined is not None and combined.sum() > 0:
            img = _apply_one_layer(img, combined, color)
            legend.append((cls, color))
            seen_cls.add(cls)
            painted_any = True

    # Fallback: paint full final output mask with a single color
    if not painted_any:
        final_mask, _ = executor.finalize(output_id)
        if final_mask.sum() > 0:
            cls = next(iter(cls_to_sids), "rust")
            color = ELEMENT_COLORS.get(cls, (220, 50, 50))
            img = _apply_one_layer(img, final_mask, color)
            legend.append((cls, color))

    return img, legend


def format_reasoning(trace, executor_log):
    """
    Build a human-readable reasoning display from VLM trace + executor log.
    executor_log lines look like:
      [1] segment_elements()
      [3] select(source=1, class='girder') -> 12345 px
      [5] intersect -> 5678 px
      [6] count_pixels -> 5678
    Returns a markdown string.
    """
    lines = []

    # VLM text reasoning (when present)
    if trace.strip():
        lines.append("**VLM Reasoning:**")
        lines.append(trace.strip())
        lines.append("")

    # Raw executor log
    if executor_log:
        lines.append("**Execution steps:**")
        lines.append("\n".join(executor_log))

    return "\n\n".join(lines) if lines else "(No reasoning captured)"


def render_legend(legend_items):
    """Render an HTML color-swatch legend using st.markdown."""
    chips = []
    for label, (r, g, b) in legend_items:
        chips.append(
            f'<span style="display:inline-flex;align-items:center;margin-right:12px;">'
            f'<span style="display:inline-block;width:14px;height:14px;border-radius:3px;'
            f'background:rgb({r},{g},{b});margin-right:5px;border:1px solid #888;"></span>'
            f'<span style="font-size:0.85em;">{label.replace("_", " ").title()}</span>'
            f'</span>'
        )
    st.markdown(
        '<div style="margin-top:6px;">' + "".join(chips) + "</div>",
        unsafe_allow_html=True
    )


# ==========================================
# 7. 文字回复生成
# ==========================================
def generate_vlm_reasoning(question, trace, executor_log):
    """
    If the VLM produced no reasoning text, ask Gemini to synthesize one
    from the executor log steps. Returns a reasoning string.
    """
    if trace.strip():
        return trace.strip()   # already have it, no extra call needed
    if not executor_log:
        return ""
    steps_text = "\n".join(executor_log)
    prompt = (
        f"You are a bridge inspection AI assistant. A user asked: \"{question}\"\n\n"
        f"You executed the following CNN analysis steps:\n{steps_text}\n\n"
        "In 3-4 sentences, explain your reasoning: what you searched for, "
        "what the pixel counts told you, and how you reached your final conclusion. "
        "Be specific about element names and pixel counts. Do not use bullet points."
    )
    try:
        return genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt).text.strip()
    except Exception:
        return ""


def generate_text_response(query, trace, image_pil, mask_px, total_px, pdf_handle):
    coverage = f"{mask_px / total_px * 100:.1f}%" if total_px > 0 else "unknown"
    is_condition_q = any(k in query.lower() for k in
                         ["condition", "state", "cs", "rank", "severity", "level"])

    if is_condition_q and pdf_handle:
        instruction = (
            "You are a Senior Bridge Inspector at Rutgers University. "
            "Reference the attached AASHTO Manual (Corrosion defect criteria) "
            "and the provided VLM reasoning trace. "
            "Give a Condition State assessment (CS1–CS4) with evidence. "
            "Max 4 sentences."
        )
    else:
        instruction = (
            "You are BridgeGPT, an AI bridge inspection assistant at Rutgers University. "
            "Based on the VLM reasoning trace and the image, describe what was found "
            "in clear, professional language. "
            "If coverage data is available, mention it. Max 3 sentences."
        )

    prompt = f"""
{instruction}

[User Question]: {query}

[VLM Reasoning Trace]:
{trace[:1200]}

[Coverage]: The segmented region covers approximately {coverage} of the image area.

Respond without prefaces like "Sure" or "Based on". Be concise and evidence-based.
"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        payload = []
        if pdf_handle:
            payload.append(pdf_handle)
        payload.append(image_pil)
        payload.append(prompt)
        return model.generate_content(payload).text
    except Exception as e:
        return f"(Text generation error: {e})"


# ==========================================
# 8. UI
# ==========================================
st.image(BRIDGEGPT_LOGO, width=300)

cnn       = load_cnn()
vlm       = load_vlm()
aashto_pdf = load_aashto_pdf()

col1, col2 = st.columns([1, 1.8])

with col1:
    up_file = st.file_uploader("Upload Bridge Photo", type=["jpg", "png", "jpeg"])
    if up_file:
        img_pil = Image.open(up_file).convert("RGB")
        st.image(img_pil, use_container_width=True)

with col2:
    if "history" not in st.session_state:
        st.session_state["history"] = []

    chat_box = st.container(height=500)

    for msg in st.session_state["history"]:
        with chat_box.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # Collapsed reasoning trace (click to expand)
                has_trace = msg.get("trace", "").strip() or msg.get("steps_log", "").strip()
                if has_trace:
                    with st.expander("💡 Reasoning complete — click to expand", expanded=False):
                        st.markdown(format_reasoning(
                            msg.get("trace", ""),
                            msg.get("steps_log", "").splitlines()))
                # Answer + image
                if msg.get("img") is not None:
                    t_col, i_col = st.columns([1.5, 1])
                    with t_col:
                        st.markdown(msg["content"])
                        if msg.get("legend"):
                            render_legend(msg["legend"])
                    with i_col:
                        st.image(msg["img"], use_container_width=True)
                else:
                    st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])

    st.write("---")
    st.markdown("##### 💡 Popular Questions")
    q_cols = st.columns(3)
    questions = [
        "Can you simply describe the figure?",
        "Show me the girders.",
        "Segment all the rust areas.",
        "Which element has the most rust?",
        "Which elements have rust? Segment them.",
        "Which elements have no rust? Segment them.",
    ]
    selected_query = None
    for i, q in enumerate(questions):
        if q_cols[i % 3].button(q, use_container_width=True, key=f"q_{i}"):
            selected_query = q

    user_query = st.chat_input("Ask about the bridge...")
    final_query = selected_query if selected_query else user_query

    if up_file and final_query:
        last = st.session_state.get("last_query")
        if final_query != last:
            if (not st.session_state["history"] or
                    st.session_state["history"][-1]["content"] != final_query):
                st.session_state["history"].append({"role": "user", "content": final_query})
                st.rerun()

    if (st.session_state["history"] and
            st.session_state["history"][-1]["role"] == "user"):
        current_query = st.session_state["history"][-1]["content"]
        if st.session_state.get("last_query") != current_query:
            with chat_box.chat_message("assistant"):
                with st.status("🧠 BridgeGPT is thinking...", expanded=True) as status:
                    # ── Agentic VLM + CNN ──
                    plan, trace, executor = vlm.plan(current_query, img_pil, cnn)

                    # Synthesize reasoning if VLM didn't output text
                    reasoning = generate_vlm_reasoning(current_query, trace, executor.log)

                    # Show formatted reasoning (always visible while status is expanded)
                    st.markdown(format_reasoning(reasoning, executor.log))

                    result_img_np = None
                    mask_px       = 0
                    total_px      = img_pil.width * img_pil.height

                    legend_items = []
                    if plan.get("output") is not None:
                        result_img_np, legend_items = multi_color_overlay(image_pil=img_pil,
                                                                           plan=plan,
                                                                           executor=executor)
                        mask_np, _ = executor.finalize(plan["output"])
                        mask_px    = int(mask_np.sum())
                        if mask_px == 0:
                            result_img_np = None

                    # ── Text response ──
                    reply = generate_text_response(
                        current_query, trace, img_pil,
                        mask_px, total_px, aashto_pdf)

                    steps_log = "\n".join(executor.log)
                    status.update(label="💡 Reasoning complete — click to expand",
                                  state="complete", expanded=False)

                # Display answer + image outside the collapsed status
                if result_img_np is not None:
                    t_col, i_col = st.columns([1.5, 1])
                    with t_col:
                        st.markdown(reply)
                        if legend_items:
                            render_legend(legend_items)
                    with i_col:
                        st.image(result_img_np, use_container_width=True)
                else:
                    st.markdown(reply)

                st.session_state["history"].append({
                    "role": "assistant",
                    "content": reply,
                    "img": result_img_np,
                    "trace": reasoning,
                    "steps_log": steps_log,
                    "legend": legend_items,
                })
                st.session_state["last_query"] = current_query
                st.rerun()
