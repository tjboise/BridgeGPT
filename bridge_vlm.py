"""
bridge_vlm.py
==============================================================================
The VLM planner for BridgeGPT, with multiple PROMPT VARIANTS for comparison.

Single-turn: given a question (and the image), the VLM reads the tool
documentation, reasons, and emits ONE JSON plan of atomic tool calls. The
executor (bridge_tools.ToolExecutor) then runs that plan.

Prompt variants (select with BridgeVLM(variant=...)):
  zero_shot       detailed tool doc + rules, NO examples, free-form reasoning
  few_shot        + worked examples covering A / B / C question types
  zero_shot_cot   NO examples + STRUCTURED chain-of-thought scaffold
  few_shot_cot    examples + structured CoT  (expected strongest)
  minimal_doc     stripped-down tool doc (low end of the doc-detail ablation)
  few_shot_no_c   examples for A/B ONLY (tests the value of hard-class examples)

The 2x2 main axis is {zero_shot, few_shot} x {plain, _cot}; minimal_doc and
few_shot_no_c are the documentation / example-coverage ablations.
"""
import re
import json
import google.generativeai as genai

from bridge_tools import (TOOL_DOC, TOOL_DOC_MINIMAL, ELEMENT_MAP,
                          OUTPUT_FORMAT_COT, OUTPUT_FORMAT_NO_COT,
                          ToolExecutor)


# ------------------------------------------------------------
# Shared pieces
# ------------------------------------------------------------
HEADER = """You are a bridge-inspection reasoning agent. You answer a question
about a bridge image by composing atomic vision tools into a plan.

The bridge elements are: {elements}. Defects are limited to: rust.
"""

RULES = """
Rules:
- Use ONLY the tools listed. Compose them; there are no shortcut tools.
- Every mask you use must come from a previous step (reference it by id).
- Name the final answer mask's step id in "output".
- Respond with JSON ONLY, no prose outside the JSON.
"""

# Toggleable rule (ablated by the *_novisual variants). When enabled it is
# appended to RULES; when disabled the model is free to use visual impression,
# which exposes the failure mode where the VLM overrides its own tool
# computation with what the image "looks like".
NO_VISUAL_RULE = """- Base your decisions on the values returned by the tools, not on visual
  impression of the image.
"""

# Structured CoT scaffold (forces the model to separate "what to compare" from
# "what to output" -- the failure mode seen on class-C questions).
COT_SCAFFOLD = """
Think step by step before answering. In the "reasoning" field, work through the
problem in natural language -- what the question asks for, and how to obtain it
with the tools -- and only then produce the steps.
"""

# Worked examples come in TWO forms so the CoT ablation stays clean:
#   *_COT  examples include a "reasoning" field (shown to CoT variants)
#   *_PLAIN examples are steps-only, no reasoning (shown to no-CoT variants)
EX_AB_COT = """
Example:
Question: "Highlight the girder."
{
  "reasoning": "I need the girder region. Segment elements, then select girder.",
  "steps": [
    {"id": 1, "tool": "segment_elements", "args": {}},
    {"id": 2, "tool": "select", "args": {"source": 1, "class_name": "girder"}}
  ],
  "output": 2
}

Example (rust on an element):
Question: "Show the rust on the bearing."
{
  "reasoning": "Rust on bearing = intersection of the bearing region and rust.",
  "steps": [
    {"id": 1, "tool": "segment_elements", "args": {}},
    {"id": 2, "tool": "segment_rust", "args": {}},
    {"id": 3, "tool": "select", "args": {"source": 1, "class_name": "bearing"}},
    {"id": 4, "tool": "select", "args": {"source": 2, "class_name": "rust"}},
    {"id": 5, "tool": "intersect", "args": {"mask_a": 3, "mask_b": 4}}
  ],
  "output": 5
}
"""

EX_AB_PLAIN = """
Example:
Question: "Highlight the girder."
{
  "steps": [
    {"id": 1, "tool": "segment_elements", "args": {}},
    {"id": 2, "tool": "select", "args": {"source": 1, "class_name": "girder"}}
  ],
  "output": 2
}

Example (rust on an element):
Question: "Show the rust on the bearing."
{
  "steps": [
    {"id": 1, "tool": "segment_elements", "args": {}},
    {"id": 2, "tool": "segment_rust", "args": {}},
    {"id": 3, "tool": "select", "args": {"source": 1, "class_name": "bearing"}},
    {"id": 4, "tool": "select", "args": {"source": 2, "class_name": "rust"}},
    {"id": 5, "tool": "intersect", "args": {"mask_a": 3, "mask_b": 4}}
  ],
  "output": 5
}
"""

EX_C_COT = """
Example:
Question: "Which element has the most rust? Segment it."
{
  "reasoning": "Enumerate EVERY element (not just one or two) and intersect each with rust, reading back the actual pixel count from count_pixels before deciding -- never guess the winner without checking all of them. Suppose the counts come back as: bearing=18275, bracing=0, deck=0, floor_beam=0, girder=48151, substructure=9. Comparing all six numbers, girder's 48151 is the largest. The final answer is the WHOLE girder region (the step that selects girder from the element map), not the rust-on-girder intersection mask used only for the comparison.",
  "steps": [
    {"id": 1, "tool": "segment_elements", "args": {}},
    {"id": 2, "tool": "segment_rust", "args": {}},
    {"id": 3, "tool": "select", "args": {"source": 2, "class_name": "rust"}},
    {"id": 4, "tool": "select", "args": {"source": 1, "class_name": "bearing"}},
    {"id": 5, "tool": "intersect", "args": {"mask_a": 4, "mask_b": 3}},
    {"id": 6, "tool": "count_pixels", "args": {"mask": 5}},
    {"id": 7, "tool": "select", "args": {"source": 1, "class_name": "bracing"}},
    {"id": 8, "tool": "intersect", "args": {"mask_a": 7, "mask_b": 3}},
    {"id": 9, "tool": "count_pixels", "args": {"mask": 8}},
    {"id": 10, "tool": "select", "args": {"source": 1, "class_name": "deck"}},
    {"id": 11, "tool": "intersect", "args": {"mask_a": 10, "mask_b": 3}},
    {"id": 12, "tool": "count_pixels", "args": {"mask": 11}},
    {"id": 13, "tool": "select", "args": {"source": 1, "class_name": "floor_beam"}},
    {"id": 14, "tool": "intersect", "args": {"mask_a": 13, "mask_b": 3}},
    {"id": 15, "tool": "count_pixels", "args": {"mask": 14}},
    {"id": 16, "tool": "select", "args": {"source": 1, "class_name": "girder"}},
    {"id": 17, "tool": "intersect", "args": {"mask_a": 16, "mask_b": 3}},
    {"id": 18, "tool": "count_pixels", "args": {"mask": 17}},
    {"id": 19, "tool": "select", "args": {"source": 1, "class_name": "substructure"}},
    {"id": 20, "tool": "intersect", "args": {"mask_a": 19, "mask_b": 3}},
    {"id": 21, "tool": "count_pixels", "args": {"mask": 20}},
    {"id": 22, "tool": "select", "args": {"source": 1, "class_name": "girder"}}
  ],
  "output": 22
}

Example:
Question: "Which elements have rust? Segment them."
{
  "reasoning": "Enumerate EVERY element, intersect each with rust, and read back the actual pixel count from count_pixels for each one -- do not skip any element and do not assume the answer. Suppose the counts come back as: bearing=18275, bracing=0, deck=0, floor_beam=0, girder=48151, substructure=9. Any count greater than zero means that element has rust, so bearing (18275), girder (48151), and substructure (9) qualify, while bracing, deck, and floor_beam (all 0) do not. The answer is the union of the WHOLE bearing, girder, and substructure regions (not the rust-intersection masks used for the test).",
  "steps": [
    {"id": 1, "tool": "segment_elements", "args": {}},
    {"id": 2, "tool": "segment_rust", "args": {}},
    {"id": 3, "tool": "select", "args": {"source": 2, "class_name": "rust"}},
    {"id": 4, "tool": "select", "args": {"source": 1, "class_name": "bearing"}},
    {"id": 5, "tool": "intersect", "args": {"mask_a": 4, "mask_b": 3}},
    {"id": 6, "tool": "count_pixels", "args": {"mask": 5}},
    {"id": 7, "tool": "select", "args": {"source": 1, "class_name": "bracing"}},
    {"id": 8, "tool": "intersect", "args": {"mask_a": 7, "mask_b": 3}},
    {"id": 9, "tool": "count_pixels", "args": {"mask": 8}},
    {"id": 10, "tool": "select", "args": {"source": 1, "class_name": "deck"}},
    {"id": 11, "tool": "intersect", "args": {"mask_a": 10, "mask_b": 3}},
    {"id": 12, "tool": "count_pixels", "args": {"mask": 11}},
    {"id": 13, "tool": "select", "args": {"source": 1, "class_name": "floor_beam"}},
    {"id": 14, "tool": "intersect", "args": {"mask_a": 13, "mask_b": 3}},
    {"id": 15, "tool": "count_pixels", "args": {"mask": 14}},
    {"id": 16, "tool": "select", "args": {"source": 1, "class_name": "girder"}},
    {"id": 17, "tool": "intersect", "args": {"mask_a": 16, "mask_b": 3}},
    {"id": 18, "tool": "count_pixels", "args": {"mask": 17}},
    {"id": 19, "tool": "select", "args": {"source": 1, "class_name": "substructure"}},
    {"id": 20, "tool": "intersect", "args": {"mask_a": 19, "mask_b": 3}},
    {"id": 21, "tool": "count_pixels", "args": {"mask": 20}},
    {"id": 22, "tool": "union", "args": {"mask_a": 4, "mask_b": 16}},
    {"id": 23, "tool": "union", "args": {"mask_a": 22, "mask_b": 19}}
  ],
  "output": 23
}
"""

EX_C_PLAIN = """
Example (comparison -- enumerate EVERY element before deciding):
Question: "Which element has the most rust? Segment it."
{
  "steps": [
    {"id": 1, "tool": "segment_elements", "args": {}},
    {"id": 2, "tool": "segment_rust", "args": {}},
    {"id": 3, "tool": "select", "args": {"source": 2, "class_name": "rust"}},
    {"id": 4, "tool": "select", "args": {"source": 1, "class_name": "bearing"}},
    {"id": 5, "tool": "intersect", "args": {"mask_a": 4, "mask_b": 3}},
    {"id": 6, "tool": "count_pixels", "args": {"mask": 5}},
    {"id": 7, "tool": "select", "args": {"source": 1, "class_name": "bracing"}},
    {"id": 8, "tool": "intersect", "args": {"mask_a": 7, "mask_b": 3}},
    {"id": 9, "tool": "count_pixels", "args": {"mask": 8}},
    {"id": 10, "tool": "select", "args": {"source": 1, "class_name": "deck"}},
    {"id": 11, "tool": "intersect", "args": {"mask_a": 10, "mask_b": 3}},
    {"id": 12, "tool": "count_pixels", "args": {"mask": 11}},
    {"id": 13, "tool": "select", "args": {"source": 1, "class_name": "floor_beam"}},
    {"id": 14, "tool": "intersect", "args": {"mask_a": 13, "mask_b": 3}},
    {"id": 15, "tool": "count_pixels", "args": {"mask": 14}},
    {"id": 16, "tool": "select", "args": {"source": 1, "class_name": "girder"}},
    {"id": 17, "tool": "intersect", "args": {"mask_a": 16, "mask_b": 3}},
    {"id": 18, "tool": "count_pixels", "args": {"mask": 17}},
    {"id": 19, "tool": "select", "args": {"source": 1, "class_name": "substructure"}},
    {"id": 20, "tool": "intersect", "args": {"mask_a": 19, "mask_b": 3}},
    {"id": 21, "tool": "count_pixels", "args": {"mask": 20}},
    {"id": 22, "tool": "select", "args": {"source": 1, "class_name": "girder"}}
  ],
  "output": 22
}

Example (which elements have rust -- enumerate EVERY element before deciding):
Question: "Which elements have rust? Segment them."
{
  "steps": [
    {"id": 1, "tool": "segment_elements", "args": {}},
    {"id": 2, "tool": "segment_rust", "args": {}},
    {"id": 3, "tool": "select", "args": {"source": 2, "class_name": "rust"}},
    {"id": 4, "tool": "select", "args": {"source": 1, "class_name": "bearing"}},
    {"id": 5, "tool": "intersect", "args": {"mask_a": 4, "mask_b": 3}},
    {"id": 6, "tool": "count_pixels", "args": {"mask": 5}},
    {"id": 7, "tool": "select", "args": {"source": 1, "class_name": "bracing"}},
    {"id": 8, "tool": "intersect", "args": {"mask_a": 7, "mask_b": 3}},
    {"id": 9, "tool": "count_pixels", "args": {"mask": 8}},
    {"id": 10, "tool": "select", "args": {"source": 1, "class_name": "deck"}},
    {"id": 11, "tool": "intersect", "args": {"mask_a": 10, "mask_b": 3}},
    {"id": 12, "tool": "count_pixels", "args": {"mask": 11}},
    {"id": 13, "tool": "select", "args": {"source": 1, "class_name": "floor_beam"}},
    {"id": 14, "tool": "intersect", "args": {"mask_a": 13, "mask_b": 3}},
    {"id": 15, "tool": "count_pixels", "args": {"mask": 14}},
    {"id": 16, "tool": "select", "args": {"source": 1, "class_name": "girder"}},
    {"id": 17, "tool": "intersect", "args": {"mask_a": 16, "mask_b": 3}},
    {"id": 18, "tool": "count_pixels", "args": {"mask": 17}},
    {"id": 19, "tool": "select", "args": {"source": 1, "class_name": "substructure"}},
    {"id": 20, "tool": "intersect", "args": {"mask_a": 19, "mask_b": 3}},
    {"id": 21, "tool": "count_pixels", "args": {"mask": 20}},
    {"id": 22, "tool": "union", "args": {"mask_a": 4, "mask_b": 16}},
    {"id": 23, "tool": "union", "args": {"mask_a": 22, "mask_b": 19}}
  ],
  "output": 23
}
"""

QUESTION_LINE = '\nQuestion: "{question}"\n'


# ------------------------------------------------------------
# Variant assembly
# ------------------------------------------------------------
def _build(variant, doc, examples_kind, cot, no_visual=True):
    """
    examples_kind: "" (none) | "ab" (A/B only) | "abc" (A/B/C)
    cot: if True -> CoT scaffold + reasoning-bearing examples + reasoning output
         if False -> no scaffold + steps-only examples + no-reasoning output
    no_visual: if True -> append the rule forbidding visual-impression decisions
    """
    rules = RULES + (NO_VISUAL_RULE if no_visual else "")
    parts = [HEADER, "\n", doc, rules]
    if cot:
        parts.append(COT_SCAFFOLD)

    if examples_kind == "ab":
        parts.append(EX_AB_COT if cot else EX_AB_PLAIN)
    elif examples_kind == "abc":
        parts.append((EX_AB_COT + EX_C_COT) if cot else (EX_AB_PLAIN + EX_C_PLAIN))

    parts.append(OUTPUT_FORMAT_COT if cot else OUTPUT_FORMAT_NO_COT)
    parts.append(QUESTION_LINE)
    return "".join(parts)


VARIANTS = {
    "zero_shot":           dict(doc=TOOL_DOC,         examples_kind="",    cot=False, no_visual=True),
    "few_shot":            dict(doc=TOOL_DOC,         examples_kind="abc", cot=False, no_visual=True),
    "zero_shot_cot":       dict(doc=TOOL_DOC,         examples_kind="",    cot=True,  no_visual=True),
    "few_shot_cot":        dict(doc=TOOL_DOC,         examples_kind="abc", cot=True,  no_visual=True),
    "minimal_doc":         dict(doc=TOOL_DOC_MINIMAL, examples_kind="",    cot=False, no_visual=True),
    "few_shot_no_c":       dict(doc=TOOL_DOC,         examples_kind="ab",  cot=False, no_visual=True),
    # ablation: same as few_shot_cot but WITHOUT the no-visual rule, to measure
    # the rule's effect (compare against few_shot_cot)
    "few_shot_cot_novisual": dict(doc=TOOL_DOC,       examples_kind="abc", cot=True,  no_visual=False),
}


class BridgeVLM:
    def __init__(self, model_name="gemini-2.5-flash", variant="zero_shot"):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. "
                             f"Choose from: {list(VARIANTS)}")
        self.model = genai.GenerativeModel(model_name)
        self.variant = variant
        cfg = VARIANTS[variant]
        self.template = _build(variant, cfg["doc"], cfg["examples_kind"],
                               cfg["cot"], cfg.get("no_visual", True))

    def build_prompt(self, question):
        # Use literal replacement (not str.format) because the templates contain
        # JSON examples with { } braces that str.format would misinterpret.
        return (self.template
                .replace("{elements}", str(list(ELEMENT_MAP.keys())))
                .replace("{question}", question))

    def plan(self, question, image_pil=None):
        prompt = self.build_prompt(question)
        try:
            if image_pil is not None:
                res = self.model.generate_content([prompt, image_pil])
            else:
                res = self.model.generate_content(prompt)
            raw = res.text
        except Exception as e:
            return {"reasoning": f"VLM error: {e}", "steps": [], "output": None}, ""
        return self._parse_json(raw), raw

    @staticmethod
    def _parse_json(text):
        if not text:
            return {"reasoning": "", "steps": [], "output": None}
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not m:
            m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                obj.setdefault("reasoning", "")
                obj.setdefault("steps", [])
                obj.setdefault("output", None)
                return obj
            except Exception:
                pass
        return {"reasoning": "", "steps": [], "output": None}


# ==============================================================================
# AGENTIC (multi-turn) VLM
# ==============================================================================
# The single-shot BridgeVLM above has a structural blind spot: it must write
# the ENTIRE plan -- including which elements to union together for "which
# elements have rust" -- before any tool has actually run, so it has no real
# count_pixels numbers to base that decision on. It can only guess, and on
# class-C questions (comparisons across elements) it visibly does: plans come
# back with fabricated numbers in the reasoning text, or with no usable
# "output" at all because the model can't commit to an answer it doesn't know.
#
# BridgeVLMAgentic fixes this by using Gemini's native function calling in a
# real multi-turn loop: the model calls ONE tool at a time, the tool actually
# runs against this image's CNN output, and the model SEES the real return
# value (especially count_pixels) before deciding its next call. There is no
# turn limit -- the loop runs until the model calls finish().
# ==============================================================================

AGENTIC_TOOLS = [{
    "function_declarations": [
        {
            "name": "segment_elements",
            "description": ("Run the CNN to label every pixel as one of the bridge "
                            "elements. Returns an internal element-map id; use "
                            "select() on it to extract one element."),
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "segment_rust",
            "description": ("Run the CNN to find all rusty/corroded pixels in the "
                            "whole image. Returns an internal rust-map id; use "
                            "select() with class_name='rust' to extract it."),
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "select",
            "description": ("Extract the region of one class as a mask, from a "
                            "previous segment_elements or segment_rust result."),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "integer",
                              "description": "step id of a previous segment_elements/segment_rust call"},
                    "class_name": {"type": "string",
                                  "description": ("one of bearing, bracing, deck, floor_beam, "
                                                  "girder, substructure (element source) or "
                                                  "'rust' (rust source)")},
                },
                "required": ["source", "class_name"],
            },
        },
        {
            "name": "intersect",
            "description": ("Pixels present in BOTH masks (e.g. rust ON the girder = "
                            "intersect(girder_mask, rust_mask))."),
            "parameters": {
                "type": "object",
                "properties": {
                    "mask_a": {"type": "integer", "description": "step id of a previous mask result"},
                    "mask_b": {"type": "integer", "description": "step id of a previous mask result"},
                },
                "required": ["mask_a", "mask_b"],
            },
        },
        {
            "name": "union",
            "description": "Pixels present in EITHER mask (e.g. combine two elements).",
            "parameters": {
                "type": "object",
                "properties": {
                    "mask_a": {"type": "integer", "description": "step id of a previous mask result"},
                    "mask_b": {"type": "integer", "description": "step id of a previous mask result"},
                },
                "required": ["mask_a", "mask_b"],
            },
        },
        {
            "name": "subtract",
            "description": "Pixels in mask_a but NOT in mask_b.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mask_a": {"type": "integer", "description": "step id of a previous mask result"},
                    "mask_b": {"type": "integer", "description": "step id of a previous mask result"},
                },
                "required": ["mask_a", "mask_b"],
            },
        },
        {
            "name": "count_pixels",
            "description": ("Count the foreground pixels of a mask. Returns an integer "
                            "you must read and reason about before deciding your next "
                            "step -- never guess this number."),
            "parameters": {
                "type": "object",
                "properties": {
                    "mask": {"type": "integer", "description": "step id of a previous mask result"},
                },
                "required": ["mask"],
            },
        },
        {
            "name": "finish",
            "description": ("Call this ONCE you have determined the final answer mask, "
                            "based on REAL tool results (not a guess). Ends the session. "
                            "Always precede this call with a short plain-text sentence "
                            "summarizing the real numbers you compared and why this step "
                            "is the answer."),
            "parameters": {
                "type": "object",
                "properties": {
                    "output": {"type": "integer",
                              "description": "the step id (returned by a previous tool call) holding the final answer mask"},
                },
                "required": ["output"],
            },
        },
    ]
}]

AGENTIC_GUIDANCE = """
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
"""


# Worked example for the agentic FEW-SHOT variant: not a JSON blob (there's no
# single static plan in agentic mode) but a TRANSCRIPT showing the call ->
# real-result -> next-call rhythm, including the existence-check-then-skip
# pattern for class-C comparisons. Numbers are made-up/illustrative; the
# prompt explicitly tells the model not to reuse them.
AGENTIC_FEWSHOT_EXAMPLE = """
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
  call segment_elements() -> {"id": 1}
  call segment_rust() -> {"id": 2}
  call select(source=2, class_name="rust") -> {"id": 3}
  call select(source=1, class_name="bearing") -> {"id": 4}
  call intersect(mask_a=4, mask_b=3) -> {"id": 5}
  call count_pixels(mask=5) -> {"id": 5, "count": 18275}
  call select(source=1, class_name="bracing") -> {"id": 6}
  call intersect(mask_a=6, mask_b=3) -> {"id": 7}
  call count_pixels(mask=7) -> {"id": 7, "count": 0}
  call select(source=1, class_name="deck") -> {"id": 8}
  call intersect(mask_a=8, mask_b=3) -> {"id": 9}
  call count_pixels(mask=9) -> {"id": 9, "count": 0}
  call select(source=1, class_name="floor_beam") -> {"id": 10}
  call intersect(mask_a=10, mask_b=3) -> {"id": 11}
  call count_pixels(mask=11) -> {"id": 11, "count": 0}
  call select(source=1, class_name="girder") -> {"id": 12}
  call intersect(mask_a=12, mask_b=3) -> {"id": 13}
  call count_pixels(mask=13) -> {"id": 13, "count": 48151}
  call select(source=1, class_name="substructure") -> {"id": 14}
  call intersect(mask_a=14, mask_b=3) -> {"id": 15}
  call count_pixels(mask=15) -> {"id": 15, "count": 9}
  -- comparing 18275, 0, 0, 0, 48151, 9: girder's 48151 is the largest --
  call finish(output=12)   # the WHOLE girder select (step 12), NOT the
                           # intersection mask (step 13) used for comparison

Question: "Which elements have rust? Segment them."
  (same per-element enumeration as above, reusing the same counts: bearing=18275,
  bracing=0, deck=0, floor_beam=0, girder=48151, substructure=9)
  -- any count > 0 means that element has rust: bearing, girder, and
  substructure qualify; bracing, deck, floor_beam (all 0) do not --
  call union(mask_a=4, mask_b=12) -> {"id": 16}
  call union(mask_a=16, mask_b=14) -> {"id": 17}
  call finish(output=17)   # union of the WHOLE bearing, girder, substructure
                           # selects, NOT the intersection masks

Question: "Which elements have NO rust? Segment them."
  (same per-element enumeration as above, reusing the same counts: bearing=18275,
  bracing=0, deck=0, floor_beam=0, girder=48151, substructure=9)
  -- any count == 0 means that element is rust-free: bracing, deck, and
  floor_beam qualify; bearing, girder, and substructure (all > 0) do not --
  call union(mask_a=7, mask_b=8) -> {"id": 18}
  call union(mask_a=18, mask_b=10) -> {"id": 19}
  call finish(output=19)   # union of the WHOLE bracing, deck, floor_beam
                           # selects, NOT the intersection masks
"""


def _build_agentic_prompt(no_visual=True, few_shot=False):
    rules = RULES + (NO_VISUAL_RULE if no_visual else "")
    parts = [HEADER, "\n", TOOL_DOC, rules, AGENTIC_GUIDANCE]
    if few_shot:
        parts.append(AGENTIC_FEWSHOT_EXAMPLE)
    parts.append(QUESTION_LINE)
    return "".join(parts)


AGENTIC_VARIANTS = {
    "zero_shot": dict(few_shot=False),
    "few_shot":  dict(few_shot=True),
}


class BridgeVLMAgentic:
    """
    Multi-turn agentic planner. Use plan(question, image_pil, cnn) -- it runs
    the full tool-calling loop against the given image/cnn and returns a plan
    dict with the same {"reasoning", "steps", "output"} shape the single-shot
    BridgeVLM produces, so saved plan files and plan_accuracy.py stay
    compatible. It also returns the ToolExecutor (already populated with every
    real result), so the caller can get the predicted mask via
    executor.finalize(plan["output"]) without re-running anything.

    variant: "zero_shot" (just the instructions) or "few_shot" (instructions +
    a worked transcript showing the call -> real-result -> next-call rhythm).
    This mirrors the single-shot BridgeVLM's zero_shot/few_shot ablation, but
    for the agentic architecture.
    """
    def __init__(self, model_name="gemini-2.5-flash", variant="zero_shot", no_visual=True):
        if variant not in AGENTIC_VARIANTS:
            raise ValueError(f"Unknown agentic variant '{variant}'. "
                             f"Choose from: {list(AGENTIC_VARIANTS)}")
        self.variant = variant
        self.model = genai.GenerativeModel(model_name, tools=AGENTIC_TOOLS)
        self.template = _build_agentic_prompt(no_visual, **AGENTIC_VARIANTS[variant])

    def build_prompt(self, question):
        return (self.template
                .replace("{elements}", str(list(ELEMENT_MAP.keys())))
                .replace("{question}", question))

    @staticmethod
    def _normalize_args(args):
        """Gemini function-call args sometimes come back as floats for
        integer-typed parameters; cast the step-id-valued ones back to int so
        they match the integer keys ToolExecutor.results is keyed by."""
        out = dict(args)
        for k in ("source", "mask", "mask_a", "mask_b"):
            if k in out:
                try:
                    out[k] = int(out[k])
                except (TypeError, ValueError):
                    pass
        return out

    def plan(self, question, image_pil, cnn):
        executor = ToolExecutor(cnn, image_pil)
        prompt = self.build_prompt(question)
        chat = self.model.start_chat()
        steps, transcript = [], []
        output_id = None
        content = [prompt, image_pil]

        while True:
            try:
                response = chat.send_message(content)
            except Exception as e:
                transcript.append(f"VLM error: {e}")
                break

            parts = response.candidates[0].content.parts
            fn_calls = [p.function_call for p in parts
                       if getattr(p, "function_call", None) and p.function_call.name]
            texts = [p.text for p in parts if getattr(p, "text", "")]
            transcript.extend(texts)

            if not fn_calls:
                break  # model stopped without calling finish() -- incomplete plan

            response_parts = []
            finished = False
            for fc in fn_calls:
                name = fc.name
                args = self._normalize_args(dict(fc.args))
                if name == "finish":
                    try:
                        output_id = int(args.get("output"))
                    except (TypeError, ValueError):
                        output_id = None
                    finished = True
                    break
                sid, info = executor.call(name, args)
                steps.append({"id": sid, "tool": name, "args": args})
                response_parts.append(genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(name=name, response=info)))
            if finished:
                break
            content = response_parts

        plan = {"reasoning": "\n".join(transcript), "steps": steps, "output": output_id}
        return plan, "\n".join(transcript), executor


# ============================================================
# OpenAI backend (gpt-4.1, gpt-5.5, o3, o4-mini, …)
# ============================================================
# Convert AGENTIC_TOOLS (Gemini format) → OpenAI function-calling format
_OPENAI_TOOLS = []
for _fd in AGENTIC_TOOLS[0]["function_declarations"]:
    _OPENAI_TOOLS.append({
        "type": "function",
        "function": {
            "name": _fd["name"],
            "description": _fd["description"],
            "parameters": _fd.get("parameters", {"type": "object", "properties": {}}),
        }
    })


def _pil_to_b64(pil_image):
    import base64, io
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


class AgenticVLMOpenAI:
    """Same agentic loop as AgenticVLM but using the OpenAI API."""

    def __init__(self, model_name="gpt-4.1", variant="few_shot", api_key=None):
        if variant not in AGENTIC_VARIANTS:
            raise ValueError(f"Unknown agentic variant '{variant}'. "
                             f"Choose from: {list(AGENTIC_VARIANTS)}")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.variant = variant
        self.template = _build_agentic_prompt(no_visual=True, **AGENTIC_VARIANTS[variant])

    def build_prompt(self, question):
        return (self.template
                .replace("{elements}", str(list(ELEMENT_MAP.keys())))
                .replace("{question}", question))

    def plan(self, question, image_pil, cnn):
        executor = ToolExecutor(cnn, image_pil)
        system_prompt = self.build_prompt(question)
        b64 = _pil_to_b64(image_pil)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": question},
            ]},
        ]

        steps, transcript = [], []
        output_id = None

        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=_OPENAI_TOOLS,
                    tool_choice="auto",
                )
            except Exception as e:
                transcript.append(f"VLM error: {e}")
                break

            msg = resp.choices[0].message
            if msg.content:
                transcript.append(msg.content)

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                break

            # Append assistant message with tool_calls
            messages.append({"role": "assistant",
                              "content": msg.content or "",
                              "tool_calls": [
                                  {"id": tc.id, "type": "function",
                                   "function": {"name": tc.function.name,
                                                "arguments": tc.function.arguments}}
                                  for tc in tool_calls
                              ]})

            finished = False
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                # cast integer step-id args
                for k in ("source", "mask", "mask_a", "mask_b"):
                    if k in args:
                        try:
                            args[k] = int(args[k])
                        except (TypeError, ValueError):
                            pass

                if name == "finish":
                    try:
                        output_id = int(args.get("output"))
                    except (TypeError, ValueError):
                        output_id = None
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": "done"})
                    finished = True
                    break

                sid, info = executor.call(name, args)
                steps.append({"id": sid, "tool": name, "args": args})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(info)})

            if finished:
                break

        plan = {"reasoning": "\n".join(transcript), "steps": steps, "output": output_id}
        return plan, "\n".join(transcript), executor