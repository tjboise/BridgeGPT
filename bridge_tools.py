"""
bridge_tools.py
==============================================================================
Atomic tool library for BridgeGPT reasoning-segmentation.

Design philosophy
-----------------
Only MINIMAL, atomic operations are exposed. There are NO high-level shortcuts
like "select_most_corroded_element" or "for_each_element". The VLM must compose
these primitives itself, step by step, to answer a question. Harder questions
(e.g. "which element has the most rust") force the VLM to manually enumerate
elements, intersect each with rust, compare pixel counts in its reasoning, and
select the winner -- so a weak prompt will visibly fail on them.

The execution model is a simple data-flow graph:
  - Each step produces a result, addressed by its integer step id.
  - Later steps reference earlier results by id.
  - Mask results are 2-D boolean/uint8 arrays at the image's native resolution.
  - count_pixels returns a scalar (int), used by the VLM only as information it
    reads back in its reasoning; the VLM must decide subsequent steps itself.
  - The final answer mask is the step id named in the plan's "output" field.

Encoding of the produced mask
------------------------------
Tool results that are masks are binary (0/1) over the image. The final output
mask is also returned binary; cIoU is computed against the binary GT mask
(any class id > 0 counts as foreground), matching generate_gt.py.

CNN interface
-------------
segment_elements / segment_rust call the AECIF-Net wrapper
HRnet_Segmentation.get_miou_png(image_pil) -> (element_pil, defect_pil),
where the element map uses ids 1..6 and the defect map uses 1 for rust.
"""
import numpy as np

# Element id <-> name (must match GT encoding in generate_gt.py)
ELEMENT_MAP = {
    "bearing": 1, "bracing": 2, "deck": 3,
    "floor_beam": 4, "girder": 5, "substructure": 6,
}
ELEMENT_MAP_REV = {v: k for k, v in ELEMENT_MAP.items()}


# ============================================================
# TOOL DOCUMENTATION (this string is shown to the VLM verbatim)
# ============================================================
TOOL_DOC = """
You control a bridge-inspection vision system through a small set of ATOMIC
tools. Each tool does ONE simple thing. You must COMPOSE them, step by step,
to answer the question. There are no shortcut tools: anything complex (for
example, finding which element is the most rusted) must be built by hand out of
these primitives.

Execution model
---------------
- You output a list of steps. Each step has an integer "id", a "tool" name, and
  "args". A step's result is referred to by its id.
- A later step references an earlier result by putting that id as an arg value.
- A mask is a region of the image. A count is a number.
- At the end you name which step id holds the final answer mask ("output").

Tools
-----
1. segment_elements()
     Run the CNN to label every pixel as one of the bridge elements.
     Returns: an element map (internal). Use `select` to extract one element.

2. segment_rust()
     Run the CNN to find all rusty/corroded pixels in the whole image.
     Returns: a rust map (internal). Use `select` with class_name "rust".

3. select(source, class_name)
     Extract the region of one class as a mask.
     - source: the id of a previous segment_elements or segment_rust step.
     - class_name: one of ["bearing","bracing","deck","floor_beam","girder",
       "substructure"] for an element source, or "rust" for a rust source.
     Returns: a binary mask of that class.

4. intersect(mask_a, mask_b)
     Pixels present in BOTH masks. (e.g. rust ON the girder = intersect of the
     girder mask and the rust mask.)
     Returns: a binary mask.

5. union(mask_a, mask_b)
     Pixels present in EITHER mask. (e.g. combine two elements.)
     Returns: a binary mask.

6. subtract(mask_a, mask_b)
     Pixels in mask_a but NOT in mask_b.
     Returns: a binary mask.

7. count_pixels(mask)
     Count the foreground pixels of a mask.
     Returns: an integer. Use this number in your reasoning to compare regions
     (for example, to decide which element has more rust). You must do the
     comparison yourself in your reasoning and then choose the next steps.
"""


# ============================================================
# MINIMAL TOOL DOC (low-detail end of the documentation ablation)
# Same 7 tools, but only one-line descriptions, no execution-model explanation,
# no output-format example. Weaker prompts should struggle more with this.
# ============================================================
TOOL_DOC_MINIMAL = """
Tools: segment_elements(), segment_rust(), select(source, class_name),
intersect(mask_a, mask_b), union(mask_a, mask_b), subtract(mask_a, mask_b),
count_pixels(mask).
"""


# ============================================================
# OUTPUT-FORMAT SPECS (appended per variant by bridge_vlm.py)
# A CoT variant asks for a "reasoning" field; a no-CoT variant does NOT,
# so the model emits steps directly. This keeps the CoT ablation clean.
# ============================================================
OUTPUT_FORMAT_COT = """
Output format (JSON only):
{
  "reasoning": "<your step-by-step thinking>",
  "steps": [
    {"id": 1, "tool": "segment_elements", "args": {}},
    {"id": 2, "tool": "segment_rust", "args": {}},
    {"id": 3, "tool": "select", "args": {"source": 1, "class_name": "girder"}},
    {"id": 4, "tool": "select", "args": {"source": 2, "class_name": "rust"}},
    {"id": 5, "tool": "intersect", "args": {"mask_a": 3, "mask_b": 4}}
  ],
  "output": 5
}
"""

OUTPUT_FORMAT_NO_COT = """
Output format (JSON only). Do NOT include any reasoning or explanation; output
the steps directly:
{
  "steps": [
    {"id": 1, "tool": "segment_elements", "args": {}},
    {"id": 2, "tool": "segment_rust", "args": {}},
    {"id": 3, "tool": "select", "args": {"source": 1, "class_name": "girder"}},
    {"id": 4, "tool": "select", "args": {"source": 2, "class_name": "rust"}},
    {"id": 5, "tool": "intersect", "args": {"mask_a": 3, "mask_b": 4}}
  ],
  "output": 5
}
"""


# ============================================================
# EXECUTOR
# ============================================================
class ToolExecutor:
    """
    Runs a plan (list of steps) against one image using the CNN wrapper.

    cnn: an object with get_miou_png(image_pil) -> (element_pil, defect_pil).
    image_pil: the PIL image for this question.

    Results are stored per step id. Mask results are uint8 {0,1} arrays sized
    to the image. count_pixels results are python ints.
    """
    def __init__(self, cnn, image_pil):
        self.cnn = cnn
        self.image_pil = image_pil
        self.h, self.w = np.array(image_pil).shape[:2]
        self._element_map = None      # cached CNN element output (ids 0..6)
        self._rust_map = None         # cached CNN rust output (0/1)
        self.results = {}             # step id -> mask (uint8) or int
        self.log = []                 # human-readable trace
        self._next_id = 1             # auto-incrementing id for agentic mode

    # ---- CNN (run once, cached) ----
    def _run_cnn(self):
        if self._element_map is None:
            elem_pil, defect_pil = self.cnn.get_miou_png(self.image_pil)
            self._element_map = np.array(elem_pil)
            rust = np.array(defect_pil)
            self._rust_map = (rust == 1).astype(np.uint8)

    def _empty(self):
        return np.zeros((self.h, self.w), dtype=np.uint8)

    def _as_mask(self, ref):
        """Resolve an arg that should be a mask (by step id)."""
        val = self.results.get(ref)
        if isinstance(val, np.ndarray):
            return (val > 0).astype(np.uint8)
        # referencing a raw segment_* step: treat as its map's foreground
        return self._empty()

    # ---- atomic tools ----
    def segment_elements(self, step_id, args):
        self._run_cnn()
        # store a sentinel; `select` reads the cached element map
        self.results[step_id] = ("ELEMENT_SOURCE",)
        self.log.append(f"[{step_id}] segment_elements()")

    def segment_rust(self, step_id, args):
        self._run_cnn()
        self.results[step_id] = ("RUST_SOURCE",)
        self.log.append(f"[{step_id}] segment_rust()")

    def select(self, step_id, args):
        source = args.get("source")
        class_name = str(args.get("class_name", "")).lower()
        src = self.results.get(source)
        mask = self._empty()
        if src == ("ELEMENT_SOURCE",):
            eid = ELEMENT_MAP.get(class_name, 0)
            if eid > 0 and self._element_map is not None:
                mask = (self._element_map == eid).astype(np.uint8)
        elif src == ("RUST_SOURCE",):
            if "rust" in class_name and self._rust_map is not None:
                mask = self._rust_map.copy()
        self.results[step_id] = mask
        self.log.append(f"[{step_id}] select(source={source}, class='{class_name}') "
                        f"-> {int(mask.sum())} px")

    def intersect(self, step_id, args):
        a = self._as_mask(args.get("mask_a"))
        b = self._as_mask(args.get("mask_b"))
        mask = np.logical_and(a, b).astype(np.uint8)
        self.results[step_id] = mask
        self.log.append(f"[{step_id}] intersect -> {int(mask.sum())} px")

    def union(self, step_id, args):
        a = self._as_mask(args.get("mask_a"))
        b = self._as_mask(args.get("mask_b"))
        mask = np.logical_or(a, b).astype(np.uint8)
        self.results[step_id] = mask
        self.log.append(f"[{step_id}] union -> {int(mask.sum())} px")

    def subtract(self, step_id, args):
        a = self._as_mask(args.get("mask_a"))
        b = self._as_mask(args.get("mask_b"))
        mask = np.logical_and(a, np.logical_not(b)).astype(np.uint8)
        self.results[step_id] = mask
        self.log.append(f"[{step_id}] subtract -> {int(mask.sum())} px")

    def count_pixels(self, step_id, args):
        m = self._as_mask(args.get("mask"))
        n = int(m.sum())
        self.results[step_id] = n
        self.log.append(f"[{step_id}] count_pixels -> {n}")

    _DISPATCH = {
        "segment_elements": segment_elements,
        "segment_rust": segment_rust,
        "select": select,
        "intersect": intersect,
        "union": union,
        "subtract": subtract,
        "count_pixels": count_pixels,
    }

    def run(self, plan):
        """
        Execute a plan dict {steps:[...], output:id}. Returns (final_mask, log).
        Unknown tools / bad references degrade to empty masks rather than crash,
        so a malformed plan simply scores low cIoU (which is the point).
        """
        steps = plan.get("steps", [])
        if not isinstance(steps, list):
            steps = []
        for step in steps:
            if not isinstance(step, dict):
                self.log.append(f"SKIP malformed step (not an object): {str(step)[:60]}")
                continue
            sid = step.get("id")
            tool = step.get("tool", "")
            args = step.get("args", {})
            if not isinstance(args, dict):
                args = {}
            fn = self._DISPATCH.get(tool)
            if fn is None:
                self.log.append(f"[{sid}] UNKNOWN TOOL '{tool}' -> skipped")
                self.results[sid] = self._empty()
                continue
            try:
                fn(self, sid, args)
            except Exception as e:
                self.log.append(f"[{sid}] ERROR in {tool}: {e}")
                self.results[sid] = self._empty()

        out_id = plan.get("output")
        final = self.results.get(out_id)
        if not isinstance(final, np.ndarray):
            final = self._empty()
        return (final > 0).astype(np.uint8), self.log

    # ============================================================
    # AGENTIC (incremental) EXECUTION
    # ============================================================
    # Unlike run(), which executes a whole pre-written plan at once, call()
    # executes ONE tool call immediately and hands back a small JSON-safe
    # result. This is what makes the multi-turn agentic VLM (BridgeVLMAgentic
    # in bridge_vlm.py) actually grounded: the model sees the REAL count_pixels
    # value before deciding its next move, instead of writing a full plan
    # upfront and guessing numbers that don't exist yet.
    def call(self, tool, args):
        """
        Execute one tool call now; auto-assigns the next integer step id.
        Returns (step_id, info):
          - count_pixels -> info = {"id": step_id, "count": <int>}
          - everything else -> info = {"id": step_id}  (no size info "for
            free" -- the model must call count_pixels explicitly, same as in
            batch mode, so this doesn't quietly change the tools' semantics)
          - unknown tool / execution error -> info = {"id": step_id, "error": "..."}
        """
        sid = self._next_id
        self._next_id += 1
        fn = self._DISPATCH.get(tool)
        if fn is None:
            self.log.append(f"[{sid}] UNKNOWN TOOL '{tool}' -> skipped")
            self.results[sid] = self._empty()
            return sid, {"id": sid, "error": f"unknown tool '{tool}'"}
        try:
            fn(self, sid, args)
        except Exception as e:
            self.log.append(f"[{sid}] ERROR in {tool}: {e}")
            self.results[sid] = self._empty()
            return sid, {"id": sid, "error": str(e)}
        if tool == "count_pixels":
            return sid, {"id": sid, "count": self.results.get(sid)}
        return sid, {"id": sid}

    def finalize(self, output_id):
        """Resolve the final answer mask from a step id (agentic mode).
        Same fallback behaviour as run(): a bad/missing id -> empty mask."""
        final = self.results.get(output_id)
        if not isinstance(final, np.ndarray):
            final = self._empty()
        return (final > 0).astype(np.uint8), self.log