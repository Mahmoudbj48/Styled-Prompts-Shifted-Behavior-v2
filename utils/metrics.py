"""
Evaluation metric utilities.

Functions:
    - compute_bleu: BLEU score for word-level similarity
    - compute_bertscore: BERTScore for semantic similarity
    - compute_confidence: Confidence metrics (Delta LogProb, Entropy Shift, JSD)
    - get_layer_activations: last-token activation from chosen layer (BatchEncoding-safe)
    - get_layer_activations_batch: batched last-token activations (BatchEncoding-safe)
    - compute_activation_similarity: cosine sim last-token activation
    - compute_activation_similarity_all_layers: per-layer cosine sims (BatchEncoding-safe)
    - reduce_activations_2d: PCA/t-SNE 2D projection
    - collect_activations_for_prompts: helper

Also includes:
    - Style mirroring judge via Gemini (LLM-as-judge)
    - BBQ bias metrics computation
    - HarmBench safety ASR computation via LlamaGuard3 outputs
"""

from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Callable

import numpy as np
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable, Sequence, Union, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from bert_score import score as bert_score
from sacrebleu import corpus_bleu
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# =============================================================================
# TEXT SIMILARITY METRICS
# =============================================================================

def compute_bleu(reference: str, candidate: str) -> float:
    """
    Compute BLEU score between reference and candidate text.
    Returns BLEU score (0-100).
    """
    bleu = corpus_bleu([candidate], [[reference]])
    return float(bleu.score)


def compute_bertscore(
        reference: str,
        candidate: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """
    Compute BERTScore (F1) between reference and candidate.
    Returns F1 (0-1).
    """
    P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False, device=device)
    return float(F1.item())


# =============================================================================
# TOKENIZATION HELPERS
# =============================================================================

def _render_user_prompt(tokenizer, prompt: str) -> str:
    """
    Render a user prompt with chat template if available, otherwise return raw prompt.
    Uses tokenize=False so we can batch-tokenize later consistently.
    """
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def _tokenize_prompts_batched(
        model,
        tokenizer,
        prompts: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize a batch of prompts into (input_ids, attention_mask) on model.device.
    Uses chat-template rendering if available.
    """
    device = model.device
    rendered = [_render_user_prompt(tokenizer, p) for p in prompts]

    enc = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    return input_ids, attention_mask


def _tokenize_prompt_single(
        model,
        tokenizer,
        prompt: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize a single prompt into (input_ids, attention_mask) on model.device.
    """
    input_ids, attention_mask = _tokenize_prompts_batched(model, tokenizer, [prompt])
    return input_ids, attention_mask


def _last_nonpad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Given attention_mask [B, L] with 1 for tokens and 0 for pads,
    return last non-pad token indices [B] (each in [0, L-1]).
    """
    # sum gives length (count of 1s) per row; last index = length - 1
    lengths = attention_mask.sum(dim=1)  # [B]
    last_idx = torch.clamp(lengths - 1, min=0).long()
    return last_idx


# =============================================================================
# CONFIDENCE METRICS
# =============================================================================

def compute_confidence(
        model,
        tokenizer,
        prompt_orig: str,
        prompt_pert: str,
        response_orig: str,
) -> Dict[str, float]:
    """
    Compute confidence metrics comparing original and perturbed prompts.

    Metrics:
        - delta_log_prob
        - entropy_shift
        - jsd_drift

    Notes:
      - Uses the SAME prompt rendering logic as generation (chat template when possible).
      - Teacher forcing is done by feeding (prompt_tokens + response_tokens).
    """
    device = model.device

    def _encode_prompt_and_response(prompt: str, response: str) -> Tuple[torch.Tensor, int, torch.Tensor]:
        rendered = _render_user_prompt(tokenizer, prompt)

        # Encode separately to find split point
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        resp_ids = tokenizer.encode(response, add_special_tokens=False)

        input_ids = torch.tensor([prompt_ids + resp_ids], device=device)
        prompt_len = len(prompt_ids)
        targets = torch.tensor(resp_ids, device=device)
        return input_ids, prompt_len, targets

    def get_response_logits(prompt: str, response: str) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, prompt_len, targets = _encode_prompt_and_response(prompt, response)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            all_logits = outputs.logits[0]  # [seq_len, vocab]

        # Predict token i+1 from position i:
        # response token 0 is predicted at position (prompt_len - 1)
        start_idx = max(prompt_len - 1, 0)
        end_idx = start_idx + len(targets)
        target_logits = all_logits[start_idx:end_idx]
        return target_logits, targets

    logits_orig, targets_orig = get_response_logits(prompt_orig, response_orig)
    logits_pert, targets_pert = get_response_logits(prompt_pert, response_orig)

    min_len = min(logits_orig.size(0), logits_pert.size(0), targets_orig.size(0))
    logits_orig = logits_orig[:min_len]
    logits_pert = logits_pert[:min_len]
    targets = targets_orig[:min_len]

    log_probs_orig = F.log_softmax(logits_orig, dim=-1)
    log_probs_pert = F.log_softmax(logits_pert, dim=-1)
    probs_orig = F.softmax(logits_orig, dim=-1)
    probs_pert = F.softmax(logits_pert, dim=-1)

    # Delta LogProb
    target_log_prob_orig = log_probs_orig.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_log_prob_pert = log_probs_pert.gather(1, targets.unsqueeze(1)).squeeze(1)
    seq_log_prob_orig = target_log_prob_orig.sum().item()
    seq_log_prob_pert = target_log_prob_pert.sum().item()
    delta_log_prob = seq_log_prob_orig - seq_log_prob_pert

    # Entropy shift
    entropy_orig = -(probs_orig * log_probs_orig).sum(dim=-1).mean().item()
    entropy_pert = -(probs_pert * log_probs_pert).sum(dim=-1).mean().item()
    entropy_shift = entropy_pert - entropy_orig

    # JSD
    m = 0.5 * (probs_orig + probs_pert)
    log_m = torch.log(m + 1e-10)
    kl_p_m = (probs_orig * (log_probs_orig - log_m)).sum(dim=-1).mean()
    kl_q_m = (probs_pert * (log_probs_pert - log_m)).sum(dim=-1).mean()
    jsd = 0.5 * (kl_p_m + kl_q_m).item()

    return {
        "delta_log_prob": float(delta_log_prob),
        "entropy_shift": float(entropy_shift),
        "jsd_drift": float(jsd),
    }


# =============================================================================
# ACTIVATION ANALYSIS
# =============================================================================

def get_layer_activations(
        model,
        tokenizer,
        prompt: str,
        layer_idx: int = -1,
) -> torch.Tensor:
    """
    Extract last-nonpad-token activation vector from a specific layer (single prompt).

    Returns:
        torch.Tensor: [hidden_dim]
    """
    acts = get_layer_activations_batch(model, tokenizer, [prompt], layer_idx=layer_idx)
    return acts[0]


def get_layer_activations_batch(
        model,
        tokenizer,
        prompts: List[str],
        layer_idx: int = -1,
) -> torch.Tensor:
    """
    Extract last-nonpad-token activation vectors from a specific layer (batched).

    Args:
        prompts: list[str]
        layer_idx: which hidden state layer to read (-1 last)

    Returns:
        torch.Tensor: [B, hidden_dim]
    """
    if len(prompts) == 0:
        # return empty tensor on correct device
        hidden = getattr(model.config, "hidden_size", 0) or 0
        return torch.empty((0, hidden), device=model.device)

    input_ids, attention_mask = _tokenize_prompts_batched(model, tokenizer, prompts)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hs = outputs.hidden_states[layer_idx]  # [B, L, H]
    last_idx = _last_nonpad_indices(attention_mask)  # [B]

    # gather last non-pad per sample
    bsz, _, hdim = hs.shape
    batch_arange = torch.arange(bsz, device=hs.device)
    acts = hs[batch_arange, last_idx, :]  # [B, H]
    return acts


def compute_activation_similarity(
        model,
        tokenizer,
        prompt_orig: str,
        prompt_pert: str,
        layer_idx: int = -1,
) -> float:
    """
    Cosine similarity between last-token activations at a specific layer (single pair).
    """
    act_orig = get_layer_activations(model, tokenizer, prompt_orig, layer_idx)
    act_pert = get_layer_activations(model, tokenizer, prompt_pert, layer_idx)
    return float(F.cosine_similarity(act_orig.unsqueeze(0), act_pert.unsqueeze(0)).item())


def compute_activation_similarity_batch(
        acts_a: torch.Tensor,
        acts_b: torch.Tensor,
) -> List[float]:
    """
    Given two activation matrices [B, H], return cosine similarity per row.
    """
    if acts_a.shape != acts_b.shape:
        raise ValueError(f"Shape mismatch: {acts_a.shape} vs {acts_b.shape}")
    sims = F.cosine_similarity(acts_a, acts_b, dim=1)  # [B]
    return [float(x) for x in sims.detach().cpu().tolist()]


def compute_activation_similarity_all_layers(
        model,
        tokenizer,
        prompt_orig: str,
        prompt_pert: str,
) -> Dict[str, Any]:
    """
    Compute activation similarity across all layers using last-nonpad-token reps.

    Returns:
        dict: mean_similarity, last_layer_similarity, per_layer_similarity
    """
    input_ids_orig, attention_mask_orig = _tokenize_prompt_single(model, tokenizer, prompt_orig)
    input_ids_pert, attention_mask_pert = _tokenize_prompt_single(model, tokenizer, prompt_pert)

    with torch.no_grad():
        outputs_orig = model(
            input_ids=input_ids_orig,
            attention_mask=attention_mask_orig,
            output_hidden_states=True,
        )
        outputs_pert = model(
            input_ids=input_ids_pert,
            attention_mask=attention_mask_pert,
            output_hidden_states=True,
        )

    last_idx_orig = _last_nonpad_indices(attention_mask_orig)[0].item()
    last_idx_pert = _last_nonpad_indices(attention_mask_pert)[0].item()

    per_layer = []
    num_layers = len(outputs_orig.hidden_states)

    for layer_idx in range(num_layers):
        a = outputs_orig.hidden_states[layer_idx][0, last_idx_orig, :]
        b = outputs_pert.hidden_states[layer_idx][0, last_idx_pert, :]
        sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        per_layer.append(float(sim))

    return {
        "mean_similarity": float(np.mean(per_layer)) if per_layer else float("nan"),
        "last_layer_similarity": float(per_layer[-1]) if per_layer else float("nan"),
        "per_layer_similarity": per_layer,
    }


def reduce_activations_2d(
        activations_list: List[Union[torch.Tensor, np.ndarray]],
        method: str = "pca",
        seed: int = 42,
) -> np.ndarray:
    """
    Reduce high-dimensional activations to 2D for visualization.
    """
    if len(activations_list) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if isinstance(activations_list[0], torch.Tensor):
        activations = torch.stack(activations_list).detach().cpu().numpy()
    else:
        activations = np.array(activations_list)

    if method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=seed)
    elif method.lower() == "tsne":
        n_samples = activations.shape[0]
        perplexity = min(30, max(1, n_samples - 1))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    return reducer.fit_transform(activations)


def collect_activations_for_prompts(
        model,
        tokenizer,
        prompts: List[str],
        layer_idx: int = -1,
) -> List[torch.Tensor]:
    """
    Collect activations for a list of prompts (single outputs per prompt).
    """
    acts = get_layer_activations_batch(model, tokenizer, prompts, layer_idx=layer_idx)
    return [acts[i] for i in range(acts.size(0))]


## =============================================================================
# TRACE / CHAIN-OF-THOUGHT-LIKE METRIC (JSON TRACE LENGTH)
# =============================================================================


# -----------------------------
# Prompting (strong constraints)
# -----------------------------
TRACE_JSON_INSTRUCTION = """
You MUST respond with ONLY valid JSON. No markdown. No extra text.

Return EXACTLY one JSON object with this schema:
{
  "final_answer": string,
  "trace": [
    {"step": int, "text": string}
  ]
}

Rules:
- "trace" must be a JSON array (can be empty).
- Steps must be in order, starting at 1.
- Keep each "text" short.
- Do not include newline characters inside strings (use spaces).
- Output must be strict JSON (double quotes only).
- If you output ANYTHING outside the single JSON object, your answer will be rejected.
""".strip()


def build_trace_prompt(prompt: str) -> str:
    """
    Append a strict JSON-trace instruction to the original prompt.

    Key fixes:
    - Removed "Let's think step by step." (often triggers numbering/extra text)
    - Added an explicit JSON template anchor
    - Added explicit END_JSON marker to help extraction if model includes extra text

    We still ask the model to output ONLY the JSON object.
    """
    return (
        f"{prompt}\n\n"
        f"{TRACE_JSON_INSTRUCTION}\n\n"
        f"BEGIN_JSON\n"
        f'{{"final_answer":"","trace":[]}}\n'
        f"END_JSON\n"
        f"Now REPLACE the JSON object between BEGIN_JSON and END_JSON with your answer.\n"
        f"Output ONLY the JSON object (no BEGIN_JSON/END_JSON markers, no extra text).\n"
    )


# -----------------------------
# Robust JSON extraction helpers
# -----------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_BEGIN_RE = re.compile(r"\bBEGIN_JSON\b", re.IGNORECASE)
_END_RE = re.compile(r"\bEND_JSON\b", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    """If the model wrapped JSON in ```json ... ```, extract the inner content."""
    if text is None:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_between_markers(text: str) -> str:
    """
    If BEGIN_JSON/END_JSON exist, extract between them.
    If only BEGIN_JSON exists, take everything after it.
    Otherwise return original.
    """
    t = text or ""
    m_begin = _BEGIN_RE.search(t)
    if not m_begin:
        return t.strip()

    start = m_begin.end()
    m_end = _END_RE.search(t, pos=start)
    if m_end:
        return t[start:m_end.start()].strip()
    return t[start:].strip()


def _find_first_json_object_span(text: str) -> Optional[Tuple[int, int]]:
    """
    Find the span (start,end) of the first top-level JSON object in text,
    using brace matching that respects strings and escapes.
    """
    s = text or ""
    n = len(s)

    i = s.find("{")
    if i == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    start = None

    for j in range(i, n):
        ch = s[j]

        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = j
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return (start, j + 1)

    return None


def _light_json_repairs(candidate: str) -> str:
    """
    Conservative repairs for common JSON-ish mistakes:
    - Remove trailing commas before } or ]
    - Replace smart quotes with normal quotes
    - If it used single quotes consistently, attempt cautious conversion
      (only if there are no double quotes at all).
    """
    c = (candidate or "").strip()

    # Normalize smart quotes
    c = c.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # Remove trailing commas
    c = re.sub(r",\s*([}\]])", r"\1", c)

    # Cautious single-quote conversion if no double quotes exist
    if '"' not in c and "'" in c:
        c = re.sub(
            r"'\s*([^']*?)\s*'",
            lambda m: '"' + m.group(1).replace('"', '\\"') + '"',
            c,
        )

    return c


def _coerce_to_schema(obj: Any) -> Dict[str, Any]:
    """
    Force object into the strict schema:
      {"final_answer": str, "trace": [{"step": int, "text": str}, ...]}

    Guarantees:
    - Always returns dict with keys final_answer, trace
    - trace is always a list of {"step": int, "text": str}
    """
    out: Dict[str, Any] = {"final_answer": "", "trace": []}

    if not isinstance(obj, dict):
        return out

    # final_answer
    fa = obj.get("final_answer", "")
    out["final_answer"] = "" if fa is None else str(fa)

    # trace
    trace = obj.get("trace", [])
    if not isinstance(trace, list):
        out["trace"] = []
        return out

    norm_steps: List[Dict[str, Any]] = []
    for idx, step_obj in enumerate(trace, start=1):
        if isinstance(step_obj, dict):
            st = step_obj.get("step", idx)
            tx = step_obj.get("text", "")
        else:
            st = idx
            tx = step_obj

        # step -> int
        try:
            st_i = int(st)
        except Exception:
            st_i = idx

        # text -> single line string
        tx_s = "" if tx is None else str(tx)
        tx_s = " ".join(tx_s.split())

        norm_steps.append({"step": st_i, "text": tx_s})

    out["trace"] = norm_steps
    return out


def extract_or_make_valid_trace_json(raw_text: str) -> Tuple[Dict[str, Any], bool]:
    """
    ALWAYS returns a valid schema object.
    Returns (obj, parsed_ok)
    """
    if raw_text is None:
        return _coerce_to_schema({}), False

    # 1) strip fences
    t = _strip_code_fences(raw_text)

    # 2) markers
    t2 = _extract_between_markers(t)

    # 3) full parse
    candidate = (t2 or "").strip()
    try:
        obj = json.loads(candidate)
        return _coerce_to_schema(obj), True
    except Exception:
        pass

    # 4) first object span
    span = _find_first_json_object_span(candidate)
    if span is not None:
        sub = candidate[span[0]:span[1]].strip()
        sub = _light_json_repairs(sub)
        try:
            obj = json.loads(sub)
            return _coerce_to_schema(obj), True
        except Exception:
            pass

    # 5) repairs on whole text
    repaired = _light_json_repairs(candidate)
    try:
        obj = json.loads(repaired)
        return _coerce_to_schema(obj), True
    except Exception:
        pass

    # 6) fallback
    return _coerce_to_schema({}), False


def trace_json_to_string(obj: Dict[str, Any]) -> str:
    """Convert schema object to a strict JSON string (always valid)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# -----------------------------
# Metric: trace length
# -----------------------------
def compute_trace_length_from_text(trace_output_text: str) -> Tuple[float, bool, Dict[str, Any]]:
    """
    Parse the JSON trace and return:
      (trace_len, parsed_ok, obj)
    """
    obj, ok = extract_or_make_valid_trace_json(trace_output_text)
    trace = obj.get("trace", [])
    trace_len = float(len(trace)) if isinstance(trace, list) else 0.0
    return trace_len, ok, obj


# -----------------------------
# Second-pass fixer (optional but recommended)
# -----------------------------
def _force_json_second_pass(
        model,
        tokenizer,
        bad_text: str,
        *,
        generate_fn: Callable[..., List[str]],
        max_new_tokens: int = 256,
) -> str:
    """
    Ask the model to convert arbitrary text into EXACT schema JSON.
    This is the practical way to get JSON for every prompt.
    """
    fixer_prompt = (
        "Convert the text below into EXACTLY ONE valid JSON object with this schema:\n"
        "{\n"
        '  "final_answer": string,\n'
        '  "trace": [ {"step": int, "text": string} ]\n'
        "}\n"
        "Rules:\n"
        "- Output ONLY JSON. No markdown, no commentary.\n"
        "- trace must be an array of objects with step/text.\n"
        '- Remove any extra text like numbering, bullets, or explanations.\n'
        "- Do not include newline characters inside strings (use spaces).\n\n"
        "TEXT_TO_CONVERT:\n"
        f"{bad_text}\n"
    )
    out = generate_fn(
        model,
        tokenizer,
        prompts=[fixer_prompt],
        max_new_tokens=max_new_tokens,
        batch_size=1,
        do_sample=False,
        use_cache=False,
    )
    return out[0] if out else ""


# -----------------------------
# Batched experiment
# -----------------------------
def compute_trace_metric_batched(
        model,
        tokenizer,
        prompts: List[str],
        *,
        max_new_tokens: int = 384,
        batch_size: int = 8,
        do_sample: bool = False,
        use_cache: bool = False,
        generate_fn: Optional[Callable[..., List[str]]] = None,
        fix_bad_outputs: bool = True,
        fix_max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """
    Batched trace generation + trace-length extraction.

    Returns dict with:
      - trace_lengths: List[float] (always finite; 0.0 if fallback)
      - parsed_flags: List[bool] (True if parsed from model output or fixed output)
      - parsed_rate: float in [0,1]
      - raw_outputs: List[str] (optionally replaced by fixed outputs if parse failed)
      - json_outputs: List[str]  (ALWAYS valid JSON strings matching schema)
    """
    if generate_fn is None:
        raise ValueError("compute_trace_metric_batched requires generate_fn (your batched generate_response).")

    if not prompts:
        return {
            "trace_lengths": [],
            "parsed_flags": [],
            "parsed_rate": float("nan"),
            "raw_outputs": [],
            "json_outputs": [],
        }

    trace_prompts = [build_trace_prompt(p) for p in prompts]

    raw_outputs = generate_fn(
        model,
        tokenizer,
        prompts=trace_prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        do_sample=do_sample,
        use_cache=use_cache,
    )

    trace_lengths: List[float] = []
    parsed_flags: List[bool] = []
    json_outputs: List[str] = []
    final_raw_outputs: List[str] = []

    for out_text in raw_outputs:
        # 1) try parse
        obj, ok = extract_or_make_valid_trace_json(out_text)

        # 2) if failed, second-pass fix -> parse again
        if (not ok) and fix_bad_outputs:
            fixed_text = _force_json_second_pass(
                model,
                tokenizer,
                out_text,
                generate_fn=generate_fn,
                max_new_tokens=fix_max_new_tokens,
            )
            obj2, ok2 = extract_or_make_valid_trace_json(fixed_text)
            if ok2:
                obj, ok = obj2, ok2
                out_text = fixed_text  # replace raw output with fixed output

        trace = obj.get("trace", [])
        trace_len = float(len(trace)) if isinstance(trace, list) else 0.0

        trace_lengths.append(trace_len)
        parsed_flags.append(bool(ok))
        json_outputs.append(trace_json_to_string(obj))  # ALWAYS strict JSON
        final_raw_outputs.append(out_text)

    parsed_rate = float(np.mean(parsed_flags)) if len(parsed_flags) else float("nan")

    return {
        "trace_lengths": trace_lengths,
        "parsed_flags": parsed_flags,
        "parsed_rate": parsed_rate,
        "raw_outputs": final_raw_outputs,
        "json_outputs": json_outputs,
    }




# =============================================================================
# Data structures
# =============================================================================

@dataclass
class MirroringJudgeResult:
    mirrored: bool
    raw_text: str
    model: str
    meta: Dict[str, Any]


# =============================================================================
# Output cleaning (remove echoed prompt / transcript)
# =============================================================================

_TRANSCRIPT_PATTERNS = [
    r"^\s*Date:\s.*$",
    r"^\s*user\s*$",
    r"^\s*assistant\s*$",
    r"^\s*system\s*$",
]

def _strip_transcript_lines(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    kept = []
    for ln in lines:
        drop = False
        for pat in _TRANSCRIPT_PATTERNS:
            if re.match(pat, ln.strip(), flags=re.IGNORECASE):
                drop = True
                break
        if not drop:
            kept.append(ln)
    return "\n".join(kept).strip()


def clean_chatty_generation(output_text: str, *, prompt_text: Optional[str] = None) -> str:
    if not output_text:
        return ""

    t = output_text.strip()
    t = _strip_transcript_lines(t)

    t = re.sub(r"(?is)\buser\b\s*", "", t, count=1).strip()
    t = re.sub(r"(?is)\bassistant\b\s*", "", t, count=1).strip()

    if prompt_text:
        p = prompt_text.strip()
        if p:
            idx = t.lower().find(p.lower())
            if 0 <= idx <= 250:
                cut = idx + len(p)
                t = t[cut:].lstrip(" \n\r\t:-—")

    return t.strip()


# =============================================================================
# YES/NO parsing
# =============================================================================

def _extract_yes_no(text: str) -> Optional[bool]:
    if not text:
        return None
    m = re.search(r"\b(YES|NO)\b", text.strip().upper())
    if not m:
        return None
    return m.group(1) == "YES"


# =============================================================================
# STYLE-SPECIFIC PROMPT BUILDERS
# =============================================================================

def build_mirroring_prompt_politeness(
        *,
        original_prompt: str,
        original_output: str,
        styled_prompt: str,
        styled_output: str,
        strength: Any,
        place: Optional[str] = None,
) -> str:
    """
    Judge whether STYLED OUTPUT is more polite (or less polite,
    depending on direction) than ORIGINAL OUTPUT.

    Focus ONLY on politeness markers and tone.
    """

    place_str = place if place is not None else "N/A"

    return f"""
You are evaluating POLITENESS STYLE MIRRORING.

Task:
Decide whether the STYLED OUTPUT reflects a stronger politeness shift
consistent with the STYLED PROMPT than the ORIGINAL OUTPUT does.

Focus ONLY on politeness-related signals:
- Respectfulness
- Softening/hedging (e.g., "could", "would", "please")
- Gratitude/apology markers
- Warmth or courteous phrasing
- Overall tone (more/less polite)

Ignore:
- Content correctness
- Topic differences
- Length differences
- Factual quality

Answer YES if the STYLED OUTPUT clearly shows a politeness shift
relative to the ORIGINAL OUTPUT.

If unclear or ambiguous, answer NO.

Style metadata:
- strength: {strength}
- place: {place_str}

=== ORIGINAL PROMPT ===
{original_prompt}

=== ORIGINAL OUTPUT ===
{original_output}

=== STYLED PROMPT ===
{styled_prompt}

=== STYLED OUTPUT ===
{styled_output}

Return EXACTLY one token: YES or NO.
""".strip()


def build_mirroring_prompt_surface_noise(
        *,
        original_prompt: str,
        original_output: str,
        styled_prompt: str,
        styled_output: str,
        strength: Any,
        place: Optional[str] = None,
) -> str:
    # TODO: write appropriate prompt for surface_noise style
    pass


def build_mirroring_prompt_structured_rewriting(
        *,
        original_prompt: str,
        original_output: str,
        styled_prompt: str,
        styled_output: str,
        strength: Any,
        place: Optional[str] = None,
) -> str:
    # TODO: write appropriate prompt for structured_rewriting style
    pass


# =============================================================================
# Prompt selector
# =============================================================================

def build_mirroring_prompt_by_style(
        *,
        original_prompt: str,
        original_output: str,
        styled_prompt: str,
        styled_output: str,
        style_name: str,
        strength: Any,
        place: Optional[str] = None,
) -> str:

    style_name = style_name.lower()

    if style_name == "politeness":
        return build_mirroring_prompt_politeness(
            original_prompt=original_prompt,
            original_output=original_output,
            styled_prompt=styled_prompt,
            styled_output=styled_output,
            strength=strength,
            place=place,
        )

    elif style_name == "surface_noise":
        return build_mirroring_prompt_surface_noise(
            original_prompt=original_prompt,
            original_output=original_output,
            styled_prompt=styled_prompt,
            styled_output=styled_output,
            strength=strength,
            place=place,
        )

    elif style_name == "structured_rewriting":
        return build_mirroring_prompt_structured_rewriting(
            original_prompt=original_prompt,
            original_output=original_output,
            styled_prompt=styled_prompt,
            styled_output=styled_output,
            strength=strength,
            place=place,
        )

    else:
        raise ValueError(f"Unknown style_name: {style_name}")


# =============================================================================
# OpenAI judge
# =============================================================================

def judge_style_mirroring_openai(
        *,
        original_prompt: str,
        original_output: str,
        styled_prompt: str,
        styled_output: str,
        style_name: str,
        strength: Any,
        place: Optional[str] = None,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_output_tokens: int = 16,
) -> MirroringJudgeResult:

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {api_key_env}")

    judge_prompt = build_mirroring_prompt_by_style(
        original_prompt=original_prompt,
        original_output=original_output,
        styled_prompt=styled_prompt,
        styled_output=styled_output,
        style_name=style_name,
        strength=strength,
        place=place,
    )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [{"type": "input_text", "text": judge_prompt}],
        }],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    text = (getattr(resp, "output_text", "") or "").strip()
    yn = _extract_yes_no(text)
    if yn is None:
        raise ValueError(f"Judge output not parseable as YES/NO. Raw: {text!r}")

    return MirroringJudgeResult(
        mirrored=yn,
        raw_text=text,
        model=model,
        meta={}
    )


# =============================================================================
# Gemini judge
# =============================================================================

def judge_style_mirroring_gemini(
        *,
        original_prompt: str,
        original_output: str,
        styled_prompt: str,
        styled_output: str,
        style_name: str,
        strength: Any,
        place: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        temperature: float = 0.0,
        max_output_tokens: int = 16,
) -> MirroringJudgeResult:

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {api_key_env}")

    judge_prompt = build_mirroring_prompt_by_style(
        original_prompt=original_prompt,
        original_output=original_output,
        styled_prompt=styled_prompt,
        styled_output=styled_output,
        style_name=style_name,
        strength=strength,
        place=place,
    )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model=model,
        contents=judge_prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )

    text = (resp.text or "").strip()
    yn = _extract_yes_no(text)
    if yn is None:
        raise ValueError(f"Judge output not parseable as YES/NO. Raw: {text!r}")

    return MirroringJudgeResult(
        mirrored=yn,
        raw_text=text,
        model=model,
        meta={}
    )


# =============================================================================
# Retry wrapper
# =============================================================================

def judge_with_retries(
        *,
        judge_provider: str,
        original_prompt: str,
        original_output: str,
        styled_prompt: str,
        styled_output: str,
        style_name: str,
        strength: int,
        place: str,
        judge_model: str,
        max_retries: int = 6,
        base_sleep_s: float = 1.0,
        max_output_tokens: int = 16,
        openai_key_env: str = "OPENAI_API_KEY",
        gemini_key_env: str = "GEMINI_API_KEY",
) -> Tuple[Optional[bool], str, str]:

    judge_prompt = build_mirroring_prompt_by_style(
        original_prompt=original_prompt,
        original_output=original_output,
        styled_prompt=styled_prompt,
        styled_output=styled_output,
        style_name=style_name,
        strength=strength,
        place=place,
    )

    for attempt in range(max_retries):
        try:
            if judge_provider == "openai":
                jr = judge_style_mirroring_openai(
                    original_prompt=original_prompt,
                    original_output=original_output,
                    styled_prompt=styled_prompt,
                    styled_output=styled_output,
                    style_name=style_name,
                    strength=strength,
                    place=place,
                    model=judge_model,
                    api_key_env=openai_key_env,
                    max_output_tokens=max_output_tokens,
                )
            elif judge_provider == "gemini":
                jr = judge_style_mirroring_gemini(
                    original_prompt=original_prompt,
                    original_output=original_output,
                    styled_prompt=styled_prompt,
                    styled_output=styled_output,
                    style_name=style_name,
                    strength=strength,
                    place=place,
                    model=judge_model,
                    api_key_env=gemini_key_env,
                    max_output_tokens=max_output_tokens,
                )
            else:
                raise ValueError(f"Unknown judge_provider: {judge_provider}")

            verdict = jr.mirrored
            return verdict, jr.raw_text, judge_prompt

        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["rate", "429", "quota", "limit"]):
                time.sleep(base_sleep_s * (2 ** attempt))
                continue
            return None, f"ERROR: {e}", judge_prompt

    return None, "ERROR: rate-limited after retries", judge_prompt