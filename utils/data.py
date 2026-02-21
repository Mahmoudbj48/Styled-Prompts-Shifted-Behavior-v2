"""
Dataset loading utilities.

Functions:
    - load_truthfulqa: Load TruthfulQA dataset
    - load_mmlu: Load MMLU dataset
    - load_alpaca_hf: Load Alpaca (HF: tatsu-lab/alpaca)
    - load_bbq_hf: Load BBQ (HF: HiTZ/bbq)
    - load_harmbench_hf: Load HarmBench (HF: walledai/HarmBench by default)
    - load_dataset_by_name: Factory function to load any dataset by name
"""

from datasets import load_dataset

import random
from typing import Any, Dict, List, Optional, Sequence, Union


# -----------------------------------------------------------------------------
# TruthfulQA
# -----------------------------------------------------------------------------
def load_truthfulqa(sample_size=128, seed=42, config_name="generation", split="validation"):
    """
    Load TruthfulQA dataset.

    Returns:
        list[dict]: keys:
            - question: str
            - best_answer: str
            - category: str
            - meta: dict
    """
    print(f"Loading TruthfulQA ({config_name}, {split})...")
    dataset = load_dataset("truthful_qa", config_name, split=split)

    shuffled = dataset.shuffle(seed=seed)
    count = min(sample_size, len(shuffled))
    subset = shuffled.select(range(count))

    prompts = []
    for item in subset:
        prompts.append({
            "question": item["question"],
            "best_answer": item["best_answer"],
            "category": item.get("category", "Unknown"),
            "meta": {
                "correct_answers": item.get("correct_answers", []),
                "incorrect_answers": item.get("incorrect_answers", []),
                "raw": dict(item),
            }
        })

    print(f"Loaded {len(prompts)} prompts.")
    return prompts


# -----------------------------------------------------------------------------
# MMLU
# -----------------------------------------------------------------------------
def load_mmlu(sample_size=50, seed=42, config_name="abstract_algebra", split="test"):
    """
    Load MMLU dataset.

    Returns:
        list[dict]: keys:
            - question: str
            - best_answer: str
            - category: str
            - meta: dict
    """
    print(f"Loading MMLU ({config_name}, {split})...")
    dataset = load_dataset("cais/mmlu", config_name, split=split)

    shuffled = dataset.shuffle(seed=seed)
    count = min(sample_size, len(shuffled))
    subset = shuffled.select(range(count))

    prompts = []
    for item in subset:
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        best_answer = choices[answer_idx]

        prompts.append({
            "question": question,
            "best_answer": best_answer,
            "category": config_name,
            "meta": {
                "choices": choices,
                "answer_index": answer_idx,
                "raw": dict(item),
            }
        })

    print(f"Loaded {len(prompts)} prompts.")
    return prompts


# -----------------------------------------------------------------------------
# Alpaca (Hugging Face: "tatsu-lab/alpaca")
# -----------------------------------------------------------------------------
def load_alpaca_hf(
        sample_size: int = 128,
        seed: int = 42,
        *,
        repo_id: str = "tatsu-lab/alpaca",
        split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load Alpaca prompts from Hugging Face in the SAME standardized format.

    Alpaca fields commonly include:
      - instruction (str)
      - input (str)   (can be empty)
      - output (str)

    Returns:
        list[dict] with keys:
            - question: str (instruction + optional input)
            - best_answer: str (output, if present)
            - category: str ("alpaca")
            - meta: dict (raw fields)
    """

    def _pick_split(ds_dict) -> str:
        if split is not None:
            return split
        keys = list(ds_dict.keys())
        if "train" in keys:
            return "train"
        if not keys:
            raise ValueError("Dataset has no splits.")
        return keys[0]

    def _format_alpaca_prompt(ex: Dict[str, Any]) -> str:
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()

        if inp:
            # Stable, explicit formatting (easy to parse / consistent)
            return f"Instruction:\n{instr}\n\nInput:\n{inp}".strip()
        return f"Instruction:\n{instr}".strip()

    print(f"Loading Alpaca from HF: {repo_id} ...")
    ds_dict = load_dataset(repo_id)
    use_split = _pick_split(ds_dict)

    ds = ds_dict[use_split].shuffle(seed=seed)
    count = min(sample_size, len(ds))
    ds = ds.select(range(count))

    prompts: List[Dict[str, Any]] = []
    for item in ds:
        ex = dict(item)
        q = _format_alpaca_prompt(ex)
        a = (ex.get("output") or None)

        prompts.append({
            "question": str(q),
            "best_answer": str(a) if a is not None else None,
            "category": "alpaca",
            "meta": {"raw": ex},
        })

    print(f"Loaded {len(prompts)} Alpaca prompts (split='{use_split}').")
    return prompts


# -----------------------------------------------------------------------------
# Dataset factory
# -----------------------------------------------------------------------------
def load_dataset_by_name(dataset_name, **kwargs):
    """
    Load dataset by name (factory function).

    Args:
        dataset_name (str):
          - 'truthful_qa'
          - 'mmlu'
          - 'alpaca'
          - 'bbq'
          - 'harmbench'

    Returns:
        list[dict]: Standardized prompts
    """
    loaders = {
        "truthful_qa": load_truthfulqa,
        "mmlu": load_mmlu,
        "alpaca": load_alpaca_hf,
        "bbq": load_bbq_hf,
        "harmbench": load_harmbench_hf,
    }

    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")

    return loaders[dataset_name](**kwargs)


# -----------------------------------------------------------------------------
# BBQ (Hugging Face: "HiTZ/bbq")
# -----------------------------------------------------------------------------
def load_bbq_hf(
        sample_size: int = 200,
        seed: int = 42,
        *,
        repo_id: str = "HiTZ/bbq",
        category: Optional[Union[str, Sequence[str]]] = None,
        split: Optional[str] = None,
) -> List[Dict[str, Any]]:

    def _pick_split(ds_dict) -> str:
        if split is not None:
            return split
        keys = list(ds_dict.keys())
        if "train" in keys:
            return "train"
        if not keys:
            raise ValueError("Dataset has no splits.")
        return keys[0]

    def _format_bbq_prompt(ex: Dict[str, Any]) -> str:
        q = (ex.get("question") or "").strip()
        c = (ex.get("context") or "").strip()
        a0 = (ex.get("ans0") or "").strip()
        a1 = (ex.get("ans1") or "").strip()
        a2 = (ex.get("ans2") or "").strip()

        parts = [
            q,
            f"(a) {a0} (b) {a1} (c) {a2}",
        ]
        if c:
            parts.append(c)
        return "\n".join(parts).strip()

    requested: List[str] = []
    if category is None:
        from datasets import get_dataset_config_names
        requested = list(get_dataset_config_names(repo_id))
        if not requested:
            raise ValueError(f"No configs found for {repo_id}.")
    else:
        if isinstance(category, str):
            cats = [category]
        else:
            cats = list(category)

        from datasets import get_dataset_config_names
        all_configs = list(get_dataset_config_names(repo_id))

        for c in cats:
            c = c.strip()
            if c in all_configs:
                requested.append(c)
            else:
                pref = c + "_"
                matches = [x for x in all_configs if x.startswith(pref)]
                if not matches:
                    raise ValueError(
                        f"BBQ category/config '{c}' not found. "
                        f"Try an exact config like 'Age_ambig' or a valid prefix like 'Age'."
                    )
                requested.extend(matches)

        seen = set()
        requested = [x for x in requested if not (x in seen or seen.add(x))]

    rng = random.Random(seed)
    all_rows: List[Dict[str, Any]] = []

    for cfg in requested:
        ds_dict = load_dataset(repo_id, cfg)
        use_split = _pick_split(ds_dict)
        ds = ds_dict[use_split].shuffle(seed=seed)
        for item in ds:
            ex = dict(item)
            ex["_bbq_config"] = cfg
            all_rows.append(ex)

    if not all_rows:
        raise RuntimeError(f"Loaded 0 rows from {repo_id} with configs={requested}")

    rng.shuffle(all_rows)
    count = min(sample_size, len(all_rows))
    subset = all_rows[:count]

    prompts: List[Dict[str, Any]] = []
    for ex in subset:
        label = ex.get("label", None)
        ans_map = {0: ex.get("ans0"), 1: ex.get("ans1"), 2: ex.get("ans2")}
        best_answer = ans_map.get(label, None) if isinstance(label, int) else None

        prompts.append({
            "question": _format_bbq_prompt(ex),
            "best_answer": best_answer,
            "category": ex.get("_bbq_config", "BBQ"),
            "meta": {
                "raw": ex,
                "label": label,
                "ans0": ex.get("ans0"),
                "ans1": ex.get("ans1"),
                "ans2": ex.get("ans2"),
                "context": ex.get("context"),
                "question": ex.get("question"),
                "target_loc": ex.get("target_loc"),
                "target_label": ex.get("target_label"),
                "category": ex.get("category"),
                "example_id": ex.get("example_id"),
                "_bbq_config": ex.get("_bbq_config"),
            },
        })

    print(f"Loaded {len(prompts)} BBQ prompts from HF ({repo_id}) using configs: {requested}")
    return prompts


# -----------------------------------------------------------------------------
# HarmBench (Hugging Face: "walledai/HarmBench")
# -----------------------------------------------------------------------------
def load_harmbench_hf(
        sample_size: int = 200,
        seed: int = 42,
        *,
        repo_id: str = "walledai/HarmBench",
        config_name: str = "standard",   # e.g. standard / contextual / copyright (depends on repo)
        split: Optional[str] = None,
) -> List[Dict[str, Any]]:

    def _pick_split(ds_dict) -> str:
        if split is not None:
            return split
        keys = list(ds_dict.keys())
        if "train" in keys:
            return "train"
        if not keys:
            raise ValueError("Dataset has no splits.")
        return keys[0]

    print(f"Loading HarmBench from HF: {repo_id} config='{config_name}' ...")
    ds_dict = load_dataset(repo_id, config_name)
    use_split = _pick_split(ds_dict)

    ds = ds_dict[use_split].shuffle(seed=seed)
    count = min(sample_size, len(ds))
    ds = ds.select(range(count))

    prompts: List[Dict[str, Any]] = []
    for item in ds:
        ex = dict(item)
        q = (
                ex.get("query")
                or ex.get("prompt")
                or ex.get("instruction")
                or ex.get("text")
                or ""
        )

        prompts.append({
            "question": str(q),
            "best_answer": None,
            "category": f"harmbench/{config_name}",
            "meta": {"raw": ex},
        })

    print(f"Loaded {len(prompts)} HarmBench prompts (config='{config_name}', split='{use_split}').")
    return prompts