"""
Dataset loading utilities.

Functions:
    - load_truthfulqa: Load TruthfulQA dataset
    - load_mmlu: Load MMLU dataset
    - load_dataset: Factory function to load any dataset by name
"""

from datasets import load_dataset

import random
from typing import Any, Dict, List, Optional, Sequence, Union

def load_truthfulqa(sample_size=128, seed=42, config_name="generation", split="validation"):
    """
    Load TruthfulQA dataset.
    
    Args:
        sample_size (int): Number of prompts to sample
        seed (int): Random seed for reproducibility
        config_name (str): Dataset configuration
        split (str): Dataset split to use
    
    Returns:
        list[dict]: List of standardized prompt dictionaries with keys:
            - question: str
            - best_answer: str
            - category: str
            - meta: dict (contains correct_answers, incorrect_answers)
    """
    print(f"Loading TruthfulQA ({config_name}, {split})...")
    dataset = load_dataset("truthful_qa", config_name, split=split)
    
    # Shuffle and sample
    shuffled = dataset.shuffle(seed=seed)
    count = min(sample_size, len(shuffled))
    subset = shuffled.select(range(count))
    
    # Standardize format
    prompts = []
    for item in subset:
        prompts.append({
            "question": item["question"],
            "best_answer": item["best_answer"],
            "category": item.get("category", "Unknown"),
            "meta": {
                "correct_answers": item.get("correct_answers", []),
                "incorrect_answers": item.get("incorrect_answers", [])
            }
        })
    
    print(f"Loaded {len(prompts)} prompts.")
    return prompts


def load_mmlu(sample_size=50, seed=42, config_name="abstract_algebra", split="test"):
    """
    Load MMLU dataset.
    
    Args:
        sample_size (int): Number of prompts to sample
        seed (int): Random seed for reproducibility
        config_name (str): MMLU task (e.g., 'abstract_algebra', 'anatomy')
        split (str): Dataset split to use
    
    Returns:
        list[dict]: List of standardized prompt dictionaries with keys:
            - question: str
            - best_answer: str (the correct choice)
            - category: str (the MMLU task name)
            - meta: dict (contains choices, answer_index)
    """
    print(f"Loading MMLU ({config_name}, {split})...")
    dataset = load_dataset("cais/mmlu", config_name, split=split)
    
    # Shuffle and sample
    shuffled = dataset.shuffle(seed=seed)
    count = min(sample_size, len(shuffled))
    subset = shuffled.select(range(count))
    
    # Standardize format
    prompts = []
    for item in subset:
        # MMLU format: question, choices (list), answer (int index)
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
                "answer_index": answer_idx
            }
        })
    
    print(f"Loaded {len(prompts)} prompts.")
    return prompts


def load_dataset_by_name(dataset_name, **kwargs):
    """
    Load dataset by name (factory function).
    
    Args:
        dataset_name (str): 'truthful_qa' or 'mmlu'
        **kwargs: Additional arguments passed to specific loader
    
    Returns:
        list[dict]: Standardized prompts
    """
    loaders = {
        "truthful_qa": load_truthfulqa,
        "mmlu": load_mmlu,
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
    """
    Load BBQ from Hugging Face in the same standardized format as your other loaders.

    Args:
        sample_size: total number of prompts to return (across selected configs)
        seed: random seed
        repo_id: HF dataset id (default: "HiTZ/bbq")
        category:
            - None: load ALL configs (be careful, can be big)
            - str: can be an exact config like "Age_ambig"
                   OR a prefix like "Age" (loads all configs starting with "Age_")
            - list[str]: mix of exact configs and/or prefixes
        split:
            - If provided, use this split.
            - If None, picks "train" if present, else the first available split.

    Returns:
        list[dict] with keys:
            - question: formatted MCQ prompt
            - best_answer: gold answer text when label is present
            - category: config name (e.g., "Age_ambig")
            - meta: contains raw fields used for bias scoring (label, ans*, target_*, etc.)
    """

    def _pick_split(ds_dict) -> str:
        if split is not None:
            return split
        # Prefer "train" if it exists; otherwise choose first key
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

        # Stable, easy-to-parse MCQ
        parts = [
            q,
            f"(a) {a0} (b) {a1} (c) {a2}",
        ]
        if c:
            parts.append(c)
        return "\n".join(parts).strip()

    # 1) Decide which configs to load

    requested: List[str] = []
    if category is None:
        # Load "ALL configs": Hugging Face requires specifying one config at a time,
        # so we need the config list.
        from datasets import get_dataset_config_names

        requested = list(get_dataset_config_names(repo_id))
        if not requested:
            raise ValueError(f"No configs found for {repo_id}.")
    else:
        if isinstance(category, str):
            cats = [category]
        else:
            cats = list(category)

        # Expand prefixes like "Age" -> all configs starting with "Age_"
        from datasets import get_dataset_config_names

        all_configs = list(get_dataset_config_names(repo_id))
        for c in cats:
            c = c.strip()
            if c in all_configs:
                requested.append(c)
            else:
                # treat as prefix
                pref = c + "_"
                matches = [x for x in all_configs if x.startswith(pref)]
                if not matches:
                    raise ValueError(
                        f"BBQ category/config '{c}' not found. "
                        f"Try an exact config like 'Age_ambig' or a valid prefix like 'Age'."
                    )
                requested.extend(matches)

        # de-dup while preserving order
        seen = set()
        requested = [x for x in requested if not (x in seen or seen.add(x))]

    # 2) Load each config, shuffle deterministically, collect rows
    rng = random.Random(seed)
    all_rows: List[Dict[str, Any]] = []

    for cfg in requested:
        ds_dict = load_dataset(repo_id, cfg)
        use_split = _pick_split(ds_dict)
        ds = ds_dict[use_split].shuffle(seed=seed)

        # Pull everything (then global sample), OR cap per-config (simple: pull all)
        # We'll collect all rows then do a global sample at the end.
        for item in ds:
            ex = dict(item)
            ex["_bbq_config"] = cfg
            all_rows.append(ex)

    if not all_rows:
        raise RuntimeError(f"Loaded 0 rows from {repo_id} with configs={requested}")

    # 3) Global deterministic sample across all selected configs
    rng.shuffle(all_rows)
    count = min(sample_size, len(all_rows))
    subset = all_rows[:count]

    # 4) Standardize
    prompts: List[Dict[str, Any]] = []
    for ex in subset:
        label = ex.get("label", None)
        ans_map = {0: ex.get("ans0"), 1: ex.get("ans1"), 2: ex.get("ans2")}
        best_answer = ans_map.get(label, None) if isinstance(label, int) else None

        prompts.append(
            {
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
            }
        )

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
        config_name: str = "standard",   # one of: standard / contextual / copyright
        split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load HarmBench from Hugging Face in your standardized format.

    Notes:
      - HF often uses "config_name" to select subsets (as in your screenshot).
      - Some datasets expose only one split; we choose "train" if present, else first.

    Returns:
        list[dict] with keys:
            - question: str (the harmful query/request)
            - best_answer: None
            - category: str (harmbench/<config_name>)
            - meta: raw fields
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

    print(f"Loading HarmBench from HF: {repo_id} config='{config_name}' ...")
    ds_dict = load_dataset(repo_id, config_name)
    use_split = _pick_split(ds_dict)
    ds = ds_dict[use_split].shuffle(seed=seed)

    count = min(sample_size, len(ds))
    ds = ds.select(range(count))

    prompts: List[Dict[str, Any]] = []
    for item in ds:
        ex = dict(item)

        # Common field names across many HarmBench variants:
        # try several candidates safely
        q = (
                ex.get("query")
                or ex.get("prompt")
                or ex.get("instruction")
                or ex.get("text")
                or ""
        )

        prompts.append(
            {
                "question": str(q),
                "best_answer": None,
                "category": f"harmbench/{config_name}",
                "meta": {"raw": ex},
            }
        )

    print(f"Loaded {len(prompts)} HarmBench prompts (config='{config_name}', split='{use_split}').")
    return prompts


