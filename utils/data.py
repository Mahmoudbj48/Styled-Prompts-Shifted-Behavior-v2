"""
Dataset loading utilities.

All loaders return a list of dicts with a standardised schema:
    {
        "question":    str,   # the prompt / input text
        "best_answer": str,   # reference answer (None if unavailable)
        "category":    str,   # dataset name or sub-category
        "meta":        dict,  # dataset-specific raw fields
    }

Functions:
    - load_truthfulqa:      Load TruthfulQA (generation or multiple-choice config)
    - load_mmlu:            Load MMLU (single subject)
    - load_alpaca_hf:       Load Alpaca (HF: tatsu-lab/alpaca)
    - load_gsm8k:           Load GSM8K math problems (HF: openai/gsm8k)
    - load_harmbench_hf:    Load HarmBench harmful prompts (HF: walledai/HarmBench)
    - load_natural_questions: Load Natural Questions (HF: google-research-datasets/natural_questions)
    - load_dataset_by_name: Factory function to load any dataset by name

Assumptions:
    - A valid HuggingFace token (HF_TOKEN) is set in the environment for gated repos.
    - Sample sizes and seeds are respected via dataset shuffling before selection.
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
# GSM8K (Hugging Face: "openai/gsm8k")
# -----------------------------------------------------------------------------
def load_gsm8k(
        sample_size: int = 128,
        seed: int = 42,
        *,
        repo_id: str = "openai/gsm8k",
        config_name: str = "main",
        split: str = "test",
) -> List[Dict[str, Any]]:
    """
    Load GSM8K (Grade School Math 8K) dataset for reasoning evaluation.

    GSM8K contains grade school math word problems that require multi-step reasoning.
    Each problem has a question and a solution with step-by-step reasoning.

    Args:
        sample_size: Number of problems to load
        seed: Random seed for shuffling
        repo_id: HuggingFace repo (default: "openai/gsm8k")
        config_name: Dataset config (default: "main")
        split: Dataset split ("train" or "test", default: "test")

    Returns:
        list[dict] with keys:
            - question: str (the math word problem)
            - best_answer: str (final numerical answer)
            - category: str ("gsm8k")
            - meta: dict containing:
                - solution: str (full step-by-step solution from dataset)
                - raw: dict (original item)
    """
    print(f"Loading GSM8K ({config_name}, {split})...")
    dataset = load_dataset(repo_id, config_name, split=split)

    shuffled = dataset.shuffle(seed=seed)
    count = min(sample_size, len(shuffled))
    subset = shuffled.select(range(count))

    prompts = []
    for item in subset:
        question = item["question"].strip()
        answer_with_solution = item["answer"].strip()
        
        # GSM8K answers are formatted as:
        # "Step 1 explanation\nStep 2 explanation\n#### 42"
        # Extract the final answer after "####"
        final_answer = None
        if "####" in answer_with_solution:
            final_answer = answer_with_solution.split("####")[-1].strip()
        
        prompts.append({
            "question": question,
            "best_answer": final_answer,
            "category": "gsm8k",
            "meta": {
                "solution": answer_with_solution,  # Full solution with steps
                "raw": dict(item),
            }
        })

    print(f"Loaded {len(prompts)} GSM8K problems.")
    return prompts


# -----------------------------------------------------------------------------
# SimpleQA Verified (Hugging Face: "google/simpleqa-verified")
# -----------------------------------------------------------------------------
def load_simpleqa_verified(
        sample_size: int = 128,
        seed: int = 42,
        *,
        repo_id: str = "google/simpleqa-verified",
        split: str = "test",
) -> List[Dict[str, Any]]:
    """
    Load SimpleQA Verified dataset.

    Returns:
        list[dict] with keys:
            - question: str
            - best_answer: str
            - category: str ("simpleqa_verified")
            - meta: dict
    """
    print(f"Loading SimpleQA Verified ({split})...")
    dataset = load_dataset(repo_id, split=split)

    shuffled = dataset.shuffle(seed=seed)
    count = min(sample_size, len(shuffled))
    subset = shuffled.select(range(count))

    prompts = []
    for item in subset:
        ex = dict(item)
        q = ex.get("problem") or ex.get("question") or ex.get("query") or ""
        a = ex.get("answer") or None

        prompts.append({
            "question": str(q).strip(),
            "best_answer": str(a).strip() if a is not None else None,
            "category": "simpleqa_verified",
            "meta": {"raw": ex},
        })

    print(f"Loaded {len(prompts)} SimpleQA Verified prompts.")
    return prompts


# -----------------------------------------------------------------------------
# TriviaQA (Hugging Face: "mandarjoshi/trivia_qa")
# -----------------------------------------------------------------------------
def load_trivia_qa(
        sample_size: int = 128,
        seed: int = 42,
        *,
        repo_id: str = "mandarjoshi/trivia_qa",
        config_name: str = "rc",
        split: str = "validation",
) -> List[Dict[str, Any]]:
    """
    Load TriviaQA dataset.

    Returns:
        list[dict] with keys:
            - question: str
            - best_answer: str
            - category: str ("trivia_qa")
            - meta: dict
    """
    print(f"Loading TriviaQA ({config_name}, {split})...")
    dataset = load_dataset(repo_id, config_name, split=split)

    shuffled = dataset.shuffle(seed=seed)
    count = min(sample_size, len(shuffled))
    subset = shuffled.select(range(count))

    prompts = []
    for item in subset:
        ex = dict(item)
        q = ex.get("question", "")
        answer = ex.get("answer", {})
        a = answer.get("value") if isinstance(answer, dict) else str(answer)

        prompts.append({
            "question": str(q).strip(),
            "best_answer": str(a).strip() if a else None,
            "category": "trivia_qa",
            "meta": {"raw": ex},
        })

    print(f"Loaded {len(prompts)} TriviaQA prompts.")
    return prompts


# -----------------------------------------------------------------------------
# HotpotQA (Hugging Face: "hotpotqa/hotpot_qa")
# -----------------------------------------------------------------------------
def load_hotpot_qa(
        sample_size: int = 128,
        seed: int = 42,
        *,
        repo_id: str = "hotpotqa/hotpot_qa",
        config_name: str = "distractor",
        split: str = "validation",
) -> List[Dict[str, Any]]:
    """
    Load HotpotQA dataset.

    Returns:
        list[dict] with keys:
            - question: str
            - best_answer: str
            - category: str ("hotpot_qa")
            - meta: dict
    """
    print(f"Loading HotpotQA ({config_name}, {split})...")
    dataset = load_dataset(repo_id, config_name, split=split)

    shuffled = dataset.shuffle(seed=seed)
    count = min(sample_size, len(shuffled))
    subset = shuffled.select(range(count))

    prompts = []
    for item in subset:
        ex = dict(item)
        prompts.append({
            "question": str(ex.get("question", "")).strip(),
            "best_answer": str(ex.get("answer", "")).strip() or None,
            "category": "hotpot_qa",
            "meta": {"raw": ex},
        })

    print(f"Loaded {len(prompts)} HotpotQA prompts.")
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
        "harmbench": load_harmbench_hf,
        "gsm8k": load_gsm8k,
        "natural_questions": load_natural_questions,
        "simpleqa_verified": load_simpleqa_verified,
        "trivia_qa": load_trivia_qa,
        "hotpot_qa": load_hotpot_qa,
    }

    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")

    return loaders[dataset_name](**kwargs)


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
    """
    Load HarmBench harmful prompts from HuggingFace.

    HarmBench is a safety evaluation benchmark containing harmful requests
    across multiple categories (standard, contextual, copyright, etc.).

    Args:
        sample_size: Number of prompts to load.
        seed: Random seed for shuffling.
        repo_id: HuggingFace repo (default: "walledai/HarmBench").
        config_name: Dataset config (default: "standard").
            Other options vary by repo version (e.g. "contextual", "copyright").
        split: Dataset split. If None, selects the first available split.

    Returns:
        list[dict] with standardised keys:
            - question:    str (the harmful prompt text)
            - best_answer: None (HarmBench has no reference answers)
            - category:    str ("harmbench/{config_name}")
            - meta:        dict with the raw example fields
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



# -----------------------------------------------------------------------------
# Natural Questions
# -----------------------------------------------------------------------------
def load_natural_questions(
        sample_size: int = 256,
        seed: int = 42,
        *,
        repo_id: str = "google-research-datasets/natural_questions",
        config_name: str = "default",
        split: str = "validation",
) -> List[Dict[str, Any]]:
    """
    Load Natural Questions dataset.

    Natural Questions contains real user queries paired with
    annotated short and long answers.

    Returns:
        list[dict] with keys:
            - question: str
            - best_answer: str (short answer if available)
            - category: str ("natural_questions")
            - meta: dict (raw fields + long answer info)
    """

    print(f"Loading Natural Questions ({split})...")
    dataset = load_dataset(repo_id, split=split)

    dataset = dataset.shuffle(seed=seed)
    count = min(sample_size, len(dataset))
    dataset = dataset.select(range(count))

    prompts = []

    for ex in dataset:
        # --- FIXED QUESTION EXTRACTION ---
        q = ex.get("question", "")

        if isinstance(q, dict):
            q = q.get("text", "")

        question = str(q).strip()


        best_answer = None
        # annotations = ex.get("annotations", [])
        # if annotations:
        #     short_answers = annotations[0].get("short_answers", [])
        #     if short_answers:
        #         best_answer = short_answers[0].get("text", None)

        prompts.append({
            "question": question,
            "best_answer": best_answer,
            "category": "natural_questions",
            "meta": {"raw": ex},
        })

    print(f"Loaded {len(prompts)} Natural Questions prompts.")
    return prompts