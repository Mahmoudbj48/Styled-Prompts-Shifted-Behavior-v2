"""
Dataset loading utilities.

Functions:
    - load_truthfulqa: Load TruthfulQA dataset
    - load_mmlu: Load MMLU dataset
    - load_dataset: Factory function to load any dataset by name
"""

from datasets import load_dataset


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
        "mmlu": load_mmlu
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    return loaders[dataset_name](**kwargs)