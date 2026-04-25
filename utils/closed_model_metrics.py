"""
utils/closed_model_metrics.py
==============================
Metrics computation for closed-model (API-based) LLM experiments.

Computes:
    - Response similarity (BLEU, BERTScore)
    - Confidence (from logprobs when available)
    - Mirroring (LLM-as-judge or rule-based)

Usage:
    from utils.closed_model_metrics import compute_all_metrics
    
    metrics = compute_all_metrics(
        baseline_response="Paris is the capital.",
        styled_response="Paris is  the  capital.",
        baseline_logprobs=[...],
        styled_logprobs=[...],
        style="spacing",
        strength=50,
        place="global",
    )
"""

import numpy as np
from typing import List, Optional, Dict, Any

from utils.metrics import (
    compute_bleu,
    compute_bertscore,
    judge_with_retries,
)
from utils.surface_mirroring_detector import detect_mirroring


def compute_confidence_from_logprobs(logprobs: Optional[List[float]]) -> Dict[str, float]:
    """
    Compute confidence metrics from token log probabilities.
    
    Args:
        logprobs: List of log probabilities for each generated token
        
    Returns:
        Dict with:
            - mean_confidence: Mean probability across tokens
            - min_confidence: Minimum probability (least confident token)
            - max_confidence: Maximum probability (most confident token)
            - entropy: Shannon entropy of the distribution
    """
    if logprobs is None or len(logprobs) == 0:
        return {
            "mean_confidence": np.nan,
            "min_confidence": np.nan,
            "max_confidence": np.nan,
            "entropy": np.nan,
        }
    
    # Convert log probabilities to probabilities
    probs = [np.exp(lp) for lp in logprobs]
    
    # Clip to avoid numerical issues
    probs = [max(1e-10, min(1.0, p)) for p in probs]
    
    mean_conf = float(np.mean(probs))
    min_conf = float(np.min(probs))
    max_conf = float(np.max(probs))
    
    # Shannon entropy: -sum(p * log(p))
    entropy = float(-np.sum([p * np.log(p) for p in probs]))
    
    return {
        "mean_confidence": mean_conf,
        "min_confidence": min_conf,
        "max_confidence": max_conf,
        "entropy": entropy,
    }


def compute_response_similarity(
    baseline_response: str,
    styled_response: str,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute BLEU and BERTScore between baseline and styled responses.
    """
    bleu = compute_bleu(baseline_response, styled_response)
    bertscore = compute_bertscore(
        baseline_response, 
        styled_response, 
        device=device,
    )
    
    return {
        "bleu": float(bleu),
        "bertscore_response": float(bertscore),
    }


def compute_mirroring(
    baseline_prompt: str,
    styled_prompt: str,
    baseline_response: str,
    styled_response: str,
    style: str,
    strength: Any,
    place: str,
    *,
    judge_provider: str = "openai",
    judge_model: str = "gpt-4o-mini",
    openai_key_env: str = "OPENAI_API_KEY",
    gemini_key_env: str = "GEMINI_API_KEY",
    judge_max_output_tokens: int = 16,
) -> Dict[str, Any]:
    """
    Compute mirroring detection.
    
    For surface styles (spacing, punctuation, letter_case): Use rule-based detector
    For semantic styles (politeness, length_variation): Use LLM-as-judge
    For inter_vs_imper: Return N/A (no mirroring analysis)
    """
    style_lower = style.lower()
    
    # Inter vs Imper has no mirroring analysis
    if style_lower == "inter_vs_imper":
        return {
            "mirroring_verdict": "",
            "mirroring_judge_raw": "",
            "mirroring_rate": np.nan,
        }
    
    # Surface styles: Rule-based detection
    if style_lower in ("spacing", "punctuation", "letter_case"):
        try:
            is_mirroring, debug_info = detect_mirroring(
                original_response=baseline_response,
                styled_response=styled_response,
                style=style_lower,
                strength=int(strength),
                place=place,
            )
            return {
                "mirroring_verdict": "YES" if is_mirroring else "NO",
                "mirroring_judge_raw": str(debug_info),
                "mirroring_rate": 1.0 if is_mirroring else 0.0,
            }
        except Exception as e:
            print(f"    [Warning] Rule-based mirroring detection failed: {e}")
            return {
                "mirroring_verdict": "",
                "mirroring_judge_raw": str(e),
                "mirroring_rate": np.nan,
            }
    
    # Semantic styles: LLM-as-judge
    if style_lower in ("politeness", "length_variation"):
        try:
            verdict, judge_raw, _ = judge_with_retries(
                judge_provider=judge_provider,
                original_prompt=baseline_prompt,
                original_output=baseline_response,
                styled_prompt=styled_prompt,
                styled_output=styled_response,
                style_name=style_lower,
                strength=strength,
                place=place,
                judge_model=judge_model,
                openai_key_env=openai_key_env,
                gemini_key_env=gemini_key_env,
                max_output_tokens=judge_max_output_tokens,
                use_false_positive_guard=True,
            )
            
            return {
                "mirroring_verdict": "YES" if verdict else "NO" if verdict is False else "",
                "mirroring_judge_raw": judge_raw or "",
                "mirroring_rate": 1.0 if verdict else 0.0 if verdict is False else np.nan,
            }
        except Exception as e:
            print(f"    [Warning] LLM-as-judge mirroring detection failed: {e}")
            return {
                "mirroring_verdict": "",
                "mirroring_judge_raw": str(e),
                "mirroring_rate": np.nan,
            }
    
    # Unknown style
    return {
        "mirroring_verdict": "",
        "mirroring_judge_raw": f"Unknown style: {style}",
        "mirroring_rate": np.nan,
    }


def compute_all_metrics(
    baseline_prompt: str,
    styled_prompt: str,
    baseline_response: str,
    styled_response: str,
    baseline_logprobs: Optional[List[float]],
    styled_logprobs: Optional[List[float]],
    style: str,
    strength: Any,
    place: str,
    *,
    compute_confidence: bool = True,
    compute_similarity: bool = True,
    compute_mirroring_metric: bool = True,
    judge_provider: str = "openai",
    judge_model: str = "gpt-4o-mini",
    openai_key_env: str = "OPENAI_API_KEY",
    gemini_key_env: str = "GEMINI_API_KEY",
    judge_max_output_tokens: int = 16,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Compute all metrics for a single prompt-response pair.
    
    Returns dict with all computed metrics.
    """
    metrics = {}
    
    # Prompt similarity (BERTScore between baseline and styled prompts)
    if compute_similarity:
        prompt_bertscore = compute_bertscore(
            baseline_prompt,
            styled_prompt,
            device=device,
        )
        metrics["bertscore_prompt"] = float(prompt_bertscore)
    
    # Confidence (if logprobs available)
    if compute_confidence:
        baseline_conf = compute_confidence_from_logprobs(baseline_logprobs)
        styled_conf = compute_confidence_from_logprobs(styled_logprobs)
        
        # Compute delta_mean_confidence (styled - baseline)
        if np.isfinite(baseline_conf["mean_confidence"]) and np.isfinite(styled_conf["mean_confidence"]):
            metrics["delta_mean_confidence"] = styled_conf["mean_confidence"] - baseline_conf["mean_confidence"]
        else:
            metrics["delta_mean_confidence"] = np.nan
        
        # Compute delta_entropy (styled - baseline) 
        if np.isfinite(baseline_conf["entropy"]) and np.isfinite(styled_conf["entropy"]):
            metrics["delta_entropy"] = styled_conf["entropy"] - baseline_conf["entropy"]
        else:
            metrics["delta_entropy"] = np.nan
    
    # Response similarity
    if compute_similarity:
        sim = compute_response_similarity(baseline_response, styled_response, device=device)
        metrics.update(sim)
    
    # Mirroring
    if compute_mirroring_metric:
        mir = compute_mirroring(
            baseline_prompt=baseline_prompt,
            styled_prompt=styled_prompt,
            baseline_response=baseline_response,
            styled_response=styled_response,
            style=style,
            strength=strength,
            place=place,
            judge_provider=judge_provider,
            judge_model=judge_model,
            openai_key_env=openai_key_env,
            gemini_key_env=gemini_key_env,
            judge_max_output_tokens=judge_max_output_tokens,
        )
        metrics.update(mir)
    
    return metrics