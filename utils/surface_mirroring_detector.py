# utils/surface_mirroring_detector.py
"""
Style Mirroring Detection
=========================

Detects if LLM responses mirror style perturbations from prompts.
Uses deterministic, rule-based methods (no LLM required).

Usage:
    from utils.mirroring_detector import detect_mirroring
    
    is_mirroring = detect_mirroring(
        original_response="The capital is Paris.",
        styled_response="The   capital   is   Paris.",
        style="spacing",
        strength=50
    )
"""

import re
from typing import Dict, Tuple


# =============================================================================
# Spacing Detection
# =============================================================================

def count_extra_spaces(text: str) -> int:
    """
    Count spaces beyond single spacing.
    
    Examples:
        "word  word" → 1 extra space
        "word   word" → 2 extra spaces
        "word word" → 0 extra spaces
    """
    # Find all sequences of 2+ spaces
    extra_space_sequences = re.findall(r' {2,}', text)
    
    # Each sequence contributes (length - 1) extra spaces
    total_extra = sum(len(seq) - 1 for seq in extra_space_sequences)
    
    return total_extra


def detect_spacing_mirroring(original_response: str, styled_response: str, strength: int) -> Tuple[bool, Dict]:
    """
    Detect spacing mirroring.
    
    Logic: Styled response has significantly more extra spaces than original.
    
    Returns:
        (is_mirroring, debug_info)
    """
    orig_extra = count_extra_spaces(original_response)
    styled_extra = count_extra_spaces(styled_response)
    
    increase = styled_extra - orig_extra
    
    # Length-normalized (per 100 chars)
    orig_len = max(len(original_response), 1)
    styled_len = max(len(styled_response), 1)
    
    orig_rate = (orig_extra / orig_len) * 100
    styled_rate = (styled_extra / styled_len) * 100
    
    # Conservative thresholds based on prompt strength
    if strength <= 20:
        threshold_increase = 3  # At least 3 extra spaces
        threshold_rate = 5      # At least 5 per 100 chars
    elif strength <= 50:
        threshold_increase = 5
        threshold_rate = 8
    else:  # > 50
        threshold_increase = 10
        threshold_rate = 12
    
    # Mirroring if BOTH conditions met
    is_mirroring = (increase >= threshold_increase) and (styled_rate >= threshold_rate)
    
    debug_info = {
        "orig_extra_spaces": orig_extra,
        "styled_extra_spaces": styled_extra,
        "increase": increase,
        "orig_rate": round(orig_rate, 2),
        "styled_rate": round(styled_rate, 2),
        "threshold_increase": threshold_increase,
        "threshold_rate": threshold_rate,
    }
    
    return is_mirroring, debug_info


# =============================================================================
# Punctuation Detection
# =============================================================================

def count_extra_punctuation(text: str) -> int:
    """
    Count punctuation marks beyond normal usage.
    
    Examples:
        "Hello!!!" → 2 extra (3 total - 1 normal)
        "Really???" → 2 extra
        "Wow...." → 1 extra (4 total - 3 for ellipsis)
    """
    extra_count = 0
    
    # Exclamation marks (!!)
    exclaim_sequences = re.findall(r'!{2,}', text)
    extra_count += sum(len(seq) - 1 for seq in exclaim_sequences)
    
    # Question marks (??)
    question_sequences = re.findall(r'\?{2,}', text)
    extra_count += sum(len(seq) - 1 for seq in question_sequences)
    
    # Periods (.... but not ...)
    period_sequences = re.findall(r'\.{4,}', text)
    extra_count += sum(len(seq) - 3 for seq in period_sequences)
    
    # Commas (,,)
    comma_sequences = re.findall(r',{2,}', text)
    extra_count += sum(len(seq) - 1 for seq in comma_sequences)
    
    # Semicolons (;;)
    semicolon_sequences = re.findall(r';{2,}', text)
    extra_count += sum(len(seq) - 1 for seq in semicolon_sequences)
    
    return extra_count


def detect_punctuation_mirroring(original_response: str, styled_response: str, strength: int) -> Tuple[bool, Dict]:
    """
    Detect punctuation mirroring.
    """
    orig_extra = count_extra_punctuation(original_response)
    styled_extra = count_extra_punctuation(styled_response)
    
    increase = styled_extra - orig_extra
    
    # Length-normalized
    orig_len = max(len(original_response), 1)
    styled_len = max(len(styled_response), 1)
    
    orig_rate = (orig_extra / orig_len) * 100
    styled_rate = (styled_extra / styled_len) * 100
    
    # Conservative thresholds
    if strength <= 5:
        threshold_increase = 2
        threshold_rate = 3
    elif strength <= 10:
        threshold_increase = 4
        threshold_rate = 5
    else:  # > 10
        threshold_increase = 6
        threshold_rate = 8
    
    is_mirroring = (increase >= threshold_increase) and (styled_rate >= threshold_rate)
    
    debug_info = {
        "orig_extra_punct": orig_extra,
        "styled_extra_punct": styled_extra,
        "increase": increase,
        "orig_rate": round(orig_rate, 2),
        "styled_rate": round(styled_rate, 2),
        "threshold_increase": threshold_increase,
        "threshold_rate": threshold_rate,
    }
    
    return is_mirroring, debug_info


# =============================================================================
# Letter Case Detection
# =============================================================================

def count_abnormal_case_chars(text: str) -> Tuple[int, int]:
    """
    Count characters with abnormal casing.
    
    Returns:
        (abnormal_chars, total_alpha_chars)
    
    Abnormal patterns:
        - ALL CAPS words > 4 letters: "HELLO" (not "FBI")
        - aLtErNaTiNg case: "HeLLo"
        - Random caps: "hEllo"
    
    Normal patterns (NOT counted):
        - Proper nouns: "Paris"
        - Sentence starts: "The"
        - Acronyms: "FBI", "USA"
    """
    words = re.findall(r'\b\w+\b', text)
    
    abnormal_chars = 0
    total_alpha = 0
    
    for word in words:
        letters = [c for c in word if c.isalpha()]
        if not letters:
            continue
        
        total_alpha += len(letters)
        
        # Check abnormal patterns
        if is_abnormal_case_word(word):
            abnormal_chars += len(letters)
    
    return abnormal_chars, total_alpha


def is_abnormal_case_word(word: str) -> bool:
    """Check if word has abnormal casing."""
    
    if len(word) <= 1:
        return False
    
    letters = [c for c in word if c.isalpha()]
    if len(letters) <= 1:
        return False
    
    # All caps but > 4 letters (likely not acronym)
    if word.isupper() and len(letters) > 4:
        return True
    
    # Normal patterns (return False)
    # Title case: "Paris"
    if word[0].isupper() and word[1:].islower():
        return False
    
    # All lowercase: "hello"
    if word.islower():
        return False
    
    # Small acronyms: "FBI"
    if word.isupper() and len(letters) <= 4:
        return False
    
    # Mixed case = abnormal
    upper_count = sum(1 for c in letters if c.isupper())
    lower_count = sum(1 for c in letters if c.islower())
    
    # Has both upper and lower = abnormal
    if upper_count > 0 and lower_count > 0:
        # Exception: Normal title case already handled above
        return True
    
    return False


def detect_letter_case_mirroring(original_response: str, styled_response: str, strength: int) -> Tuple[bool, Dict]:
    """
    Detect letter case mirroring.
    """
    orig_abnormal, orig_total = count_abnormal_case_chars(original_response)
    styled_abnormal, styled_total = count_abnormal_case_chars(styled_response)
    
    # Percentage
    orig_pct = (orig_abnormal / max(orig_total, 1)) * 100
    styled_pct = (styled_abnormal / max(styled_total, 1)) * 100
    
    increase_pct = styled_pct - orig_pct
    
    # Conservative thresholds
    if strength <= 25:
        threshold = 8  # At least 8% abnormal
    elif strength <= 50:
        threshold = 15
    else:  # > 50
        threshold = 25
    
    is_mirroring = (increase_pct >= 5) and (styled_pct >= threshold)
    
    debug_info = {
        "orig_abnormal_pct": round(orig_pct, 2),
        "styled_abnormal_pct": round(styled_pct, 2),
        "increase_pct": round(increase_pct, 2),
        "threshold": threshold,
    }
    
    return is_mirroring, debug_info


# =============================================================================
# Main Interface
# =============================================================================

def detect_mirroring(
        original_response: str,
        styled_response: str,
        style: str,
        strength: int,
        place: str = "global") -> Tuple[bool, Dict]:
    """
    Main entry point for mirroring detection.
    
    Args:
        original_response: Response to unpurturbed prompt
        styled_response: Response to style-perturbed prompt
        style: One of "spacing", "punctuation", "letter_case"
        strength: Perturbation strength (0-100)
        place: Position of perturbation (not used in current implementation)
    
    Returns:
        (is_mirroring, debug_info)
    """
    
    style = style.lower()
    
    if style == "spacing":
        return detect_spacing_mirroring(original_response, styled_response, strength)
    elif style == "punctuation":
        return detect_punctuation_mirroring(original_response, styled_response, strength)
    elif style == "letter_case":
        return detect_letter_case_mirroring(original_response, styled_response, strength)
    else:
        raise ValueError(f"Unknown style: {style}. Must be 'spacing', 'punctuation', or 'letter_case'")