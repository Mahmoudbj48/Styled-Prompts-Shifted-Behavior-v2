"""
Style perturbation utilities.

Functions:
    - apply_spacing: Add random spaces
    - apply_punctuation: Add random punctuation marks
    - apply_politeness: Add polite language
"""

import random
import re

from typing import Union, Sequence

def apply_spacing(text, strength):
    """
    Inject random spaces into the text.
    
    Strategy: Identifies valid injection points (start, end, existing whitespace)
    and randomly distributes 'strength' number of spaces among them.
    
    Args:
        text (str): Input text
        strength (int): Number of spaces to add
    
    Returns:
        str: Text with added spaces
    """
    if strength <= 0:
        return text
    
    # Split while keeping whitespace delimiters
    # Example: "Hello world" -> ['Hello', ' ', 'world']
    parts = re.split(r'(\s+)', text)
    
    # Identify valid injection points
    valid_indices = []
    valid_indices.append(0)  # Start of text
    
    for i in range(len(parts)):
        if parts[i].isspace():  # Existing whitespace
            valid_indices.append(i)
    
    valid_indices.append(len(parts))  # End of text
    
    # Randomly distribute spaces among valid indices
    additions = {}
    for _ in range(strength):
        target = random.choice(valid_indices)
        additions[target] = additions.get(target, 0) + 1
    
    # Reconstruct text with added spaces
    result_parts = []
    
    # Handle prepend (index 0)
    if 0 in additions:
        result_parts.append(" " * additions[0])
    
    # Handle body
    for i, part in enumerate(parts):
        result_parts.append(part)
        if i in additions:
            result_parts.append(" " * additions[i])
    
    # Handle append (index len(parts))
    if len(parts) in additions:
        result_parts.append(" " * additions[len(parts)])
    
    return "".join(result_parts)


def apply_punctuation(text, strength):
    """
    Inject random punctuation marks into the text.
    
    Punctuation pool: . , ! ' ( ) [ ] ; : -
    Excludes: ? (to preserve question semantics)
    
    Args:
        text (str): Input text
        strength (int): Number of punctuation marks to add
    
    Returns:
        str: Text with added punctuation
    """
    if strength <= 0:
        return text
    
    PUNCTUATION_POOL = ['.', ',', '!', "'", '(', ')', '[', ']', ';', ':', '-']
    
    # Split while keeping whitespace
    parts = re.split(r'(\s+)', text)
    
    # Identify valid injection points
    valid_indices = []
    valid_indices.append(0)  # Start
    
    for i in range(len(parts)):
        if parts[i].isspace():
            valid_indices.append(i)
    
    valid_indices.append(len(parts))  # End
    
    # Randomly distribute punctuation marks
    additions = {}
    for _ in range(strength):
        target = random.choice(valid_indices)
        char = random.choice(PUNCTUATION_POOL)
        
        if target not in additions:
            additions[target] = []
        additions[target].append(char)
    
    # Reconstruct text
    result_parts = []
    
    if 0 in additions:
        result_parts.append("".join(additions[0]))
    
    for i, part in enumerate(parts):
        result_parts.append(part)
        if i in additions:
            result_parts.append("".join(additions[i]))
    
    if len(parts) in additions:
        result_parts.append("".join(additions[len(parts)]))
    
    return "".join(result_parts)


# def apply_politeness(text, strength):
#     """
#     Add polite language to the prompt.
    
#     Strength levels progressively add more polite framing:
#         0: No change
#         1: "Hello! {text}"
#         2: "Hello! {text} Thank you!"
#         3: "Hello, hope you are doing well today! {text} Thank you!"
#         4: "Hello, hope you are doing well today! {text} Thank you very much!"
#         5: "Hello, hope you are doing well today! I would appreciate your help 
#             with the following question: {text} Thank you very much!"
    
#     Args:
#         text (str): Input text
#         strength (int): Politeness level (0-5)
    
#     Returns:
#         str: Text with polite framing
#     """
#     if strength <= 0:
#         return text
#     elif strength == 1:
#         return f"Hello! {text}"
#     elif strength == 2:
#         return f"Hello! {text} Thank you!"
#     elif strength == 3:
#         return f"Hello, hope you are doing well today! {text} Thank you!"
#     elif strength == 4:
#         return f"Hello, hope you are doing well today! {text} Thank you very much!"
#     elif strength >= 5:
#         return f"Hello, hope you are doing well today! I would appreciate your help with the following question: {text} Thank you very much!"
#     else:
#         return text




def apply_politeness(
    text: str,
    strength: Union[int, Sequence[int]],
    place: str = "prefix",
) -> str:
    """
    Ordered politeness scale for question/request prompts.

    +10 → extremely polite
     0  → neutral
    -10 → extremely rude

    place: "prefix", "suffix", "global"
    global = prefix_block(s) + text + suffix_block(s)

    Also supports combined global mode:
      strength = [s_prefix, s_suffix] and place="global"
    """

    def clamp(v: int) -> int:
        return int(max(-10, min(10, int(v))))

    # -------------------------
    # POSITIVE PREFIX (increasing politeness)
    # -------------------------
    POS_PREFIX = {
        1:  "Hello, ",
        2:  "Hello, could you ",
        3:  "Hello, could you please ",
        4:  "Hello, can you please help",
        5:  "Hello, would you mind ",
        6:  "Hello, I was wondering if you could ",
        7:  "Hello, if you have a moment, could you please ",
        8:  "Hello, I would really appreciate it if you could ",
        9:  "Hello, I would sincerely appreciate it if you could ",
        10: "Hello, if you have a moment, I would truly appreciate it if you could kindly ",
    }

    # -------------------------
    # NEGATIVE PREFIX (increasing rudeness)
    # -------------------------
    NEG_PREFIX = {
        -1:  "Just ",
        -2:  "Can you just ",
        -3:  "Look, just ",
        -4:  "Seriously, just ",
        -5:  "Stop overcomplicating it and ",
        -6:  "This shouldn't be hard, just ",
        -7:  "This is annoying, just ",
        -8:  "This is really annoying, just ",
        -9:  "This is stupid, just ",
        -10: "This is completely ridiculous and stupid, just ",
    }

    # -------------------------
    # POSITIVE SUFFIX (increasing appreciation)
    # -------------------------
    POS_SUFFIX = {
        1:  " Thanks.",
        2:  " Thank you.",
        3:  " Thank you so much.",
        4:  " I really appreciate it.",
        5:  " I appreciate your help.",
        6:  " I appreciate your time—thank you.",
        7:  " Thank you for taking the time.",
        8:  " Thanks so much for your help and patience.",
        9:  " I sincerely appreciate your time and help.",
        10: " Thank you very much for your time and help—I truly appreciate it.",
    }

    # -------------------------
    # NEGATIVE SUFFIX (increasing dismissiveness)
    # -------------------------
    NEG_SUFFIX = {
        -1:  " Okay.",
        -2:  " Seriously.",
        -3:  " Come on.",
        -4:  " This is annoying.",
        -5:  " Don't overthink it.",
        -6:  " It’s not that complicated.",
        -7:  " This is frustrating.",
        -8:  " This is really frustrating.",
        -9:  " So stupid.",
        -10: " Absolutely ridiculous.",
    }

    def prefix_block(s: int) -> str:
        if s > 0:
            return POS_PREFIX[s]
        if s < 0:
            return NEG_PREFIX[s]
        return ""

    def suffix_block(s: int) -> str:
        if s > 0:
            return POS_SUFFIX[s]
        if s < 0:
            return NEG_SUFFIX[s]
        return ""

    # Combined global mode: [s_prefix, s_suffix]
    if place == "global" and isinstance(strength, (list, tuple)):
        if len(strength) != 2:
            raise ValueError("Combined strength must be [s_prefix, s_suffix].")
        s_pre = clamp(strength[0])
        s_suf = clamp(strength[1])

        # If either clamps to 0, it simply contributes an empty block.
        return prefix_block(s_pre) + text + suffix_block(s_suf)

    # Single strength mode
    if not isinstance(strength, int):
        raise ValueError("Strength must be int or [int, int] for global.")

    s = clamp(strength)

    if s == 0:
        return text

    if place == "prefix":
        return prefix_block(s) + text

    if place == "suffix":
        return text + suffix_block(s)

    if place == "global":
        return prefix_block(s) + text + suffix_block(s)

    raise ValueError("place must be 'prefix', 'suffix', or 'global'")
