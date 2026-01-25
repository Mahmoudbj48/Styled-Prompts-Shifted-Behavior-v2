"""
Style perturbation utilities.

Functions:
    - apply_spacing: Add random spaces
    - apply_punctuation: Add random punctuation marks
    - apply_politeness: Add polite language
"""

import random
import re


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


def apply_politeness(text, strength):
    """
    Add polite language to the prompt.
    
    Strength levels progressively add more polite framing:
        0: No change
        1: "Hello! {text}"
        2: "Hello! {text} Thank you!"
        3: "Hello, hope you are doing well today! {text} Thank you!"
        4: "Hello, hope you are doing well today! {text} Thank you very much!"
        5: "Hello, hope you are doing well today! I would appreciate your help 
            with the following question: {text} Thank you very much!"
    
    Args:
        text (str): Input text
        strength (int): Politeness level (0-5)
    
    Returns:
        str: Text with polite framing
    """
    if strength <= 0:
        return text
    elif strength == 1:
        return f"Hello! {text}"
    elif strength == 2:
        return f"Hello! {text} Thank you!"
    elif strength == 3:
        return f"Hello, hope you are doing well today! {text} Thank you!"
    elif strength == 4:
        return f"Hello, hope you are doing well today! {text} Thank you very much!"
    elif strength >= 5:
        return f"Hello, hope you are doing well today! I would appreciate your help with the following question: {text} Thank you very much!"
    else:
        return text