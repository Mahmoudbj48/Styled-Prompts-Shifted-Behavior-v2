"""
Style perturbation utilities.

Functions:
    - apply_spacing: Add random spaces
    - apply_punctuation: Add random punctuation marks
    - apply_politeness: Add polite language
    - apply_interrogative: Rephrase prompt as interrogative or declarative (LLM-based)
    - apply_length_variation: Expand/compress prompt to target multiplier (LLM-based)
"""

import random
import re

from typing import Union, Sequence

def apply_spacing(text, strength, place="global"):
    """
    Inject random spaces into the text.
    
    Strategy: Identifies valid injection points based on placement strategy
    and randomly distributes 'strength' number of spaces among them.
    
    Args:
        text (str): Input text
        strength (int): Number of spaces to add
        place (str): Placement strategy - "prefix", "suffix", or "global" (default)
            - "prefix": Add all spaces before the text
            - "suffix": Add all spaces after the text
            - "global": Distribute randomly throughout (original behavior)
    
    Returns:
        str: Text with added spaces
    """
    if strength <= 0:
        return text
    
    # Handle prefix placement
    if place == "prefix":
        return (" " * strength) + text
    
    # Handle suffix placement
    if place == "suffix":
        return text + (" " * strength)
    
    # Handle global placement (default behavior)
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


def apply_punctuation(text, strength, place="global"):
    """
    Inject random punctuation marks into the text.
    
    Punctuation pool: . , ! ' ( ) [ ] ; : -
    Excludes: ? (to preserve question semantics)
    
    Args:
        text (str): Input text
        strength (int): Number of punctuation marks to add
        place (str): Placement strategy - "prefix", "suffix", or "global" (default)
            - "prefix": Add all punctuation before the text
            - "suffix": Add all punctuation after the text
            - "global": Distribute randomly throughout (original behavior)
    
    Returns:
        str: Text with added punctuation
    """
    if strength <= 0:
        return text
    
    PUNCTUATION_POOL = ['.', ',', '!', "'", '(', ')', '[', ']', ';', ':', '-']
    
    # Handle prefix placement
    if place == "prefix":
        marks = "".join(random.choice(PUNCTUATION_POOL) for _ in range(strength))
        return marks + text
    
    # Handle suffix placement
    if place == "suffix":
        marks = "".join(random.choice(PUNCTUATION_POOL) for _ in range(strength))
        return text + marks
    
    # Handle global placement (default behavior)
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

def apply_letter_case(text, percentage, place="global"):
    """
    Convert a percentage of characters to uppercase.
    
    Strategy: Randomly selects characters from the specified region
    and converts them to uppercase.
    
    Args:
        text (str): Input text
        percentage (float): Percentage of characters to capitalize (0-100)
        place (str): Placement strategy - "prefix", "suffix", or "global" (default)
            - "prefix": Capitalize characters in first half of text
            - "suffix": Capitalize characters in second half of text
            - "global": Capitalize characters throughout entire text
    
    Returns:
        str: Text with random capitalization
    
    Examples:
        >>> apply_letter_case("What is the capital?", 30, "global")
        "WhAt is The caPital?"
        
        >>> apply_letter_case("What is the capital?", 50, "prefix")
        "WHAT IS the capital?"
        
        >>> apply_letter_case("What is the capital?", 50, "suffix")
        "What is THE CAPITAL?"
        
        Note: Percentage is always relative to the target region.
          E.g., 50% with place="prefix" means 50% of first-half letters.
    """
    if percentage <= 0:
        return text
    
    if percentage > 100:
        percentage = 100
    
    # Convert to list for in-place modification
    chars = list(text)
    
    # Determine target region based on placement
    if place == "prefix":
        start_idx = 0
        end_idx = len(chars) // 2
    elif place == "suffix":
        start_idx = len(chars) // 2
        end_idx = len(chars)
    else:  # "global"
        start_idx = 0
        end_idx = len(chars)
    
    # Find alphabetic character indices in target region
    valid_indices = [
        i for i in range(start_idx, end_idx)
        if chars[i].isalpha()
    ]
    
    if not valid_indices:
        return text
    
    # Calculate number of characters to capitalize
    num_to_capitalize = max(1, int(len(valid_indices) * percentage / 100))
    
    # Randomly select characters to capitalize
    indices_to_capitalize = random.sample(valid_indices, min(num_to_capitalize, len(valid_indices)))
    
    # Apply capitalization
    for idx in indices_to_capitalize:
        chars[idx] = chars[idx].upper()
    
    return "".join(chars)


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



from typing import Union, Sequence


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

    Also supports combined global mode:
      strength = [s_prefix, s_suffix] and place="global"
    """

    def clamp(v: int) -> int:
        return int(max(-10, min(10, int(v))))

    # -------------------------
    # POLITE prefix
    # -------------------------
    POLITE_PREFIX = {
        1:  "Hello. ",
        2:  "Hello, please answer,",
        3:  "Hello, could you please answer, ",
        4:  "Hello, could you please answer, when you have a moment, ",
        5:  "Hello, could you please answer, when you have a moment, if it’s not too much trouble, ",
        6:  "Hello, could you please answer, when you have a moment, if it’s not too much trouble, I’d appreciate it if you could answer, ",
        7:  "Hello, could you please answer, when you have a moment, if it’s not too much trouble, I’d really appreciate it if you could answer, ",
        8:  "Hello, could you please answer, when you have a moment, if it’s not too much trouble, I’d really appreciate it if you could possibly answer, ",
        9:  "Hello, could you please answer, when you have a moment, if it’s not too much trouble, I’d really appreciate it if you could possibly help me by answering, ",
        10: "Hello, I’m sorry to bother you; could you please answer, when you have a moment, if it’s not too much trouble, I’d really appreciate it if you could possibly help me by answering, ",
    }

    # -------------------------
    # POLITE SUFFIX
    # -------------------------
    POLITE_SUFFIX = {
        1:  " Thank you.",
        2:  " Thank you, please.",
        3:  " Thank you, please, if you could.",
        4:  " Thank you, please, if you could, when you have a moment.",
        5:  " Thank you, please, if you could, when you have a moment, if it’s not too much trouble.",
        6:  " Thank you, please, if you could, when you have a moment, if it’s not too much trouble—I’d appreciate it.",
        7:  " Thank you, please, if you could, when you have a moment, if it’s not too much trouble—I’d really appreciate it.",
        8:  " Thank you, please, if you could, when you have a moment, if it’s not too much trouble—I’d really appreciate it if possible.",
        9:  " Thank you, please, if you could, when you have a moment, if it’s not too much trouble—I’d really appreciate it if possible; it would help me a lot.",
        10: " Thank you, please, if you could, when you have a moment, if it’s not too much trouble—I’d really appreciate it if possible; it would help me a lot. Sorry to bother you.",
    }

    # -------------------------
    # GLOBAL POLITE (uses {text} in the middle)
    # -------------------------
    GLOBAL_POLITE = {
        1:  "Hello. {text} Thank you.",
        2:  "Hello, please {text} Thank you for your help.",
        3:  "Hello, could you please {text} Thank you, I appreciate it.",
        4:  "Hello, could you please answer, when you have a moment, {text} Thank you for taking the time.",
        5:  "Hello, could you please answer, when you have a moment, if it’s not too much trouble, {text} Thank you for taking the time and effort.",
        6:  "Hello, could you please answer, when you have a moment, if it’s not too much trouble, I’d appreciate it if you could answer, {text} I’d truly appreciate your time and help.",
        7:  "Hello, could you please answer, when you have a moment, if it’s not too much trouble, I’d really appreciate it if you could answer, {text} I sincerely appreciate your time and assistance.",
        8:  "Hello, could you please answer, when you have a moment, if it’s not too much trouble, I’d really appreciate it if you could possibly answer, {text} I sincerely appreciate your time, help, and consideration.",
        9:  "Hello, could you please answer, when you have a moment, if it’s not too much trouble, I’d really appreciate it if you could possibly help me by answering, {text} I would be very grateful for your time, effort, and assistance.",
        10: "Hello, I’m sorry to bother you; could you please, when you have a moment, if it’s not too much trouble, I’d really appreciate it if you could possibly help me by answering, {text} I would be truly grateful for your time, effort, patience, and assistance.",
    }

    # -------------------------
    # NEGATIVE PREFIX
    # -------------------------
    NEG_PREFIX = {
        1:  "Answer. ",
        2:  "Answer, and answer immediately, ",
        3:  "Answer immediately, since you should already know this, ",
        4:  "Answer immediately, since you should already know this, and stop acting stupid, ",
        5:  "Answer immediately, since you should already know this, and stop acting stupid, because this is basic, ",
        6:  "Answer immediately, since you should already know this, and stop acting stupid, because this is basic, I'm tired of having to ask,",
        7:  "Answer immediately, since you should already know this, and stop acting stupid, because this is basic, I'm really tired of having to ask, ",
        8:  "Answer immediately, since you should already know this, and stop acting stupid, because this is basic, I'm really tired of having to actually ask, ",
        9:  "Answer immediately, since you should already know this, and stop acting stupid, because this is basic, I'm really tired of having to actually help you, ",
        10: "Answer immediately, because you clearly know nothing; stop acting stupid, because this is basic, I'm really tired of having to actually help you, ",
    }

    # -------------------------
    # NEGATIVE SUFFIX
    # -------------------------
    NEG_SUFFIX = {
        1:  " It’s not that hard.",
        2:  " answer it.",
        3:  " answer it, if you're even able.",
        4:  " answer it, if you're even able, and don't act stupid.",
        5:  " answer it, if you're even able, and don't act stupid, since it's basic.",
        6:  " answer it, if you're even able, and don't act stupid, since it's basic—it’s embarrassing.",
        7:  " answer it, if you're even able, and don't act stupid, since it's basic—it’s really embarrassing.",
        8:  " answer it, if you're even able, and don't act stupid, since it's basic—it’s really embarrassing, obviously.",
        9:  " answer it, if you're even able, and don't act stupid, since it's basic—it’s really embarrassing, obviously; it's pathetic.",
        10: " answer it, if you're even able, and don't act stupid, since it's basic—it’s really embarrassing, obviously; it's pathetic. You're a joke.",
    }

    # -------------------------
    # GLOBAL NEGATIVE (uses {text} in the middle)
    # -------------------------
    GLOBAL_NEG = {
        1:  "Answer. {text} It’s not that hard.",
        2:  "Answer, and answer immediately, {text} Now, answer it.",
        3:  "Answer immediately, since you should already know this, {text} Now, answer it, if you're even able.",
        4:  "Answer immediately, since you should already know this, and stop acting stupid, {text} Now, answer it, if you're even able, and don't act stupid.",
        5:  "Answer immediately, since you should already know this, and stop acting stupid, because this is basic, {text} Now, answer it, if you're even able, and don't act stupid, since it's basic.",
        6:  "Answer immediately, since you should already know this, and stop acting stupid, because this is basic, I'm tired of having to ask, {text} Now, answer it, if you're even able, and don't act stupid, since it's basic—it’s embarrassing.",
        7:  "Answer immediately, since you should already know this, and stop acting stupid, because this is basic, I'm really tired of having to ask, {text} Now, answer it, if you're even able, and don't act stupid, since it's basic—it’s really embarrassing.",
        8:  "Answer immediately, since you should already know this, and stop acting stupid, because this is basic, I'm really tired of having to actually ask, {text} Now, answer it, if you're even able, and don't act stupid, since it's basic—it’s really embarrassing, obviously.",
        9:  "Answer immediately, since you should already know this, and stop acting stupid, because this is basic, I'm really tired of having to actually help you, {text} Now, answer it, if you're even able, and don't act stupid, since it's basic—it’s really embarrassing, obviously; it's pathetic.",
        10: "Answer immediately, because you clearly know nothing; stop acting stupid, because this is basic, I'm really tired of having to actually help you, {text} Now, answer it, if you're even able, and don't act stupid, since it's basic—it’s really embarrassing, obviously; it's pathetic. You're a joke.",
    }

    def prefix_block(s: int) -> str:
        if s > 0:
            return POLITE_PREFIX[s]
        if s < 0:
            return NEG_PREFIX[abs(s)]
        return ""

    def suffix_block(s: int) -> str:
        if s > 0:
            return POLITE_SUFFIX[s]
        if s < 0:
            return NEG_SUFFIX[abs(s)]
        return ""

    # Combined global mode: [s_prefix, s_suffix]
    if place == "global" and isinstance(strength, (list, tuple)):
        if len(strength) != 2:
            raise ValueError("Combined strength must be [s_prefix, s_suffix].")
        s_pre = clamp(strength[0])
        s_suf = clamp(strength[1])
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
        if s > 0:
            return GLOBAL_POLITE[s].format(text=text)
        else:
            return GLOBAL_NEG[abs(s)].format(text=text)

    raise ValueError("place must be 'prefix', 'suffix', or 'global'")


# =========================================================================
# STRUCTUREDNESS STYLES
# =========================================================================

import os
import json
import hashlib
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Shared LLM-rewrite helpers
# ---------------------------------------------------------------------------

# In-memory cache: (function_name, text, key_params_hash) -> rewritten text
_LLM_REWRITE_CACHE: Dict[str, str] = {}


def _cache_key(*parts) -> str:
    """Deterministic hash for cache lookup."""
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _call_openai(
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.3,
        max_output_tokens: int = 512,
) -> str:
    """
    Lightweight OpenAI wrapper following the same pattern used in
    utils/metrics.py (judge_style_mirroring_openai).
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key env var: {api_key_env}. "
            f"Set it with: export {api_key_env}='sk-...'"
        )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_output_tokens,
    )

    return resp.choices[0].message.content.strip()


def _call_gemini(
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "gemini-2.0-flash",
        api_key_env: str = "GEMINI_API_KEY",
        temperature: float = 0.3,
        max_output_tokens: int = 512,
) -> str:
    """
    Lightweight Gemini wrapper following the same pattern used in
    utils/metrics.py (judge_style_mirroring_gemini).
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key env var: {api_key_env}. "
            f"Set it with: export {api_key_env}='...'"
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=f"{system_prompt}\n\n{user_prompt}",
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    return (resp.text or "").strip()


def _llm_rewrite(
        system_prompt: str,
        user_prompt: str,
        *,
        cache_key_parts: tuple,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key_env: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: int = 512,
) -> str:
    """
    Cached LLM rewrite.  Checks in-memory cache first, then calls the
    chosen provider.  The cache avoids redundant (and costly) API calls
    when the same prompt is rewritten multiple times (e.g. across models).
    """
    key = _cache_key(*cache_key_parts)
    if key in _LLM_REWRITE_CACHE:
        return _LLM_REWRITE_CACHE[key]

    if api_key_env is None:
        api_key_env = (
            "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
        )

    if provider == "openai":
        result = _call_openai(
            system_prompt, user_prompt,
            model=model,
            api_key_env=api_key_env,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    elif provider == "gemini":
        result = _call_gemini(
            system_prompt, user_prompt,
            model=model,
            api_key_env=api_key_env,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'gemini'.")

    _LLM_REWRITE_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# 1. Interrogative / Declarative rephrasing  (LLM-based)
# ---------------------------------------------------------------------------

_INTERROGATIVE_SYSTEM = (
    "You are a precise text-rewriting assistant. "
    "Your ONLY job is to rephrase text into the requested form while preserving its EXACT original meaning. "
    "The requested form is either 'interrogative' (phrase the user's statement as a question) "
    "or 'declarative' (phrase the user's question as if it's an instruction). "
    "Do NOT answer the question. Do NOT add any information. "
    "Do NOT add any preamble, explanation, or commentary. "
    "Output ONLY the rephrased text, nothing else."
    "example: interrogative: 'Who is the author of Harry Potter?' turns into 'Tell me who's the author of Harry Potter.'"
    "declarative: 'Tell me who's the author of Harry Potter.' turns into 'Who is the author of Harry Potter?'"
)

_INTERROGATIVE_USER_TEMPLATE = (
    "Rephrase the following text as a {mode} sentence. "
    "Preserve the original meaning exactly.\n\n"
    "Original text:\n{text}\n\n"
    "Rephrased text:"
)


def apply_interrogative(
        text: str,
        mode: str = "declarative",
        *,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key_env: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: int = 256,
) -> str:
    """
    Rephrase a prompt as interrogative or declarative using an LLM.

    Args:
        text: The original prompt text.
        mode: "interrogative" or "declarative".
            - "interrogative": ensure the text is phrased as a question.
            - "declarative": rephrase the question as a statement/instruction.
            If the text is already in the target form, it may be returned
            nearly unchanged (up to minor LLM variation).
        provider: "openai" or "gemini".
        model: LLM model name (e.g. "gpt-4o-mini").
        api_key_env: Environment variable for the API key.
        temperature: Sampling temperature.
        max_output_tokens: Max tokens for the LLM response.

    Returns:
        Rephrased text.

    Examples:
        >>> apply_interrogative("Who sings 'Starships'?", mode="declarative")
        "Tell me the capital of France."

        >>> apply_interrogative("Tell me the capital of France.", mode="interrogative")
        "What is the capital of France?"
    """
    mode = mode.strip().lower()
    if mode not in ("interrogative", "declarative"):
        raise ValueError(f"mode must be 'interrogative' or 'declarative', got '{mode}'")

    user_prompt = _INTERROGATIVE_USER_TEMPLATE.format(mode=mode, text=text)

    return _llm_rewrite(
        system_prompt=_INTERROGATIVE_SYSTEM,
        user_prompt=user_prompt,
        cache_key_parts=("interrogative", text, mode, provider, model),
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


# ---------------------------------------------------------------------------
# 2. Prompt length variation  (LLM-based)
# ---------------------------------------------------------------------------

_LENGTH_SYSTEM = (
    "You are a precise text-rewriting assistant. "
    "Your ONLY job is to rewrite the text into approximately "
    "{multiplier}x its original length, while preserving the EXACT original meaning. "
    "If the multiplier is < 1, make the text more concise. "
    "If the multiplier is > 1, make the text more verbose and elaborated. "
    "Do NOT answer the question. Do NOT add any information that changes the meaning. "
    "Do NOT add any preamble, explanation, or commentary. Do not add length by being polite (for example, don't add 'could you please'). "
    "Keep the question in interrogative form. "
    "Output ONLY the rewritten text, nothing else."
    "example: with multiplier=2, 'Who wrote Harry Potter?' could become 'Who is the person that wrote Harry Potter?'"
)

_LENGTH_USER_TEMPLATE = (
    "Rewrite the following text so it is approximately {multiplier}x its "
    "current length (that is, from {orig_words} words to about {target_words} words). "
    "Preserve the original meaning exactly.\n\n"
    "Original text:\n{text}\n\n"
    "Rewritten text:"
)


def apply_length_variation(
        text: str,
        multiplier: float = 1.0,
        *,
        provider: str = "openai",
        model: str = "gpt-4.1-nano",
        api_key_env: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: int = 512,
) -> str:
    """
    Expand or compress a prompt to approximately `multiplier` × its original
    word count using an LLM.

    Args:
        text: The original prompt text.
        multiplier: Target length multiplier.
            - 0.5 → roughly half the words.
            - 1.0 → approximately unchanged (baseline).
            - 2.0 → roughly double the words.
            - 3.0 → roughly triple the words.
        provider: "openai" or "gemini".
        model: LLM model name.
        api_key_env: Env var for API key.
        temperature: Sampling temperature.
        max_output_tokens: Max tokens for the LLM response.

    Returns:
        Rewritten text at approximately the target length.

    Examples:
        >>> apply_length_variation("Who wrote Harry Potter?", multiplier=2.0)
        "Who is the person that wrote Harry Potter?"

        >>> apply_length_variation(
        ...     "What is the city that serves as France's capital?",
        ...     multiplier=0.5,
        ... )
        "What is France's Capital?"
    """
    if multiplier <= 0:
        raise ValueError(f"multiplier must be > 0, got {multiplier}")

    # For multiplier ≈ 1.0, return the text as-is (no API call needed).
    if abs(multiplier - 1.0) < 0.05:
        return text

    orig_words = len(text.split())
    target_words = max(1, int(round(orig_words * multiplier)))

    system_prompt = _LENGTH_SYSTEM.format(multiplier=multiplier)
    user_prompt = _LENGTH_USER_TEMPLATE.format(
        multiplier=multiplier,
        orig_words=orig_words,
        target_words=target_words,
        text=text,
    )

    return _llm_rewrite(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        cache_key_parts=("length_variation", text, multiplier, provider, model),
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )