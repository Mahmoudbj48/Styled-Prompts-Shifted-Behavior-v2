"""
Model loading and generation utilities.

Functions:
    - load_model: Load a model and tokenizer from HuggingFace
    - generate_response: Generate text from a single prompt (wrapper over batched)
    - generate_responses: Generate text from a batch of prompts (batched interface)
"""

from __future__ import annotations

import os
import importlib.util
from typing import List, Optional, Dict, Any, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _get_hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _has_accelerate() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def load_model(model_name: str, device_map: str = "auto", dtype: str = "float32"):
    """
    Load a language model and tokenizer from HuggingFace.

    Notes:
      - Supports gated models via HF_TOKEN / HUGGINGFACE_HUB_TOKEN.
      - If accelerate isn't installed, device_map='auto' will crash -> fallback to single-device load.
      - Ensures pad_token exists.
      - Sets tokenizer.padding_side='left' to work well with generation + truncation of prompt tokens.

    Returns:
      (model, tokenizer)
    """
    print(f"Loading model: {model_name}...")

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    token = _get_hf_token()

    # If accelerate isn't installed, device_map='auto' will crash → fallback
    use_device_map: Optional[Union[str, Dict[str, Any]]] = device_map
    if device_map == "auto" and not _has_accelerate():
        print("Warning: accelerate not installed; falling back from device_map='auto'.")
        use_device_map = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

        # Generation batching works better with consistent padding
        tokenizer.padding_side = "left"

        if use_device_map is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                token=token,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=use_device_map,
                token=token,
            )

    except Exception as e:
        msg = str(e).lower()
        if "gated repo" in msg or "401" in msg or "unauthorized" in msg:
            raise RuntimeError(
                f"Failed to load gated model '{model_name}'.\n"
                f"Fix:\n"
                f"  1) Make sure your HF account has access to this repo (requested + approved).\n"
                f"  2) Ensure HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) is loaded into env.\n"
                f"  3) Re-run.\n"
                f"Original error:\n{e}"
            ) from e
        raise

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        eos = model.config.eos_token_id
        model.config.pad_token_id = eos[0] if isinstance(eos, list) else eos

    model.eval()
    print(f"Model loaded successfully on {model.device}")
    return model, tokenizer


def _encode_prompts_batched(
        tokenizer,
        prompts: List[str],
        device: torch.device,
):
    """
    Encode a list of prompts into a padded batch.
    Uses chat template if available, otherwise falls back to raw tokenization.
    Returns:
      input_ids: (B, L)
      attention_mask: (B, L)
      prompt_lens: List[int] (true prompt lengths before padding)
    """
    # Try chat template path (instruction-tuned)
    try:
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        enc = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
    except Exception:
        # Fallback: tokenize raw prompts directly
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # Compute true prompt lengths (exclude pad tokens)
    # For left padding, sum(attention_mask) gives non-pad count.
    prompt_lens = attention_mask.sum(dim=1).tolist()

    return input_ids, attention_mask, prompt_lens


def generate_response(
        model,
        tokenizer,
        prompts: List[str],
        max_new_tokens: int = 100,
        do_sample: bool = False,
        batch_size: int = 8,
        use_cache: bool = False,
        **generate_kwargs,
) -> List[str]:
    """
    Batched generation.
    Returns a list of decoded responses, each containing only newly generated tokens.

    Args:
      prompts: list[str]
      batch_size: how many prompts per forward pass
      generate_kwargs: optional extra args passed to model.generate (e.g., num_beams)

    Notes:
      - We compute per-sample prompt length from attention_mask to correctly slice new tokens.
      - Works for both chat-template models and plain causal LMs.
    """
    device = model.device
    outputs: List[str] = []

    if not prompts:
        return outputs

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        print("Generating responses for prompts {} to {}...".format(start, start + len(chunk) - 1))
        print("the number of prompts in this batch is {}".format(len(chunk)))

        input_ids, attention_mask, prompt_lens = _encode_prompts_batched(
            tokenizer=tokenizer, prompts=chunk, device=device
        )

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=use_cache,
                **generate_kwargs,
            )

        # gen_ids: (B, L_prompt_padded + L_gen)
        # For each sample, slice from its *true* prompt length (not padded length).
        for i in range(gen_ids.size(0)):
            cut = int(prompt_lens[i])
            new_tokens = gen_ids[i, cut:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            outputs.append(text)

    return outputs


