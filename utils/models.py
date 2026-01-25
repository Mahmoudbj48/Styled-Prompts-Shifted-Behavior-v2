"""
Model loading and generation utilities.

Functions:
    - load_model: Load a model and tokenizer from HuggingFace
    - generate_response: Generate text from a prompt
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name, device_map="auto", dtype="float32"):
    """
    Load a language model and tokenizer.
    
    Args:
        model_name (str): HuggingFace model identifier
        device_map (str): Device mapping strategy ('auto', 'cpu', 'cuda')
        dtype (str): Data type ('float32' or 'float16')
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_name}...")
    
    # Determine torch dtype
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    
    # Fix for models without pad token (GPT-2, Llama, etc.)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    model.eval()  # Set to evaluation mode
    print(f"Model loaded successfully on {model.device}")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=100, do_sample=False):
    """
    Generate text response from a prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt (str): Input text prompt
        max_new_tokens (int): Maximum tokens to generate
        do_sample (bool): Whether to use sampling (False = greedy decoding)
    
    Returns:
        str: Generated response text (only the new tokens, not the prompt)
    """
    # Apply chat template if available (for instruction-tuned models)
    try:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
    except:
        # Fallback for models without chat template
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Create attention mask
    attention_mask = torch.ones_like(inputs)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )
    
    # Decode only the new tokens (exclude the input prompt)
    input_len = inputs.shape[1]
    response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    
    return response