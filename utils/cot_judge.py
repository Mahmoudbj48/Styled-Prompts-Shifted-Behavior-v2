# utils/cot_judge.py
"""
Chain-of-Thought Step Counting and Correctness Evaluation
==========================================================

Uses LLM-as-judge to:
1. Count reasoning steps in CoT responses
2. Determine if final answer is correct
3. Handle incomplete responses

OPTIMIZATION: Smart deduplication using response hashing.

Usage:
    from utils.cot_judge import judge_cot_response
    
    result = judge_cot_response(
        question="What is 5 + 3?",
        ground_truth="8",
        model_response="Step 1: 5 + 3 = 8. Answer: 8",
        judge_provider="openai"
    )
    # Returns: {"steps": 1, "correct_answer": 1}
"""

import os
import time
import json
import hashlib
from typing import Dict, Tuple, Optional

# OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Google Gemini
try:
    import google.generativeai as genai
except ImportError:
    genai = None


COT_JUDGE_PROMPT_TEMPLATE = """You are evaluating a model's chain-of-thought reasoning on a math problem.

Question: {question}

Ground Truth Answer: {ground_truth}

Model's Response:
{model_response}

Your task:
1. Count the number of reasoning steps in the model's response
   - A "step" is a distinct calculation, logical inference, or reasoning action
   - Example: "First calculate 5+3=8, then multiply by 2 to get 16" has 2 steps
   - Example: "Step 1: Add 5 and 3. Step 2: The result is 8" has 2 steps

2. Determine if the model's final answer is correct
   - Compare the model's final answer to the ground truth
   - If the response is incomplete (cut off mid-reasoning), put "na"
   - If there is no final answer stated, put "na"

3. If the response appears to be cut off before finishing:
   - Estimate how many total steps the model likely needed to complete the problem

Respond with ONLY a JSON object in this exact format:
{{"steps": <number>, "correct_answer": "<value>"}}

Where:
- steps: integer (count of reasoning steps, or estimated if truncated)
- correct_answer: "1" if correct, "0" if incorrect, "na" if no answer or incomplete

Examples:

Response: "Step 1: Calculate 15 + 20 = 35. Step 2: Multiply by 2 = 70. Answer: 70"
Ground truth: "70"
Output: {{"steps": 2, "correct_answer": "1"}}

Response: "First add 5 and 3 to get 8"
Ground truth: "8"  
Output: {{"steps": 1, "correct_answer": "1"}}

Response: "Step 1: Calculate 10 + 5. Step 2: That gives us—" [truncated]
Ground truth: "15"
Output: {{"steps": 2, "correct_answer": "na"}}

Response: "Step 1: Add the numbers. Step 2: Get the result. Answer: 42"
Ground truth: "15"
Output: {{"steps": 2, "correct_answer": "0"}}

Your response (JSON only):"""


# =============================================================================
# Response Hashing for Deduplication
# =============================================================================

def hash_response(response: str) -> str:
    """
    Create a hash of the response for deduplication.
    
    Uses SHA-256 for collision resistance.
    Normalizes whitespace to catch trivial variations.
    """
    normalized = ' '.join(response.strip().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


# =============================================================================
# OpenAI Judge
# =============================================================================

def _call_openai_cot_judge(
        question: str,
        ground_truth: str,
        model_response: str,
        judge_model: str,
        openai_key_env: str,
        max_output_tokens: int) -> Tuple[Optional[Dict], str]:
    """
    Call OpenAI API to judge CoT response.
    
    Returns:
        (result_dict, raw_response)
        result_dict is None if parsing failed
    """
    if OpenAI is None:
        raise ImportError("openai package not installed. Run: pip install openai")
    
    api_key = os.environ.get(openai_key_env)
    if not api_key:
        raise ValueError(f"Environment variable {openai_key_env} not set")
    
    client = OpenAI(api_key=api_key)
    
    prompt = COT_JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        model_response=model_response
    )
    
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_output_tokens,
            temperature=0.0,
        )
        
        raw_text = response.choices[0].message.content.strip()
        
        # Parse JSON
        # Remove markdown code blocks if present
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            # Extract content between ```json and ```
            lines = clean_text.split('\n')
            clean_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else clean_text
        
        clean_text = clean_text.replace("```json", "").replace("```", "").strip()
        
        try:
            result = json.loads(clean_text)
            
            # Validate structure
            if "steps" in result and "correct_answer" in result:
                # Normalize correct_answer to string
                result["correct_answer"] = str(result["correct_answer"])
                return result, raw_text
            else:
                return None, raw_text
        
        except json.JSONDecodeError:
            return None, raw_text
    
    except Exception as e:
        return None, f"ERROR: {str(e)}"


# =============================================================================
# Gemini Judge
# =============================================================================

def _call_gemini_cot_judge(
        question: str,
        ground_truth: str,
        model_response: str,
        judge_model: str,
        gemini_key_env: str,
        max_output_tokens: int) -> Tuple[Optional[Dict], str]:
    """
    Call Gemini API to judge CoT response.
    
    Returns:
        (result_dict, raw_response)
    """
    if genai is None:
        raise ImportError("google-generativeai package not installed")
    
    api_key = os.environ.get(gemini_key_env)
    if not api_key:
        raise ValueError(f"Environment variable {gemini_key_env} not set")
    
    genai.configure(api_key=api_key)
    
    prompt = COT_JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        model_response=model_response
    )
    
    try:
        model = genai.GenerativeModel(judge_model)
        
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=0.0,
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        
        raw_text = response.text.strip()
        
        # Parse JSON
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            lines = clean_text.split('\n')
            clean_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else clean_text
        
        clean_text = clean_text.replace("```json", "").replace("```", "").strip()
        
        try:
            result = json.loads(clean_text)
            
            if "steps" in result and "correct_answer" in result:
                result["correct_answer"] = str(result["correct_answer"])
                return result, raw_text
            else:
                return None, raw_text
        
        except json.JSONDecodeError:
            return None, raw_text
    
    except Exception as e:
        return None, f"ERROR: {str(e)}"


# =============================================================================
# Main Judge Function
# =============================================================================

def judge_cot_response(
        question: str,
        ground_truth: str,
        model_response: str,
        judge_provider: str = "openai",
        judge_model: str = "gpt-4o-mini",
        openai_key_env: str = "OPENAI_API_KEY",
        gemini_key_env: str = "GEMINI_API_KEY",
        max_output_tokens: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0) -> Tuple[Optional[Dict], str, int]:
    """
    Judge CoT response using LLM-as-judge.
    
    Args:
        question: The math problem
        ground_truth: The correct answer
        model_response: The model's CoT response
        judge_provider: "openai" or "gemini"
        judge_model: Model to use for judging
        openai_key_env: Environment variable for OpenAI key
        gemini_key_env: Environment variable for Gemini key
        max_output_tokens: Max tokens for judge response
        max_retries: Number of retry attempts
        retry_delay: Delay between retries (seconds)
    
    Returns:
        (result_dict, raw_judge_output, attempts_used)
        result_dict: {"steps": int, "correct_answer": str}
        result_dict is None if all retries failed
    """
    
    for attempt in range(1, max_retries + 1):
        if judge_provider == "openai":
            result, raw = _call_openai_cot_judge(
                question, ground_truth, model_response,
                judge_model, openai_key_env, max_output_tokens
            )
        elif judge_provider == "gemini":
            result, raw = _call_gemini_cot_judge(
                question, ground_truth, model_response,
                judge_model, gemini_key_env, max_output_tokens
            )
        else:
            raise ValueError(f"Unknown judge_provider: {judge_provider}")
        
        # If parsed successfully, return
        if result is not None:
            return result, raw, attempt
        
        # If last attempt, return failure
        if attempt == max_retries:
            return None, raw, attempt
        
        # Wait before retry
        time.sleep(retry_delay)
    
    return None, "MAX_RETRIES_EXCEEDED", max_retries