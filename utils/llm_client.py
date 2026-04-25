"""
utils/llm_client.py
===================
Unified LLM client with support for OpenAI, Gemini, and Claude.
Includes logprobs extraction for confidence computation.

Usage:
    from utils.llm_client import get_llm_client
    
    client = get_llm_client("gpt-5.4")
    response = client.complete(prompt, return_logprobs=True)
    
    print(response["text"])
    print(response["logprobs"])  # List of token log probabilities
    print(response["tokens_used"])
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

try:
    # Try new API first (google-generativeai >= 0.8.0)
    from google import genai
    from google.genai import types as genai_types
    _HAS_GEMINI = True
    _GEMINI_NEW_API = True
except ImportError:
    try:
        # Fall back to old API (google-generativeai < 0.8.0)
        import google.generativeai as genai
        _HAS_GEMINI = True
        _GEMINI_NEW_API = False
        genai_types = None
    except ImportError:
        _HAS_GEMINI = False
        _GEMINI_NEW_API = False
        genai = None
        genai_types = None

try:
    import anthropic as _anthropic_module
    _HAS_CLAUDE = True
except ImportError:
    _HAS_CLAUDE = False


# ── Base class ────────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    """Shared interface for all LLM backends."""

    def __init__(self, model: str):
        self.model = model
        self.supports_logprobs = False  # Override in subclasses

    @abstractmethod
    def complete(
        self,
        prompt: str,
        max_tokens: int = 4_000,
        temperature: float = 0.0,
        return_logprobs: bool = False,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Returns dict with:
            - text: str (response text)
            - tokens_used: int (total tokens)
            - logprobs: Optional[List[float]] (token log probabilities if supported)
        """


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIClient(BaseLLMClient):
    """OpenAI Chat Completions client (GPT-4o, GPT-5.x, o-series, …)."""

    def __init__(self, model: str, api_key_env: str = "OPENAI_API_KEY"):
        super().__init__(model)
        self.supports_logprobs = True
        
        if not _HAS_OPENAI:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable '{api_key_env}' not set.")
        
        self._client = OpenAI(api_key=api_key)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 4_000,
        temperature: float = 0.0,
        return_logprobs: bool = False,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> Dict[str, Any]:
        # gpt-5.x and o-series require max_completion_tokens; gpt-4.x uses max_tokens
        token_key = (
            "max_completion_tokens"
            if self.model.startswith(("gpt-5", "o1", "o3", "o4"))
            else "max_tokens"
        )

        for attempt in range(1, max_retries + 1):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    token_key: max_tokens,
                    "temperature": temperature,
                }
                
                if return_logprobs:
                    kwargs["logprobs"] = True
                    kwargs["top_logprobs"] = 1  # We only need the chosen token
                
                resp = self._client.chat.completions.create(**kwargs)
                
                text = resp.choices[0].message.content.strip()
                tokens = resp.usage.total_tokens if resp.usage else 0
                
                logprobs_list = None
                if return_logprobs and hasattr(resp.choices[0], 'logprobs'):
                    logprobs_content = resp.choices[0].logprobs
                    if logprobs_content and hasattr(logprobs_content, 'content'):
                        logprobs_list = [
                            token.logprob 
                            for token in logprobs_content.content 
                            if hasattr(token, 'logprob')
                        ]
                
                return {
                    "text": text,
                    "tokens_used": tokens,
                    "logprobs": logprobs_list,
                }

            except Exception as e:
                print(f"    [OpenAI] Attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)

        return {"text": "", "tokens_used": 0, "logprobs": None}


# ── Gemini ────────────────────────────────────────────────────────────────────

class GeminiClient(BaseLLMClient):
    """Google Gemini client (gemini-1.5-pro, gemini-2.0-flash, gemini-2.5-flash, …)."""

    def __init__(self, model: str, api_key_env: str = "GEMINI_API_KEY"):
        super().__init__(model)
        self.supports_logprobs = _GEMINI_NEW_API  # Only new API supports logprobs
        
        if not _HAS_GEMINI:
            raise ImportError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            )
        
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable '{api_key_env}' not set.")
        
        if _GEMINI_NEW_API:
            # New API: use Client
            self._client = genai.Client(api_key=api_key)
            self._model_name = model
        else:
            # Old API: use configure + GenerativeModel
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(model)
            self._model_name = model

    def complete(
        self,
        prompt: str,
        max_tokens: int = 4_000,
        temperature: float = 0.0,
        return_logprobs: bool = False,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> Dict[str, Any]:
        
        for attempt in range(1, max_retries + 1):
            try:
                if _GEMINI_NEW_API:
                    # New API (google.genai >= 0.8.0)
                    cfg_kwargs = {
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    
                    if return_logprobs:
                        cfg_kwargs["response_logprobs"] = True
                    
                    cfg = genai_types.GenerateContentConfig(**cfg_kwargs)
                    
                    resp = self._client.models.generate_content(
                        model=self._model_name,
                        contents=prompt,
                        config=cfg,
                    )
                    
                    text = (resp.text or "").strip()
                    tokens = getattr(
                        getattr(resp, "usage_metadata", None),
                        "total_token_count", 
                        0
                    )
                    
                    logprobs_list = None
                    if return_logprobs and hasattr(resp, 'candidates'):
                        candidate = resp.candidates[0] if resp.candidates else None
                        if candidate and hasattr(candidate, 'logprobs_result'):
                            logprobs_result = candidate.logprobs_result
                            if logprobs_result and hasattr(logprobs_result, 'top_candidates'):
                                # New API structure
                                logprobs_list = []
                                for position in logprobs_result.top_candidates:
                                    if position.candidates:
                                        # Take the first (most likely) candidate's log_probability
                                        logprobs_list.append(position.candidates[0].log_probability)
                
                else:
                    # Old API (google.generativeai < 0.8.0)
                    cfg = genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    )
                    
                    resp = self._model.generate_content(prompt, generation_config=cfg)
                    
                    text = resp.text.strip() if hasattr(resp, 'text') else str(resp)
                    tokens = getattr(
                        getattr(resp, "usage_metadata", None),
                        "total_token_count", 
                        0
                    )
                    
                    # Old API doesn't support logprobs
                    logprobs_list = None
                
                return {
                    "text": text,
                    "tokens_used": tokens,
                    "logprobs": logprobs_list,
                }

            except Exception as e:
                print(f"    [Gemini] Attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)

        return {"text": "", "tokens_used": 0, "logprobs": None}


# ── Claude ────────────────────────────────────────────────────────────────────

class ClaudeClient(BaseLLMClient):
    """Anthropic Claude client (claude-sonnet-4-5, claude-sonnet-4-6, claude-opus-4-6, …)."""

    def __init__(self, model: str, api_key_env: str = "ANTHROPIC_API_KEY"):
        super().__init__(model)
        self.supports_logprobs = False  # Claude doesn't support logprobs
        
        if not _HAS_CLAUDE:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable '{api_key_env}' not set.")
        
        self._client = _anthropic_module.Anthropic(api_key=api_key)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 4_000,
        temperature: float = 0.0,
        return_logprobs: bool = False,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> Dict[str, Any]:
        
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                
                text = resp.content[0].text.strip()
                tokens = resp.usage.input_tokens + resp.usage.output_tokens
                
                # Claude doesn't support logprobs
                logprobs_list = None
                
                return {
                    "text": text,
                    "tokens_used": tokens,
                    "logprobs": logprobs_list,
                }

            except Exception as e:
                print(f"    [Claude] Attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)

        return {"text": "", "tokens_used": 0, "logprobs": None}


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_client(
    model: str,
    openai_key_env: str = "OPENAI_API_KEY",
    gemini_key_env: str = "GEMINI_API_KEY",
    claude_key_env: str = "ANTHROPIC_API_KEY",
) -> BaseLLMClient:
    """
    Return the correct LLM client based on the model name prefix.

    Routing rules:
        gpt-* | o1-* | o3-* | o4-* | o5-*  →  OpenAI
        gemini-*                            →  Gemini
        claude-*                            →  Claude
        (unknown prefix)                    →  OpenAI (with warning)
    """
    m = model.lower()

    if any(m.startswith(p) for p in ("gpt-", "o1", "o3", "o4", "o5")):
        return OpenAIClient(model=model, api_key_env=openai_key_env)

    if m.startswith("gemini"):
        return GeminiClient(model=model, api_key_env=gemini_key_env)

    if m.startswith("claude"):
        return ClaudeClient(model=model, api_key_env=claude_key_env)

    # Fallback
    print(f"  [Warning] Unknown model prefix for '{model}'. Defaulting to OpenAI client.")
    return OpenAIClient(model=model, api_key_env=openai_key_env)