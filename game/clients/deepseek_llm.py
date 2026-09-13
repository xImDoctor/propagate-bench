"""
LLM client for DeepSeek native API (https://api.deepseek.com/v1).

OpenAI-compatible endpoint that supports two models:
    deepseek-flash          # non-reasoning
    deepseek-v4-pro         # reasoning (always-on: returns reasoning_content)

Differences from TogetherLLMClient:
- Usage object exposes prompt_cache_hit_tokens / prompt_cache_miss_tokens
  and completion_tokens_details.reasoning_tokens – forwarded to
  _update_token_log so calc_costs.py can bill cache tiers separately.
- Optional `verify` param for the underlying httpx client (default True).
"""

import os
from pathlib import Path
from typing import TypeVar

import httpx
from pydantic import BaseModel

from .base_client import LLMClient
from ..states import ChatMessage

DEFAULT_BASE_URL = 'https://api.deepseek.com/v1'

T = TypeVar('T', bound=BaseModel)


class DeepSeekLLMClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        request_timeout: float = 60.0,
        max_tokens: int = 4112,
        verify: bool | str = True,
        token_log_path: Path = Path('token_usage.txt'),
    ):
        super().__init__(model=model, api_type='deepseek', token_log_path=token_log_path)

        from openai import OpenAI  # lazy

        key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not key:
            raise RuntimeError('DEEPSEEK_API_KEY is not set in env or not passed to client directly')

        self.client = OpenAI(
            api_key=key,
            base_url=base_url or DEFAULT_BASE_URL,
            timeout=request_timeout,
            http_client=httpx.Client(verify=verify, timeout=request_timeout),
        )
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def _extract_usage(self, response) -> tuple[int, int, int, int]:
        """Pull (prompt, completion, cache_hit, reasoning) counts from usage."""
        usage = response.usage
        pt = getattr(usage, 'prompt_tokens', 0) or 0
        ct = getattr(usage, 'completion_tokens', 0) or 0

        cache_hit = getattr(usage, 'prompt_cache_hit_tokens', None)
        if cache_hit is None:
            details = getattr(usage, 'prompt_tokens_details', None)
            cache_hit = getattr(details, 'cached_tokens', 0) if details else 0
        cache_hit = int(cache_hit or 0)

        details = getattr(usage, 'completion_tokens_details', None)
        reasoning = int(getattr(details, 'reasoning_tokens', 0) or 0) if details else 0

        return pt, ct, cache_hit, reasoning

    def _build_kwargs(self, messages: list[ChatMessage], schema: type[T]) -> dict:
        kwargs = {
            'model': self.model,
            'messages': list(messages),
            'response_format': {'type': 'json_object'},
            'max_tokens': self.max_tokens,
        }
        if self.seed is not None:
            kwargs['seed'] = self.seed
        if self.temperature is not None:
            kwargs['temperature'] = self.temperature
        if self.top_p is not None:
            kwargs['top_p'] = self.top_p
        return kwargs

    def structured_call(
        self,
        messages: list[ChatMessage],
        schema: type[T],
    ) -> T:
        response = self.client.chat.completions.create(**self._build_kwargs(messages, schema))
        content = response.choices[0].message.content or ''

        pt, ct, cache_hit, reasoning = self._extract_usage(response)
        self._update_token_log(pt, ct, cache_hit_tokens=cache_hit, reasoning_tokens=reasoning)

        return schema.model_validate_json(content)

    def structured_call_with_reasoning(
        self,
        messages: list[ChatMessage],
        schema: type[T],
    ) -> tuple[T, str | None]:
        """DeepSeek reasoning-capable models always emit reasoning_content –
        this method just captures it. For non-reasoning models the second
        slot is None.
        
        Note: token spend is identical between structured_call
        and this method on reasoning-capable models.
        """
        response = self.client.chat.completions.create(**self._build_kwargs(messages, schema))

        msg = response.choices[0].message
        content = msg.content or ''

        reasoning_text = getattr(msg, 'reasoning_content', None)
        if reasoning_text is None:
            extra = getattr(msg, 'model_extra', None) or {}
            reasoning_text = extra.get('reasoning_content')

        pt, ct, cache_hit, reasoning = self._extract_usage(response)
        self._update_token_log(pt, ct, cache_hit_tokens=cache_hit, reasoning_tokens=reasoning)

        return schema.model_validate_json(content), reasoning_text
