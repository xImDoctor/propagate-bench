"""
LLM client for Together.ai with structured output via JSON-schema format.
"""

import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel
from openai import OpenAI

from .base_client import LLMClient
from ..states import ChatMessage

DEFAULT_BASE_URL = 'https://api.together.xyz/v1'

T = TypeVar('T', bound=BaseModel)


class TogetherLLMClient(LLMClient):
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
        token_log_path: Path = Path('token_usage.txt'),
    ):
        super().__init__(model=model, api_type='together', token_log_path=token_log_path)

        key = api_key or os.getenv('TOGETHER_API_KEY')
        if not key:
            raise RuntimeError('TOGETHER_API_KEY is not set in env or not passed to client directly')


        self.client = OpenAI(
            api_key=key,
            base_url=base_url or DEFAULT_BASE_URL,
            timeout=request_timeout,
        )
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def structured_call(
        self,
        messages: list[ChatMessage],
        schema: type[T],
    ) -> T:
        
        kwargs = {
            'model': self.model,
            'messages': list(messages),
            'response_format': {
                'type': 'json_schema',
                'schema': schema.model_json_schema(),
            },
            'max_tokens': self.max_tokens,
        }

        if self.seed is not None:
            kwargs['seed'] = self.seed

        if self.temperature is not None:
            kwargs['temperature'] = self.temperature

        if self.top_p is not None:
            kwargs['top_p'] = self.top_p

        response = self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ''
        usage = response.usage

        pt = getattr(usage, 'prompt_tokens', 0) or 0
        ct = getattr(usage, 'completion_tokens', 0) or 0
        self._update_token_log(pt, ct)

        # validation error catches by llm_runner.py
        return schema.model_validate_json(content)
