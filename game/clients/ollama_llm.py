"""
Ollama LLM client with structured output via JSON-schema format.
"""

import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .base_client import LLMClient
from ..states import ChatMessage

DEFAULT_HOST = 'http://localhost:11434'

T = TypeVar('T', bound=BaseModel)


class OllamaLLMClient(LLMClient):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        request_timeout: float = 60.0,
        token_log_path: Path = Path('token_usage.txt'),
    ):
        super().__init__(model=model, api_type='ollama', token_log_path=token_log_path)

        from ollama import Client

        host = base_url or os.getenv('OLLAMA_HOST', DEFAULT_HOST)
        self.client = Client(host=host, timeout=request_timeout)
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p

    def structured_call(
        self,
        messages: list[ChatMessage],
        schema: type[T],
    ) -> T:
        
        options: dict = {}

        if self.seed is not None:
            options['seed'] = self.seed

        if self.temperature is not None:
            options['temperature'] = self.temperature

        if self.top_p is not None:
            options['top_p'] = self.top_p

        response = self.client.chat(
            model=self.model,
            messages=list(messages),
            format=schema.model_json_schema(),
            options=options,
        )

        content = response.message.content or ''
        pt = getattr(response, 'prompt_eval_count', 0) or 0
        ct = getattr(response, 'eval_count', 0) or 0
        self._update_token_log(pt, ct)

        # validation error catches by llm_runner.py
        return schema.model_validate_json(content)
        
