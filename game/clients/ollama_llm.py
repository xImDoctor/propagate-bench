"""
Ollama LLM client with structured output via JSON-schema format
"""

import os
from pathlib import Path

from pydantic import BaseModel, ValidationError

from .base_client import LLMClient
from ..states import ChatMessage

DEFAULT_HOST = 'http://localhost:11434'


class OllamaLLMClient(LLMClient):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_retries: int = 1,
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
        self.max_retries = max_retries

    def structured_call(
        self,
        messages: list[ChatMessage],
        schema: type[BaseModel],
    ) -> BaseModel:
        
        options: dict = {}

        if self.seed is not None:
            options['seed'] = self.seed

        if self.temperature is not None:
            options['temperature'] = self.temperature

        if self.top_p is not None:
            options['top_p'] = self.top_p

        fmt = schema.model_json_schema()
        call_messages = list(messages)

        last_err: Exception | None = None

        for _ in range(self.max_retries + 1):
            response = self.client.chat(
                model=self.model,
                messages=call_messages,
                format=fmt,
                options=options,
            )

            content = response.message.content or ''

            pt = getattr(response, 'prompt_eval_count', 0) or 0
            ct = getattr(response, 'eval_count', 0) or 0
            self._update_token_log(pt, ct)

            try:
                return schema.model_validate_json(content)
            except ValidationError as e:
                last_err = e
                
                # if json is not valid, show incorrect msg to model and ask to fix
                call_messages = list(messages) + [
                    {'role': 'assistant', 'content': content},
                    {'role': 'user', 'content': (
                        'Your previous response did not match the required JSON schema. '
                        'Respond with ONLY valid JSON matching the schema, no extra text.'
                    )},
                ]

        raise last_err  # type: ignore[misc]
