import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from ..states import ChatMessage

T = TypeVar('T', bound=BaseModel)


class LLMClient(ABC):
    def __init__(self, model: str, api_type: str, token_log_path: Path = Path("token_usage.txt")):
        self.model = model
        self.api_type = api_type
        self.token_log_path = Path(token_log_path)

    @abstractmethod
    def structured_call(self, messages: list[ChatMessage], schema: type[T]) -> T:
        """Sends messages, returns parsed+validated instance of schema.

        Raises ValidationError/JSONDecodeError on parse failure, other
        exceptions on transport/server errors. 
        
        No internal retry, it is handled by game.llm_runner.call_with_retry.
        """

    def _update_token_log(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_hit_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> None:
        """Accumulate token usage per (api_type, model) into the JSON log.

        cache_hit_tokens and reasoning_tokens default to 0 for backends that
        don't break them out (Together, Ollama, Fake).
        
        DeepSeek forwards values so calc_costs.py can bill cache tiers separately.
        cache_miss is not stored because it's derived as prompt_tokens - 
        prompt_cache_hit_tokens.
        """
        key = f"{self.api_type}:{self.model}"

        try:
            data = {}

            if self.token_log_path.exists():
                data = json.loads(self.token_log_path.read_text(encoding='utf-8'))

            entry = data.get(key, {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'prompt_cache_hit_tokens': 0,
                'completion_reasoning_tokens': 0,
            })
            entry.setdefault('prompt_cache_hit_tokens', 0)
            entry.setdefault('completion_reasoning_tokens', 0)

            entry['prompt_tokens']                += int(prompt_tokens or 0)
            entry['completion_tokens']            += int(completion_tokens or 0)
            entry['prompt_cache_hit_tokens']      += int(cache_hit_tokens or 0)
            entry['completion_reasoning_tokens']  += int(reasoning_tokens or 0)
            data[key] = entry

            self.token_log_path.write_text(json.dumps(data, indent=2), encoding='utf-8')

        except Exception:
            pass  # token accounting never breaks game run


