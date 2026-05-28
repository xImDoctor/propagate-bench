import json
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from ..states import ChatMessage


class LLMClient(ABC):
    def __init__(self, model: str, api_type: str, token_log_path: Path = Path("token_usage.txt")):
        self.model = model
        self.api_type = api_type
        self.token_log_path = Path(token_log_path)

    @abstractmethod
    def structured_call(
        self,
        messages: list[ChatMessage],
        schema: type[BaseModel],
    ) -> BaseModel:
        """Sends messages, returns parsed+validated instance of `schema`.
        Raises on persistent parse/validation failure after retries"""

    # Writes token usage to log (doesn't calculate usage payment)
    def _update_token_log(self, prompt_tokens: int, completion_tokens: int) -> None:
        
        key = f"{self.api_type}:{self.model}"
        
        try:
            data = {}

            if self.token_log_path.exists():
                data = json.loads(self.token_log_path.read_text(encoding='utf-8'))

            entry = data.get(key, {'prompt_tokens': 0, 'completion_tokens': 0})
            entry['prompt_tokens'] += int(prompt_tokens or 0)
            entry['completion_tokens'] += int(completion_tokens or 0)
            data[key] = entry

            self.token_log_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        
        except Exception:
            pass # token accounting never breaks game run

