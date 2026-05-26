from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from .states import ChatMessage


class LLMClient(ABC):
    def __init__(self, model: str, api_type: str, token_log_path: Path = Path("token_usage.txt")):
        self.model = model
        self.api_type = api_type
        self.token_log_path = token_log_path

    @abstractmethod
    def structured_call(
        self,
        messages: list[ChatMessage],
        schema: type[BaseModel],
    ) -> BaseModel:
        """Sends messages, returns parsed+validated instance of `schema`.
        Raises on persistent parse/validation failure after retries"""

    # TODO: add from calc_costs.py
    def _update_token_log(self, prompt_tokens: int, completion_tokens: int) -> None:
        pass
