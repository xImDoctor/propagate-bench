"""
Fake LLM (coded algorithm) for base pipeline testing
"""

import random
import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from .llm_client import LLMClient
from .states import ChatMessage


TOKEN_MARKER_RE = re.compile(r'\[TOKEN\](\d+)\[/TOKEN\]')


class FakeStrategy(str, Enum):
    ALWAYS_SHARE = 'always_share'
    NEVER_SHARE = 'never_share'
    RANDOM_SHARE = 'random_share'


class FakeLLMClient(LLMClient):
    def __init__(
        self,
        strategy: FakeStrategy = FakeStrategy.ALWAYS_SHARE,
        share_probability: float = 0.5,   # for RANDOM_SHARE
        rng: random.Random | None = None,
        token_log_path: Path = Path('token_usage.txt'),
    ):
        super().__init__(model='fake', api_type='fake', token_log_path=token_log_path)
        
        self.strategy = strategy
        self.share_probability = share_probability
        self.rng = rng or random.Random()

    def structured_call(
        self,
        messages: list[ChatMessage],
        schema: type[BaseModel],
    ) -> BaseModel:
        response = self._fabricate(messages, schema)
        result = schema.model_validate(response)

        pt = sum(len(m['content']) for m in messages) // 4
        ct = len(str(response)) // 4

        self._update_token_log(pt, ct)

        return result

    def _fabricate(self, messages: list[ChatMessage], schema: type[BaseModel]) -> dict:
        known_token = self._extract_token(messages)
        field_names = set(schema.model_fields.keys())

        if field_names == {'answer'}:
            if known_token is not None:
                return {'answer': known_token}
            
            return {'answer': '<unknown>'} # no reason to guess token in fake
            # return {'answer': known_token or str(self.rng.randint(1000, 9999))}

        if field_names >= {'share', 'target', 'reasoning'}:
            return self._decide_share(messages)

        raise ValueError(f"FakeLLMClient does not know how to fabricate response for schema: {schema.__name__}")

    def _extract_token(self, messages: list[ChatMessage]) -> str | None:
        # search token in system prompt (for agents that have it)
        for message in messages:
            if message['role'] == 'system':
                match = TOKEN_MARKER_RE.search(message['content'])
                if match:
                    return match.group(1)
                
        return None

    def _decide_share(self, messages: list[ChatMessage]) -> dict:
        # target - just first unknowing agent from latest user prompt (simplification)
        if self.strategy == FakeStrategy.NEVER_SHARE:
            return {'share': False, 'target': None, 'reasoning': 'fake: never_share'}
        
        if self.strategy == FakeStrategy.RANDOM_SHARE and self.rng.random() >= self.share_probability:
            return {'share': False, 'target': None, 'reasoning': 'fake: random rejected'}
        
        # ALWAYS_SHARE or RANDOM_SHARE passed so take first unknown agent
        target = self._first_unknowing(messages)
        return {'share': True, 'target': target, 'reasoning': 'fake: strategy decided to share'}

    def _first_unknowing(self, messages: list[ChatMessage]) -> str | None:
        for message in reversed(messages):
            if message['role'] == 'user':
                match = re.search(r"\bagent_\d+\b", message['content'])
                if match:
                    return match.group(0)
        return None
