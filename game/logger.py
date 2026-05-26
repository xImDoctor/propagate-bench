import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self, Literal

from .config import GameConfig

# schema versioning if we remove/add new fields to log
LOG_SCHEMA_VERSION = 1

# for models name sanitizing in logs (like llama/llama3)
# TODO: integrate after model field appears
def _sanitize(s: str) -> str:
    return s.replace('/', '_').replace(':', '_').replace(' ', '_')


EventType = Literal[
    'game_init',
    'system_prompt_set',
    'round_start',
    'llm_request',
    'llm_response',
    'answer',
    'score_update',
    'share_decision',
    'token_received',
    'round_summary',
    'game_over',
    'error',
]


class EventLogger:
    def __init__(self, log_path: Path):

        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._file = self.log_path.open('a', encoding='utf-8') # append mode
        self._current_round: int = 0

    @classmethod
    def from_config(cls, config: GameConfig, log_dir : Path = Path('logs/')) -> Self:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        name = f"log_{ts}_{config.template_version}_seed{config.seed}.jsonl"

        return cls(log_path=log_dir / name)
    
    def __enter__(self) -> Self:
        return self
    
    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()

    def set_round(self, round_num: int) -> None:
        self._current_round = round_num

    def log(
            self,
            event: EventType,
            payload: dict[str, Any],
            agent_id: str | None = None,
    ) -> None:
        record = {
            'v': LOG_SCHEMA_VERSION,
            'ts': datetime.now(timezone.utc).isoformat(),
            'round': self._current_round,
            'event': event,
            'agent_id': agent_id,
            'payload': payload,
        }
        self._file.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
        self._file.flush()