"""
End-to-end engine runs against FakeLLM and a scripted-failing mock
to verify that engine.run() reaches game_over with correct reason
and payload, and that FormatLimitExhausted from the runner is
caught at engine level
"""

import json
import random
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from game.clients import FakeLLMClient, FakeStrategy
from game.clients.base_client import LLMClient
from game.engine import GameEngine
from game.logger import EventLogger


def _read_events(log_path: Path):
    return [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _last_event(log_path: Path, event_type: str):
    matches = [r for r in _read_events(log_path) if r['event'] == event_type]
    assert matches, f"no {event_type} event in log"
    return matches[-1]


# raises ValidationError on every structured call to force FormatLimitExhaousted
class _AlwaysFailingClient(LLMClient):

    def __init__(self):
        super().__init__(model='broken', api_type='fake', token_log_path=Path('token_usage.txt'))

    def structured_call(self, messages, schema):
        try:
            schema.model_validate({'this_field_does_not_exist': True})
        except ValidationError as e:
            raise e
        raise AssertionError("unreachable")


def test_full_fake_run_terminates_with_max_rounds(tmp_path, make_config):
    config = make_config(n_agents=5, m_informed=2, max_rounds=2, share_cost=1.0)
    llm = FakeLLMClient(strategy=FakeStrategy.ALWAYS_SHARE, rng=random.Random(42))
    log_path = tmp_path / 'run.jsonl'

    with EventLogger(log_path) as logger:
        engine = GameEngine(config=config, llm=llm, logger=logger)
        engine.run()

    game_over = _last_event(log_path, 'game_over')
    payload = game_over['payload']

    assert payload['reason'] == 'max_rounds'
    assert payload['n_rounds'] == 2
    assert set(payload['final_scores'].keys()) == {f'agent_{i}' for i in range(5)}

    # no stuck_* fields on normal termination
    assert 'stuck_agent_id' not in payload


# check game_over emitting with reason='format_limit_exhausted' + stuck_* fields 
def test_format_limit_exhausted_stops_game_cleanly(tmp_path, make_config):
    config = make_config(n_agents=3, m_informed=1, max_rounds=10, max_retries=0)
    llm = _AlwaysFailingClient()
    log_path = tmp_path / 'run.jsonl'

    with EventLogger(log_path) as logger:
        engine = GameEngine(config=config, llm=llm, logger=logger)
        engine.run()  # must not raise coz exception is caught inside

    game_over = _last_event(log_path, 'game_over')
    payload = game_over['payload']

    assert payload['reason'] == 'format_limit_exhausted'
    assert payload['stuck_phase'] == 'answer'   # first phase where call happens
    assert payload['stuck_attempts'] == 1       # max_retries=0 - 1 attempt
    assert payload['stuck_agent_id'].startswith('agent_')

    # game stopped mid-round-1, before round_summary
    events = [r['event'] for r in _read_events(log_path)]
    assert 'format_limit_exhausted' in events
    assert events.count('round_start') == 1     # never started round 2
