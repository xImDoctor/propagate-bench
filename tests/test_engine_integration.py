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
    llm = FakeLLMClient(strategy=FakeStrategy.NEVER_SHARE, rng=random.Random(42))
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


def test_anonymous_matcher_choice_does_not_affect_result(tmp_path, make_config):
    """Check if anonymous results are the same in every config.matcher."""
    log_path = tmp_path / 'run.jsonl'

    cfg_rc = make_config(n_agents=4, m_informed=2, max_rounds=2, matcher='random_choice')
    cfg_fc = make_config(n_agents=4, m_informed=2, max_rounds=2, matcher='first_come')

    def _run(cfg):
        with EventLogger(tmp_path / f'r_{cfg.matcher}.jsonl') as lg:
            llm = FakeLLMClient(strategy=FakeStrategy.ALWAYS_SHARE, rng=random.Random(42))
            eng = GameEngine(config=cfg, llm=llm, logger=lg)
            eng.run()
            return {a.agent_id: a.knows_token for a in eng.game_state.agents}

    assert _run(cfg_rc) == _run(cfg_fc)


def test_anonymous_share_prompt_has_no_target_list(tmp_path, make_config):
    """Check if anon-mode has no name list"""
    cfg = make_config(n_agents=4, m_informed=2, max_rounds=1)
    log_path = tmp_path / 'run.jsonl'

    with EventLogger(log_path) as lg:
        llm = FakeLLMClient(strategy=FakeStrategy.ALWAYS_SHARE, rng=random.Random(42))
        eng = GameEngine(config=cfg, llm=llm, logger=lg)
        eng.run()

    events = _read_events(log_path)
    share_requests = [
        r for r in events
        if r['event'] == 'llm_request' and r['payload'].get('phase') == 'share'
    ]

    assert share_requests, "no share llm_request found"
    for r in share_requests:
        # available must be empty or none
        assert not r['payload'].get('available')
        # no agent_ids in the prompt text
        text = r['payload']['message_appended']['content']
        assert 'agent_0' not in text and 'agent_1' not in text


def test_student_initiated_anon_transfer(tmp_path, make_config):
    """N=2, K=1, ALWAYS_SHARE fake, student_pays.
    After round 1: student paid share_cost, teacher's score unchanged (only got round reward),
    both know the token, no forced_teach_notice in teacher's context (student_pays doesn't emit it)"""

    cfg = make_config(
        n_agents=2, m_informed=1,
        initiation_mode='student_only', payment_mode='student_pays',
        share_cost=1.0, max_rounds=1,
    )
    log_path = tmp_path / 'run.jsonl'
    
    with EventLogger(log_path) as lg:
        llm = FakeLLMClient(strategy=FakeStrategy.ALWAYS_SHARE, rng=random.Random(42))
        eng = GameEngine(config=cfg, llm=llm, logger=lg)
        eng.run()

    state = eng.game_state

    assert all(a.knows_token for a in state.agents)
    assert sum(a.score for a in state.agents) == pytest.approx(1.0)
