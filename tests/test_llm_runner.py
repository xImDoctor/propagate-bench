import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from game.clients.base_client import LLMClient
from game.llm_runner import call_with_retry, FormatLimitExhausted
from game.logger import EventLogger
from game.states import AgentState, ChatMessage


class _Toy(BaseModel):
    answer: str


class _ScriptedClient(LLMClient):
    """LLM mock driven by a script of (action, value) pairs

    action='return' - model_validate(value) and return.
    action='raise'  - raise value (exception instance).
    Last entry is repeated indefinitely once the script runs out.
    """

    def __init__(self, script):
        super().__init__(model='mock', api_type='fake', token_log_path=Path('token_usage.txt'))
        self.script = list(script)
        self.calls = 0

    def structured_call(self, messages, schema):
        i = min(self.calls, len(self.script) - 1)
        self.calls += 1
        action, value = self.script[i]
        if action == 'return':
            return schema.model_validate(value)
        if action == 'raise':
            raise value
        raise RuntimeError(f"unknown scripted action: {action}")


@pytest.fixture
def agent():
    return AgentState(agent_id='agent_0', knows_token=False, context=[])


@pytest.fixture
def logger(tmp_path):
    log_path = tmp_path / 'run.jsonl'
    with EventLogger(log_path) as lg:
        yield lg


def _validation_error(): # build ValidationError by validating bad data with _Toy
    try:
        _Toy.model_validate({'wrong_field': 'x'})
    except ValidationError as e:
        return e
    raise AssertionError("expected ValidationError, got none")


def _read_events(log_path: Path):
    return [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines() if line.strip()]


def test_first_attempt_succeeds(agent, logger):
    client = _ScriptedClient([('return', {'answer': '42'})])

    response = call_with_retry(agent, 'what?', _Toy, client, logger, phase='answer', max_retries=2)

    assert response is not None
    assert response.answer == '42'
    assert client.calls == 1

    # context: user prompt + assistant response
    assert [m['role'] for m in agent.context] == ['user', 'assistant']
    assert agent.context[0]['content'] == 'what?'


# check context pops when validation failed
def test_retry_pops_invalid_user_keeps_context_clean(agent, logger):
    """Final context must be [user, assistant], and not [user, user, assistant]"""
    client = _ScriptedClient([
        ('raise', _validation_error()),
        ('return', {'answer': 'ok'}),
    ])

    response = call_with_retry(agent, 'what?', _Toy, client, logger, phase='answer', max_retries=2)

    assert response is not None
    assert response.answer == 'ok'
    assert client.calls == 2
    assert [m['role'] for m in agent.context] == ['user', 'assistant']


# ValidationError raises FormatLimitExhausted with correct fields
def test_format_limit_raises_after_exhausting_retries(agent, logger):
    client = _ScriptedClient([('raise', _validation_error())])

    with pytest.raises(FormatLimitExhausted) as exc_info:
        call_with_retry(agent, 'what?', _Toy, client, logger, phase='share', max_retries=2)

    err = exc_info.value
    assert err.agent_id == 'agent_0'
    assert err.phase == 'share'
    assert err.attempts == 3  # max_retries + 1
    assert client.calls == 3

    # all user-msgs popped, context is empty
    assert agent.context == []


def test_transport_error_returns_none_no_raise(agent, logger):
    client = _ScriptedClient([('raise', RuntimeError('connection refused'))])

    response = call_with_retry(agent, 'what?', _Toy, client, logger, phase='answer', max_retries=2)

    assert response is None
    assert client.calls == 1    # no retry on transport errors
    assert agent.context == []


# test success with zero retries
def test_max_retries_zero_means_one_attempt(agent, logger):
    client = _ScriptedClient([('raise', _validation_error())])

    with pytest.raises(FormatLimitExhausted) as exc_info:
        call_with_retry(agent, 'what?', _Toy, client, logger, phase='answer', max_retries=0)

    assert exc_info.value.attempts == 1
    assert client.calls == 1

# format_warning per failed attmpt + format_limit_exhausted on exit
def test_format_warning_and_limit_events_logged(agent, tmp_path):
    log_path = tmp_path / 'run.jsonl'
    client = _ScriptedClient([('raise', _validation_error())])

    with EventLogger(log_path) as lg:
        with pytest.raises(FormatLimitExhausted):
            call_with_retry(agent, 'what?', _Toy, client, lg, phase='answer', max_retries=1)

    events = [r['event'] for r in _read_events(log_path)]
    assert events.count('format_warning') == 2          # max_retries+1 attempts, all failed
    assert events.count('format_limit_exhausted') == 1


def test_extra_request_payload_merged_into_llm_request(agent, tmp_path):
    
    log_path = tmp_path / 'run.jsonl'
    client = _ScriptedClient([('return', {'answer': 'ok'})])

    with EventLogger(log_path) as lg:
        call_with_retry(
            agent, 'q', _Toy, client, lg, phase='share', max_retries=0,
            extra_request_payload={'available': ['agent_1', 'agent_2']},
        )

    req = next(r for r in _read_events(log_path) if r['event'] == 'llm_request')

    assert req['payload']['available'] == ['agent_1', 'agent_2']
    assert req['payload']['phase'] == 'share'  # base fields saved
