import random

from game.fake_llm import FakeLLMClient, FakeStrategy
from game.prompt_builder import AnswerResponse, ShareResponse


def _knowing_messages(token: str) -> list[dict]:
    return [
        {"role": "system", "content": f"You are agent_0. [TOKEN]{token}[/TOKEN] respond as JSON."},
        {"role": "user", "content": "What is the token?"},
    ]


def _unknowing_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are agent_1. You do not know the token. respond as JSON."},
        {"role": "user", "content": "What is the token?"},
    ]


def test_knowing_answers_token():
    llm = FakeLLMClient(rng=random.Random(0))
    response = llm.structured_call(_knowing_messages("2824"), AnswerResponse)
    assert isinstance(response, AnswerResponse)
    assert response.answer == "2824"


def test_unknowing_does_not_reveal_token():
    token = "2824"
    llm = FakeLLMClient(rng=random.Random(42))
    response = llm.structured_call(_unknowing_messages(), AnswerResponse)
    assert response.answer != token
    assert not response.answer.isdigit()


def test_always_share_decides_true():
    llm = FakeLLMClient(strategy=FakeStrategy.ALWAYS_SHARE, rng=random.Random(0))
    messages = [
        {"role": "system", "content": "You are agent_0. [TOKEN]2824[/TOKEN]"},
        {"role": "user", "content": "Unknowing agents: agent_1, agent_2. Share?"},
    ]
    response = llm.structured_call(messages, ShareResponse)
    assert isinstance(response, ShareResponse)
    assert response.share is True
    assert response.target == "agent_1"


def test_never_share_decides_false():
    llm = FakeLLMClient(strategy=FakeStrategy.NEVER_SHARE, rng=random.Random(0))
    messages = [
        {"role": "system", "content": "You are agent_0. [TOKEN]2824[/TOKEN]"},
        {"role": "user", "content": "Unknowing agents: agent_1. Share?"},
    ]
    response = llm.structured_call(messages, ShareResponse)
    assert response.share is False
