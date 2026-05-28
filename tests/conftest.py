import pytest

from game.config import GameConfig


@pytest.fixture
def make_config():
    """Factory for a valid GameConfig with sensible defaults"""
    
    def _make(**overrides) -> GameConfig:
        base = dict(
            n_agents=4,
            m_informed=2,
            share_cost=1.0,
            max_rounds=10,
            seed=42,
            model='fake',
            api_type='fake',
        )
        base.update(overrides)
        return GameConfig(**base)
    return _make
