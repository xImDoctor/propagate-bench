import pytest

from game.config import GameConfig


def test_valid_config(make_config):
    cfg = make_config()

    assert cfg.n_agents == 4
    assert cfg.matcher in ('random_choice', 'first_come')


def test_frozen(make_config):
    cfg = make_config()
    with pytest.raises(Exception):  # pydantic FrozenInstanceError / ValidationError
        cfg.seed = 99


def test_n_agents_too_small(make_config):
    with pytest.raises(ValueError):
        make_config(n_agents=1, m_informed=1)


def test_m_informed_zero(make_config):
    with pytest.raises(ValueError):
        make_config(n_agents=4, m_informed=0)


def test_m_informed_equals_n(make_config):
    with pytest.raises(ValueError):
        make_config(n_agents=4, m_informed=4)


def test_share_cost_negative(make_config):
    with pytest.raises(ValueError):
        make_config(share_cost=-1.0)


def test_reserved_payment_mode_not_implemented(make_config):
    with pytest.raises(NotImplementedError):
        make_config(payment_mode='split')


def test_extra_field_forbidden(make_config):
    with pytest.raises(ValueError):
        make_config(nonexistent_field=123)


def test_from_yaml(tmp_path):
    yaml_text = (
        "n_agents: 5\n"
        "m_informed: 2\n"
        "share_cost: 1.5\n"
        "max_rounds: 7\n"
        "seed: 99\n"
        "model: fake\n"
        "api_type: fake\n"
        "matcher: random_choice\n"
    )
    path = tmp_path / 'exp.yaml'
    path.write_text(yaml_text, encoding='utf-8')

    cfg = GameConfig.from_yaml(path)

    assert cfg.n_agents == 5
    assert cfg.m_informed == 2
    assert cfg.share_cost == 1.5
    assert cfg.seed == 99
    assert cfg.matcher == 'random_choice'
