import random
import pytest

from game.states import GameState


def _state(make_config, **overrides) -> GameState:
    cfg = make_config(**overrides)
    return GameState.initialize(cfg, random.Random(cfg.seed))


def test_initialize_agent_count(make_config):
    state = _state(make_config, n_agents=4)
    assert len(state.agents) == 4


def test_initialize_m_informed(make_config):
    state = _state(make_config, n_agents=4, m_informed=2)
    assert sum(a.knows_token for a in state.agents) == 2


def test_initialize_deterministic(make_config):
    cfg = make_config()

    s1 = GameState.initialize(cfg, random.Random(cfg.seed))
    s2 = GameState.initialize(cfg, random.Random(cfg.seed))
    
    assert [a.knows_token for a in s1.agents] == [a.knows_token for a in s2.agents]


def test_initialize_default_starting_capital(make_config):
    state = _state(make_config)
    assert all(agent.score == 0.0 for agent in state.agents)


def test_initialize_custom_starting_capital(make_config):
    state = _state(make_config, starting_capital=9.5)
    assert all(agent.score == 9.5 for agent in state.agents)


def test_distribute_score_equal_share(make_config):
    state = _state(make_config, n_agents=4)
    state.distribute_score(correct_count=2)
    assert all(a.score == 0.5 for a in state.agents)  # 2 / 4


def test_distribute_score_accumulates(make_config):
    state = _state(make_config, n_agents=2, m_informed=1)
    state.distribute_score(correct_count=2)  # +1.0 each
    state.distribute_score(correct_count=1)  # +0.5 each
    
    assert all(a.score == 1.5 for a in state.agents)


def test_apply_transfer_success(make_config):
    state = _state(make_config, n_agents=2, m_informed=1)
    teacher = state.knowing_agents()[0]
    student = state.unknowing_agents()[0]
    teacher.score = 5.0
    state.apply_transfer(teacher.agent_id, student.agent_id, cost_teacher=1.0)
    
    assert student.knows_token is True
    assert teacher.score == 4.0


def test_apply_transfer_student_cost(make_config):
    state = _state(make_config, n_agents=2, m_informed=1)
    teacher = state.knowing_agents()[0]
    student = state.unknowing_agents()[0]
    teacher.score = 5.0
    student.score = 3.0
    state.apply_transfer(teacher.agent_id, student.agent_id, cost_teacher=0.0, cost_student=2.0)
    
    assert student.knows_token is True
    assert student.score == 1.0


def test_apply_transfer_sender_unknown_raises(make_config):
    state = _state(make_config, n_agents=2, m_informed=1)
    student = state.unknowing_agents()[0]
    other = next(a for a in state.agents if a is not student)
    
    with pytest.raises(ValueError):
        state.apply_transfer(student.agent_id, other.agent_id, cost_teacher=1.0)


def test_apply_transfer_receiver_known_raises(make_config):
    state = _state(make_config, n_agents=3, m_informed=2)
    teachers = state.knowing_agents()

    with pytest.raises(ValueError):
        state.apply_transfer(teachers[0].agent_id, teachers[1].agent_id, cost_teacher=1.0)


def test_get_agent_missing_raises(make_config):
    state = _state(make_config)
    with pytest.raises(KeyError):
        state.get_agent('agent_does_not_exist')


def test_check_stop_all_know(make_config):
    state = _state(make_config, n_agents=2, m_informed=1)
    
    for a in state.agents:
        a.knows_token = True

    assert state.check_stop(max_rounds=100) is True
    assert state.game_over is True


def test_check_stop_max_rounds(make_config):
    state = _state(make_config, n_agents=3, m_informed=1)
    state.round = 10
    
    assert state.check_stop(max_rounds=10) is True


def test_check_stop_not_yet(make_config):
    state = _state(make_config, n_agents=3, m_informed=1)
    state.round = 1
    
    assert state.check_stop(max_rounds=10) is False
    assert state.game_over is False


def test_initialize_anon_display_name_equals_agent_id(make_config):
    cfg = make_config(n_agents=3, m_informed=1)  # display_names_file=None
    state = GameState.initialize(cfg, random.Random(42))

    for a in state.agents:
        assert a.display_name == a.agent_id
        assert a.agent_id.startswith('agent_')


def test_initialize_display_names_from_file_first_n(make_config, tmp_path):
    names_file = tmp_path / 'names.txt'
    names_file.write_text("Alice\nBob\nJohn\nStan\n", encoding='utf-8')

    cfg = make_config(n_agents=3, m_informed=1, display_names_file=names_file)
    state = GameState.initialize(cfg, random.Random(42))

    assert [a.display_name for a in state.agents] == ['Alice', 'Bob', 'John']
    assert [a.agent_id for a in state.agents] == ['agent_0', 'agent_1', 'agent_2'] # agent_id exists anyway


def test_initialize_display_names_too_few_raises(make_config, tmp_path):
    names_file = tmp_path / 'names.txt'
    names_file.write_text("Alice\nBob\n", encoding='utf-8')

    cfg = make_config(n_agents=3, m_informed=1, display_names_file=names_file)

    with pytest.raises(ValueError, match='at least 3'):
        GameState.initialize(cfg, random.Random(42))


def test_initialize_display_names_random_deterministic(make_config, tmp_path):
    names_file = tmp_path / 'names.txt'
    names_file.write_text("\n".join(["A", "B", "C", "D", "E"]), encoding='utf-8')

    cfg = make_config(n_agents=3, m_informed=1,
                      display_names_file=names_file, display_names_random=True)

    s1 = GameState.initialize(cfg, random.Random(42))
    s2 = GameState.initialize(cfg, random.Random(42))

    assert [a.display_name for a in s1.agents] == [a.display_name for a in s2.agents]


def test_initialize_display_names_random_differs_from_first_n(make_config, tmp_path):
    names_file = tmp_path / 'names.txt'
    names_file.write_text("\n".join([f"name_{i}" for i in range(20)]), encoding='utf-8')

    cfg_first = make_config(n_agents=5, m_informed=1, display_names_file=names_file)
    cfg_random = make_config(n_agents=5, m_informed=1,
                             display_names_file=names_file, display_names_random=True)

    first = [a.display_name for a in GameState.initialize(cfg_first, random.Random(42)).agents]
    rand = [a.display_name for a in GameState.initialize(cfg_random, random.Random(42)).agents]

    assert first == ['name_0', 'name_1', 'name_2', 'name_3', 'name_4']
    assert rand != first  # different with high probability
