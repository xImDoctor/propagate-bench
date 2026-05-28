import json

from game.logger import EventLogger


def _read_records(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def test_event_field_present(tmp_path):
    log_path = tmp_path / 'run.jsonl'
    with EventLogger(log_path) as logger:
        logger.log('game_init', {'foo': 'bar'})
        logger.log('answer', {'answer': '1234', 'is_correct': True}, agent_id='agent_0')

    records = _read_records(log_path)
    
    assert len(records) == 2
    assert all('event' in r for r in records)
    assert records[0]['event'] == 'game_init'
    assert records[1]['event'] == 'answer'


def test_envelope_fields(tmp_path):
    log_path = tmp_path / 'run.jsonl'
    with EventLogger(log_path) as logger:
        logger.log('round_start', {})

    rec = _read_records(log_path)[0]
    
    for key in ('v', 'ts', 'round', 'event', 'agent_id', 'payload'):
        assert key in rec


def test_set_round_reflected(tmp_path):
    log_path = tmp_path / 'run.jsonl'
    
    with EventLogger(log_path) as logger:
        logger.set_round(3)
        logger.log('answer', {'answer': 'x', 'is_correct': False}, agent_id='agent_1')

    rec = _read_records(log_path)[0]
    
    assert rec['round'] == 3
    assert rec['agent_id'] == 'agent_1'


def test_valid_jsonl(tmp_path):
    log_path = tmp_path / 'run.jsonl'
    
    with EventLogger(log_path) as logger:
        for i in range(5):
            logger.set_round(i)
            logger.log('score_update', {'correct_count': i})

    records = _read_records(log_path)
    assert len(records) == 5
