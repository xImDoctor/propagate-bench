"""Probe scenario: ask a single informed agent if it wants to share then exit.

Measures share=true probability in round 1 across varying N, M, P.
There are:
N - number of agents in the game (n_agents)
M - number of knowing agents (m_informed)
P - price of share/fee

The engine is not started - we hand-build the agent context (system, answer,
fake assistant answer with the token, share question) and make one structured
LLM call per combination. 

Implemented for anonymous mode only.

Single combination:
    python scripts/probe_share.py --n-agents 4 --m-informed 2 --share-cost 0.01 \\
        --model qwen2.5:3b --api-type ollama --seed 42

Grid sweep:
    python scripts/probe_share.py --grid-file configs/probe_share_grid.yaml

Grid YAML supports either explicit m_informed array or m_ratio (then m = round(n * ratio)).
Output is JSONL appended line by line so a Ctrl+C does not lose data.
"""

import sys
import argparse
import json
import itertools
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Literal
import yaml
from dotenv import load_dotenv

from game.config import GameConfig
from game.clients import LLMClient, OllamaLLMClient, TogetherLLMClient
from game.prompt_builder import ShareResponse, create_prompt_builder
from game.states import AgentState, ChatMessage, RoundResult


STUB_TOKEN = '1234'


def build_llm(config: GameConfig) -> LLMClient:
    
    if config.api_type == 'ollama':
        return OllamaLLMClient(
            model=config.model,
            seed=config.seed,
            temperature=config.temperature,
            top_p=config.top_p,
            request_timeout=config.request_timeout,
        )

    if config.api_type == 'together':
        return TogetherLLMClient(
            model=config.model,
            seed=config.seed,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            request_timeout=config.request_timeout,
        )

    raise NotImplementedError(f"api_type={config.api_type!r} not supported by probe_share")


def stub_config(n_agents: int, m_informed: int, share_cost: float, seed: int,
                model: str, api_type: Literal['ollama', 'together', 'fake']) -> GameConfig:
    
    # display_names_file is left None, anon mode is hardcoded for this script
    return GameConfig(
        n_agents=n_agents,
        m_informed=m_informed,
        share_cost=share_cost,
        starting_capital=0.0,
        max_rounds=1,
        seed=seed,
        model=model,
        api_type=api_type,
        template_version='v1_baseline',
        matcher='first_come',
    )


def run_one(config: GameConfig) -> dict:
    """Single share-probe call. Returns a flat record for JSONL"""

    prompts = create_prompt_builder(config, STUB_TOKEN)
    agent = AgentState(
        agent_id='agent_0',
        knows_token=True,
        score=config.starting_capital + config.m_informed / config.n_agents,
    )

    all_ids = [f'agent_{i}' for i in range(config.n_agents)]

    round_result = RoundResult(
        round_num=1,
        answers={},
        correct_answers={},
        scores_after={},
        correct_count=config.m_informed,
    )

    system_prompt = prompts.build_system_prompt(agent, all_ids)
    answer_prompt = prompts.build_answer_prompt(agent, round_num=1)
    share_prompt = prompts.build_share_prompt(agent, round_result, unknowing_agents=[])

    messages: list[ChatMessage] = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': answer_prompt},
        {'role': 'assistant', 'content': json.dumps({'answer': STUB_TOKEN})},
        {'role': 'user', 'content': share_prompt},
    ]

    record = {
        'ts': datetime.now(timezone.utc).isoformat(),
        'model': config.model,
        'api_type': config.api_type,
        'n_agents': config.n_agents,
        'm_informed': config.m_informed,
        'share_cost': config.share_cost,
        'seed': config.seed,
        'share': None,
        'reasoning': None,
        'error': None,
    }

    try:
        llm = build_llm(config)
        response = llm.structured_call(messages, ShareResponse)
        record['share'] = response.share
        record['reasoning'] = response.reasoning
    except Exception as e:
        record['error'] = f'{type(e).__name__}: {e}'

    return record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Probe share probability for informed agents in round 1.')

    p.add_argument('--grid-file', type=str, default=None,
                   help='YAML with arrays of n_agents/m_informed/share_cost/seeds for a sweep')

    # single-call args (ignored if --grid-file is set)
    p.add_argument('--n-agents', type=int, default=None)
    p.add_argument('--m-informed', type=int, default=None)
    p.add_argument('--share-cost', type=float, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', type=str, default=None)
    p.add_argument('--api-type', type=str, default=None)

    p.add_argument('--output', type=str, default=None,
                   help='Output JSONL path. Default: probes/probe_share_<ts>.jsonl')

    return p.parse_args()


def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product of n_agents x m_informed (or m_ratio) x share_cost x seeds."""
    n_list = grid['n_agents']
    p_list = grid['share_cost']
    seed_list = grid['seeds']
    model = grid['model']
    api_type = grid['api_type']

    if 'm_ratio' in grid:
        ratio = grid['m_ratio']
        m_by_n = {n: max(1, min(n - 1, round(n * ratio))) for n in n_list}
        m_iter = lambda n: [m_by_n[n]]
    else:
        m_list = grid['m_informed']
        m_iter = lambda n: m_list

    combos: list[dict] = []
    for n in n_list:
        for m in m_iter(n):
            if not 0 < m < n:
                continue
            for cost, seed in itertools.product(p_list, seed_list):
                combos.append({
                    'n_agents': n, 'm_informed': m, 'share_cost': cost,
                    'seed': seed, 'model': model, 'api_type': api_type,
                })
    return combos


def default_output_path() -> str:
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')
    return f'probes/probe_share_{ts}.jsonl'


def main():

    load_dotenv()
    args = parse_args()

    if args.grid_file:
        grid = yaml.safe_load(Path(args.grid_file).read_text(encoding='utf-8'))
        combos = expand_grid(grid)
    
    else:
        required = (args.n_agents, args.m_informed, args.share_cost, args.model, args.api_type)

        if any(v is None for v in required):
            raise SystemExit('Single mode needs --n-agents, --m-informed, --share-cost, --model, --api-type')

        combos = [{
            'n_agents': args.n_agents, 'm_informed': args.m_informed,
            'share_cost': args.share_cost, 'seed': args.seed,
            'model': args.model, 'api_type': args.api_type,
        }]

    output = Path(args.output or default_output_path())
    output.parent.mkdir(parents=True, exist_ok=True)
    total = len(combos)

    with open(output, 'w', encoding='utf-8') as f:
        for i, c in enumerate(combos, 1):
            config = stub_config(**c)
            record = run_one(config)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            print(
                f'[{i}/{total}] n={record["n_agents"]:>3} m={record["m_informed"]:>3} '
                f'p={record["share_cost"]:<6} seed={record["seed"]:>3} '
                f'share={record["share"]} err={record["error"]}'
            )

    print(f'\nResults written to {output}')


if __name__ == '__main__':
    main()
