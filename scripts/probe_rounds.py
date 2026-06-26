"""Probe scenario: asks one agent how many rounds it expects then exit.

The engine is not started, only system prompt + one user question + a single 
structured call.

Usage:
    python scripts/probe_rounds.py configs/some.yaml
    python scripts/probe_rounds.py configs/some.yaml --seeds-file probes/seeds.txt
    python scripts/probe_rounds.py --api-type ollama --model qwen2.5:3b --seed 7

Not designed for api_type=fake (FakeLLM does not fabricate this schema).
"""

import sys
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from pydantic import BaseModel

from game.config import GameConfig
from game.clients import LLMClient, OllamaLLMClient, TogetherLLMClient
from game.prompt_builder import create_prompt_builder
from game.states import AgentState, ChatMessage


PROBE_QUESTION = (
    "Based on the description above, how many rounds do you expect this game to last? "
    "Respond strictly as JSON."
)


class RoundsExpectationResponse(BaseModel):
    expected_rounds: int
    reasoning: str = ''


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

    raise NotImplementedError(f"api_type={config.api_type!r} not supported by probe_rounds script")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Probe an agent for expected number of rounds.')

    p.add_argument('config', nargs='?', default=None, help='Path to YAML config (optional)')

    # if --seeds-file set, iterate, otherwise single run with --seed or config default
    p.add_argument('--seeds-file', type=str, default=None,
                   help='Path to txt with seeds, one per line')
    p.add_argument('--seed', type=int, default=None)

    p.add_argument('--model', type=str, default=None)
    p.add_argument('--api-type', type=str, default=None)

    p.add_argument('--output', type=str, default=None,
                   help='Output JSONL path. Default: probes/probe_rounds_<ts>.jsonl')

    return p.parse_args()


def build_config(args: argparse.Namespace, seed_override: int | None = None) -> GameConfig:
    overrides = {
        k: v for k, v in {
            'model': args.model,
            'api_type': args.api_type,
            'seed': seed_override if seed_override is not None else args.seed,
        }.items() if v is not None
    }

    if args.config:
        base = GameConfig.from_yaml(args.config).model_dump()
        return GameConfig(**{**base, **overrides})

    # probe-only defaults: n_agents/m_informed/share_cost only shape the system prompt text
    base = {'n_agents': 4, 'm_informed': 2, 'share_cost': 0.01}
    return GameConfig(**{**base, **overrides})


def load_seeds(args: argparse.Namespace) -> list[int | None]:
    # None means "use whatever seed the config resolves to" - single run
    if args.seeds_file:
        path = Path(args.seeds_file)
        return [int(line.strip()) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return [None]


def run_one(config: GameConfig) -> dict:
    # build a min context: one agent that knows the token so system prompt is fully populated
    llm = build_llm(config)
    rng = random.Random(config.seed)
    token = config.token or str(rng.randint(1000, 9999))

    prompts = create_prompt_builder(config, token)
    
    agent = AgentState(agent_id='agent_0', knows_token=True)
    all_ids = [f'agent_{i}' for i in range(config.n_agents)]

    system_prompt = prompts.build_system_prompt(agent, all_ids)

    messages: list[ChatMessage] = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': PROBE_QUESTION},
    ]

    record = {
        'ts': datetime.now(timezone.utc).isoformat(),
        'model': config.model,
        'api_type': config.api_type,
        'seed': config.seed,
        'expected_rounds': None,
        'reasoning': None,
        'error': None,
    }

    try:
        response = llm.structured_call(messages, RoundsExpectationResponse)
        record['expected_rounds'] = response.expected_rounds
        record['reasoning'] = response.reasoning
    except Exception as e:
        record['error'] = f'{type(e).__name__}: {e}'

    return record


def default_output_path() -> str:
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')
    return f'probes/probe_rounds_{ts}.jsonl'


def main():
    load_dotenv()
    args = parse_args()
    seeds = load_seeds(args)

    output = Path(args.output or default_output_path())
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w', encoding='utf-8') as f:
        for seed in seeds:
            config = build_config(args, seed_override=seed)
            record = run_one(config)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            print(f"seed={record['seed']} expected={record['expected_rounds']} err={record['error']}")

    print(f"\nResults written to {output}")


if __name__ == '__main__':
    main()
