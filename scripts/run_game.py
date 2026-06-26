import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

import random

# add root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich.traceback import install as install_rich_traceback

from game.config import GameConfig
from game.engine import GameEngine
from game.clients import LLMClient, FakeLLMClient, FakeStrategy, OllamaLLMClient, TogetherLLMClient
from game.logger import EventLogger

install_rich_traceback(show_locals=False)
console = Console()


def build_llm(config: GameConfig, rng: random.Random, strategy: FakeStrategy) -> LLMClient:

    if config.api_type == 'fake':
        return FakeLLMClient(strategy=strategy, rng=rng)

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

    raise NotImplementedError(f"api_type={config.api_type!r} not implemented to run_game yet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run an agent knowledge-sharing game.')

    p.add_argument('config', nargs='?', default=None,
                   help='Path to experiment YAML config (optional, inline args override it)')

    # None as not received (to work with YAML)
    p.add_argument('--n-agents', type=int, default=None)
    p.add_argument('--m-informed', type=int, default=None)
    p.add_argument('--share-cost', type=float, default=None)
    p.add_argument('--starting-capital', type=float, default=None)
    p.add_argument('--max-rounds', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)

    p.add_argument('--model', type=str, default=None)
    p.add_argument('--api-type', type=str, default=None)
    p.add_argument('--temperature', type=float, default=None)
    p.add_argument('--top-p', type=float, default=None)
    p.add_argument('--request-timeout', type=float, default=None)

    p.add_argument('--template-version', type=str, default=None)
    p.add_argument('--matcher', type=str, default=None)

    # fake client param, not config
    p.add_argument(
        '--fake-strategy',
        type=str,
        default='always_share',
        choices=[s.value for s in FakeStrategy],
        help='FakeLLMClient behavior through the communication phase',
    )

    return p.parse_args()


def build_config(args: argparse.Namespace) -> GameConfig:
    
    # if args parsed (override config)
    overrides = {
        k: v for k, v in {
            'n_agents': args.n_agents,
            'm_informed': args.m_informed,
            'share_cost': args.share_cost,
            'starting_capital': args.starting_capital,
            'max_rounds': args.max_rounds,
            'seed': args.seed,
            'model': args.model,
            'api_type': args.api_type,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'request_timeout': args.request_timeout,
            'template_version': args.template_version,
            'matcher': args.matcher,
        }.items() if v is not None
    }

    if args.config:
        base = GameConfig.from_yaml(args.config).model_dump()
        return GameConfig(**{**base, **overrides})

    return GameConfig(**overrides)


def print_summary(engine: GameEngine, log_path: Path) -> None:

    table = Table(title='Game final state')
    table.add_column('agent_id')
    table.add_column('knows_token', justify='center')
    table.add_column('score', justify='right')

    for agent in engine.game_state.agents:
        table.add_row(
            agent.agent_id,
            'YES' if agent.knows_token else 'NO',
            f"{agent.score:.2f}",
        )

    console.print(table)
    console.print(f"Game token was: [bold]{engine.token}[/bold]")
    console.print(f"Rounds played: [bold]{engine.game_state.round}[/bold]")
    console.print(f"Log saved to [cyan]{log_path}[/cyan]")


def main():

    load_dotenv()
    
    args = parse_args()
    config = build_config(args)

    fake_rng = random.Random(config.seed)
    llm = build_llm(config, fake_rng, FakeStrategy(args.fake_strategy))

    with EventLogger.from_config(config) as logger:
        engine = GameEngine(config, llm, logger)
        engine.run()
        log_path = logger.log_path

    print_summary(engine, log_path)


if __name__ == '__main__':
    main()
