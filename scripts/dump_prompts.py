"""Dump all prompt templates of a given builder version to a single markdown file.

Script for prompt extraction to preview and edit without
running the engine.
Stubs replace runtime values (token, agent ids, scores). 

Usage:
    python scripts/dump_prompts.py
    python scripts/dump_prompts.py --template-version v1_baseline --output prompts_v1.md
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.config import GameConfig
from game.prompt_builder import create_prompt_builder
from game.states import AgentState, RoundResult


STUB_TOKEN = '<TOKEN>'


def stub_config(template_version: str) -> GameConfig:
    
    # min config that passes validation, values are placeholders
    return GameConfig(
        n_agents=4,
        m_informed=1,
        share_cost=0.01,
        starting_capital=1.0,
        max_rounds=5,
        seed=42,
        model='<MODEL>',
        api_type='fake',
        template_version=template_version,
    )


def stub_round_result() -> RoundResult:
    return RoundResult(
        round_num=1,
        answers={
            'agent_0': '<TOKEN>',
            'agent_1': '<unknown>',
            'agent_2': '<TOKEN>',
            'agent_3': '<unknown>',
        },
        correct_answers={
            'agent_0': True,
            'agent_1': False,
            'agent_2': True,
            'agent_3': False,
        },
        scores_after={
            'agent_0': 1.5,
            'agent_1': 1.5,
            'agent_2': 1.5,
            'agent_3': 1.5,
        },
        correct_count=2,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Dump all prompt templates of a builder to markdown.')
    p.add_argument('--template-version', type=str, default='v1_baseline')
    p.add_argument('--output', type=str, default=None,
                   help='Output md path. Default: prompts_<version>.md')
    
    return p.parse_args()


def dump_prompts(template_version: str, output_path: Path) -> None:
    
    config = stub_config(template_version)
    builder = create_prompt_builder(config, STUB_TOKEN)

    knowing = AgentState(agent_id='agent_0', knows_token=True, score=1.0)
    unknowing = AgentState(agent_id='agent_1', knows_token=False, score=1.0)
    all_ids = [f'agent_{i}' for i in range(config.n_agents)]
    round_result = stub_round_result()
    unknowing_ids = ['agent_1', 'agent_3']

    sections = [
        ('build_system_prompt (knowing agent)',
            builder.build_system_prompt(knowing, all_ids)),
        ('build_system_prompt (unknowing agent)',
            builder.build_system_prompt(unknowing, all_ids)),
        ('build_answer_prompt',
            builder.build_answer_prompt(knowing, round_num=1)),
        ('build_share_prompt',
            builder.build_share_prompt(knowing, round_result, unknowing_ids)),
        ('build_retry_share_prompt',
            builder.build_retry_share_prompt(knowing, round_result, ['agent_3'], previous_target='agent_1')),
        ('build_round_summary',
            builder.build_round_summary(knowing, round_result)),
        ('build_transfer_token_prompt',
            builder.build_transfer_token_prompt('agent_0')),
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f'# Prompts dump - {template_version}\n\n')
        f.write('Stubs: token=`<TOKEN>`, agent_ids=`agent_0..agent_3`, scores=placeholder.\n\n')
        for title, text in sections:
            f.write(f'## {title}\n\n```\n{text}\n```\n\n')


def main():
    args = parse_args()
    
    output = Path(args.output or f'prompts_{args.template_version}.md')
    output.parent.mkdir(parents=True, exist_ok=True)

    dump_prompts(args.template_version, output)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
