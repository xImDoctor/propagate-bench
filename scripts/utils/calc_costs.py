"""Estimate API spend from token_usage.txt against per-provider price tables.

Reads token counts written by LLMClient._update_token_log and multiplies
them by per-1K-token prices loaded from one or more YAML files.

Two pricing shapes are supported per model entry:
    base one: {input, output}
    tiered (used for DeepSeek): {input_cache_hit, input_cache_miss, output}
                       – prompt tokens are split into cache_hit and cache_miss
                       (miss is derived as prompt_tokens - prompt_cache_hit_tokens).

Models without a matching price entry are reported as skipped.

Usage:
    python scripts/utils/calc_costs.py
    python scripts/utils/calc_costs.py --token-log token_usage.txt
    python scripts/utils/calc_costs.py --prices configs/together_prices.yaml \\
                                       --prices configs/deepseek_prices.yaml
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml


DEFAULT_TOKEN_LOG = Path('token_usage.txt')
DEFAULT_PRICES_FILES = [
    Path('configs/together_prices.yaml'),
    Path('configs/deepseek_prices.yaml'),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Estimate API spend from token_usage.txt.')
    p.add_argument('--token-log', type=str, default=str(DEFAULT_TOKEN_LOG),
                   help=f'Path to token usage JSON (default: {DEFAULT_TOKEN_LOG})')
    p.add_argument('--prices', type=str, action='append', default=None,
                   help='Path to YAML pricing table. Repeatable. Defaults to: '
                        + ', '.join(str(f) for f in DEFAULT_PRICES_FILES))

    return p.parse_args()


def load_prices(paths: list[Path]) -> dict:
    merged: dict = {}
    for p in paths:
        if not p.exists():
            print(f'[WARN] Prices file missing: {p}')
            continue
        merged.update(yaml.safe_load(p.read_text(encoding='utf-8')) or {})
    return merged


def _cost_for_entry(entry: dict, tok: dict) -> float:
    """Bill this (model, counter) row against the given price entry.

    Tiered pricing is applied when the entry has input_cache_miss.
    """
    prompt     = int(tok.get('prompt_tokens', 0))
    completion = int(tok.get('completion_tokens', 0))

    if 'input_cache_miss' in entry:
        hit  = int(tok.get('prompt_cache_hit_tokens', 0))
        miss = max(prompt - hit, 0)
        return (hit  / 1000.0) * entry['input_cache_hit'] \
             + (miss / 1000.0) * entry['input_cache_miss'] \
             + (completion / 1000.0) * entry['output']

    return (prompt     / 1000.0) * entry['input'] \
         + (completion / 1000.0) * entry['output']


def main():
    args = parse_args()

    log_path = Path(args.token_log)
    if not log_path.exists():
        print(f'No token log found at {log_path}')
        return

    prices_paths = [Path(p) for p in (args.prices or [str(x) for x in DEFAULT_PRICES_FILES])]
    prices = load_prices(prices_paths)

    totals = json.loads(log_path.read_text(encoding='utf-8'))

    grand_total = 0.0

    print('Per-model spend estimate:')
    print('-' * 100)

    for key, tok in totals.items():
        prompt     = int(tok.get('prompt_tokens', 0))
        completion = int(tok.get('completion_tokens', 0))
        hit        = int(tok.get('prompt_cache_hit_tokens', 0))
        rsn        = int(tok.get('completion_reasoning_tokens', 0))

        entry = prices.get(key)

        if not entry:
            print(f'[WARN] No prices for {key}; tokens={prompt + completion} - skipped.')
            continue

        cost = _cost_for_entry(entry, tok)
        grand_total += cost

        extras = ''
        if 'input_cache_miss' in entry and prompt > 0:
            extras += f'  cache_hit={hit} ({hit/prompt:.1%})'
        if rsn:
            extras += f'  reasoning={rsn}'

        print(f'{key:55s} prompt={prompt:8d}  completion={completion:8d}  cost=${cost:,.4f}{extras}')

    print('-' * 100)
    print(f'TOTAL SPEND: ${grand_total:,.4f}')


if __name__ == '__main__':
    main()
