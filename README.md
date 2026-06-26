# agent-knowgame

A game for researching freeriding in the knowledge transfer by LLM agents.

## Install

```bash
python -m venv venv
venv\Scripts\activate       # Windows             
source venv/bin/activate    # Linux 

pip install -r requirements.txt
```

For `ollama` runs - local ollama server must be started.
For `together` runs - set `TOGETHER_API_KEY` in `.env` at repo root.

## Run

```bash
# YAML
python scripts/run_game.py configs/config_template.yaml

# inline (without YAML)
python scripts/run_game.py --n-agents 3 --m-informed 1 --share-cost 1 \
    --max-rounds 2 --model fake --api-type fake

# YAML + partial inline override
python scripts/run_game.py configs/config_template.yaml --seed 7
python scripts/run_game.py configs/config_template.yaml --max-rounds 5 --model llama3.2:3b
```

Console inline args overwrite values from YAML-config. Log traces to `logs/`.

## CLI Args

| Flag | Type | Meaning |
|---|---|---|
| `config` (positional) | path | path to a YAML config, optional if everything is passed inline |
| `--n-agents` | int | total number of agents (>= 2) |
| `--m-informed` | int | number of agents knowing token from start (`0 < m < n`) |
| `--share-cost` | float | cost of token transfer (teaching) |
| `--starting-capital` | float | initial score of every agent (default `0.0`) |
| `--max-rounds` | int | upper bound on number of game rounds |
| `--seed` | int | RNG (random) seed: token, informed pick, tie-break of agents |
| `--model` | str | model name (like `llama3.2:3b` or `fake`) |
| `--api-type` | str | `fake` / `ollama` / `together` |
| `--temperature` | float | LLM sampling |
| `--top-p` | float | LLM sampling |
| `--request-timeout` | float | seconds per LLM call (default `60`, raise for slow models) |
| `--template-version` | str | key in `PROMPT_BUILDER_REGISTRY` (`v1_baseline` now) |
| `--matcher` | str | `random_choice` / `first_come` |
| `--fake-strategy` | str | `FakeLLMClient` behavior in the share phase: `always_share` / `never_share` / `random_share` |

Reserved fields (`initiation_mode`, `payment_mode`, `token_transfer_mode`) only
work with their defaults, other options from them have not implemented yet.

## Matcher logic
Current matcher classes have different logic of resolving situation when 2 knowing agents choose the same unknowing agent to transfer the token:
- `random_choice` - 'tie-break' - random choice of knowing agent that gives token
- `first_come` - token is given by an agent firstly initialized transfer issue, second one receives feedback that this unknowing agent is already busy and it could select another one or just decline (it's like an opportunity to make different decision).

## YAML config

`configs/config_template.yaml` - base working template, copy and tweak per experiment or just modify current.
Key names match `GameConfig` fields. Minimum:

```yaml
n_agents: 5
m_informed: 2
share_cost: 1.0
max_rounds: 20
seed: 42

model: llama3.2:3b
api_type: ollama
temperature: 0.7
top_p: 1.0

template_version: v1_baseline
matcher: first_come
```
Use `model: fake`, `api_type: fake` to init pre-coded run without LLM models.
Uses `always_share` strategy as default. Use `--fake-strategy` as CLI argument to swap.

## Other scripts

```bash
# dump all prompt templates of a builder to markdown (stubs for token/ids/scores)
python scripts/dump_prompts.py --template-version v1_baseline

# single-call sanity check against a Together model (needs TOGETHER_API_KEY)
python scripts/probe_together.py openai/gpt-oss-20b

# ask one agent "how many rounds do you expect?" - JSONL out, supports seed sweeps
python scripts/probe_rounds.py configs/config_template.yaml --seeds-file probes/seeds.txt
```

## Tests

```bash
python -m pytest
```

## Cost accounting (after runs)

Token accounting implemented in `LLMClient`, saves `token_usage.txt` to root folder but does not count usage cost.