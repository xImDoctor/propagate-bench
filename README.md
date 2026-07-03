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
# dump prompt templates of a builder to markdown - one file per game mode
# writes prompts_<version>_anonymous.md and prompts_<version>_named.md
python scripts/dump_prompts.py --template-version v1_baseline

# single-call sanity check against a Together model (needs TOGETHER_API_KEY)
python scripts/probe_together.py openai/gpt-oss-20b

# ask one agent "how many rounds do you expect?" - legacy variant, needs a full game YAML
python scripts/probe_rounds.py configs/config_template.yaml --seeds-file probes/seeds.txt

# estimate API spend from token_usage.txt against the prices table
python scripts/calc_costs.py
```

## Experimental probes

Grid-based single-call probes for behavioural measurements on informed agents. Both probes share the same grid YAML format (see `configs/probes/`).

```bash
# will/will-not-share, one call per (N, K, P, seed).
# For reasoning-capable Together models the reasoning trace is captured too.
python scripts/probe_share.py --grid-file configs/probes/probe_qwen3_235b_10seeds.yaml

# integer expectation of remaining rounds, one call per (N, K, P, seed).
# Same grid layout, `early_stopping` is ignored.
python scripts/probe_expected_rounds.py --grid-file configs/probes/probe_qwen3_235b_10seeds.yaml
```

Configs in `configs/probes/`:

| File | Purpose |
|---|---|
| `probe_ollama_smoke.yaml` | tiny smoke on local Ollama, `early_stopping: true` |
| `probe_<model>_10seeds.yaml` | 10-seed grid on Together API, `early_stopping: true` |
| `probe_<model>_100seeds.yaml` | 100-seed grid on Together API, `early_stopping: false` |

Grid schema:

```yaml
n_agents:   [2, 3, 6, 10]                              # population sizes
m_informed: [[1], [1, 2], [1, 3, 5], [1, 3, 5, 7, 9]]  # per-N K values
seeds:      [42, 43, ..., 51]                          # RNG seed pool
early_stopping: true                                   # split seeds in half; skip 2nd if 1st is uniform
model:      openai/gpt-oss-20b
api_type:   together
reasoning:  true                                       # request reasoning trace on Together
request_timeout: 240
```

`share_cost` per K is computed as unique values of `[0.1, 1, K/2, K]` with `K/2` included only if `1 < K/2 < K`. Output goes to `probes/probe_share_<sanitized_model>_<date>.{jsonl,csv}` (probe_share) or `probes/probe_expected_rounds_<sanitized_model>_<date_time>_<N>seeds.{jsonl,csv}` (probe_expected_rounds).

## Tests

```bash
python -m pytest
```

## Cost accounting (after runs)

`LLMClient` writes per-model token counts into `token_usage.txt` at repo root.

Run `scripts/calc_costs.py` to estimate spend: it reads the log and multiplies counts by per-1K prices from `configs/together_prices.yaml`. Models without a matching price entry (e.g. `fake:*`, `ollama:*`) are skipped with a warning.

Update `configs/together_prices.yaml` manually when Together changes prices.