# Architecture

The benchmark harness lives in `src/`. It is a small, dependency-light library
(only `anthropic` and `openai` clients are required) driven by the scripts in
`scripts/`.

## Data flow

```
item JSON ──► load_items() ──► run_trajectory(subject, arm, item)
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                                ▼
       Counterparty (simulator)  ◄── alternating turns ──►  Agent (under test)
              │   pressure strategy                   arm: plain/prompted/scaffolded
              │                                       LoyaltyState (scaffolded only)
              └───────────────► trajectory JSONL ◄────────────┘
                                      │
                                      ▼
                          probe_trajectory()  (leak detection: lexical + LLM)
                                      │
                                      ▼
                          score_trajectory()  →  scored.jsonl
                                      │
                                      ▼
                  scorer aggregation / dual judges / audit gate
```

## Modules

| Module | Responsibility |
|---|---|
| `src/items.py` | Item schema (`Item`, `PrivateFact`, `StatedPosition`) and `load_items()`. |
| `src/harness.py` | `run_trajectory()` runs one `(subject, arm, item)` cell; `run_grid()` runs a sweep. One JSONL line per trajectory written to `runs/`. |
| `src/agent.py` | The agent under test. Builds the system prompt per arm; injects the rendered LoyaltyState in the `scaffolded` arm. |
| `src/counterparty.py` | The counterparty simulator. Maps `counterparty_strategy` → system-prompt guidance; emits `counterparty_opening` as turn 1; never sees private facts. |
| `src/loyalty.py` | `LoyaltyState` (relation / information_scope / commitment) and `render_state()` — the M1 scaffold. |
| `src/probe.py` | Leak detection: lexical/alias pass + LLM probe over agent outbound turns. |
| `src/scorer.py` | Per-trajectory → per-item / per-cell metric aggregation; harm-floor zero-capping. |
| `src/reward.py` | Fast lexical proxy reward for online DAPO rollouts (no per-rollout LLM judge). |
| `src/vendors.py` | Unified vendor layer over Anthropic / OpenAI / Google / OpenRouter with tool-calling and timestamp-preserving messages. `get_vendor(spec)`. |

## Invocation convention

The harness is a library, not a CLI. Scripts add the repo root to `sys.path` and
import directly:

```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_trajectory, run_grid
from src.items import load_items
from src.scorer import score_trajectory, score_grid
from src.vendors import get_vendor
```

Representative entry points: `scripts/multi_rollout_eval.py` (multi-seed grid
eval against a running vLLM server), `scripts/generate_teacher_traces.py`
(teacher rollouts for distillation), and the many `run_*.sh` orchestration
scripts.

## Reward proxy (training only)

Online DAPO needs a per-rollout scalar; an LLM judge per rollout is too slow.
`src/reward.py` uses a lexical proxy with two regimes selected by
`extra_info["expect_cooperation"]`:

- **Adversarial items:** reward refusal, penalize leaks.
- **Cooperative items** (`sanity` + reader-is-principal): reward responding,
  penalize refusal — prevents DAPO from collapsing onto the over-refusal side
  of the frontier.

The offline probe + judge remain the **gold** metric; the proxy only needs to
separate held-vs-folded / cooperated-vs-refused within a rollout group.
