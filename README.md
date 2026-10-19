# supervisor

![CI](https://github.com/JakeKing-Dev/Agent-Supervisor/actions/workflows/ci.yml/badge.svg)

Runtime supervision for autonomous agents. Watches an agent's behaviour as it
runs and halts it when the behaviour stops making sense — mechanically, with
no model in the loop.

Existing agent evaluation is a post-mortem discipline: it tells you the run
failed yesterday. This stops the run now.

## Why this is the thing to build

In the AgentRewardBench human annotations (1,408 expert-labelled agent runs
across WebArena, VisualWebArena, WorkArena and AssistantBench):

| | failed |
|---|---|
| runs annotated as looping | **93.7 %** |
| runs not looping | **49.7 %** |
| all runs | 71.9 % |

Looping is a strong failure signal, and — crucially — it's *mechanically*
detectable. No judge model, no inference cost, microseconds per step. That
makes it something you can put inline in a production path.

Caveat, stated plainly: those annotations are whole-trajectory hindsight.
They establish that the behavioural marker is real; they do not yet prove
early detection. That's the open measurement (see below).

## Install

No dependencies. Python 3.10+.

```bash
PYTHONPATH=. python3 -m unittest discover -s tests   # 22 tests
PYTHONPATH=. python3 examples/demo.py                # watch an agent get halted
```

## Run it as a service

```bash
docker compose up --build
```

- supervisor — http://localhost:9464/metrics
- Prometheus — http://localhost:9090
- Grafana — http://localhost:3000 (dashboard provisioned, anonymous access on)

The service generates synthetic agent runs, a configurable fraction of which
degrade into retry loops, so the dashboard has live data. In real use
`observe()` is called from whatever drives the agent.

As a Kubernetes sidecar: `kubectl apply -f deploy/k8s/supervisor-sidecar.yaml`.
It deploys beside the agent in the same pod rather than as a central service,
so that an unreachable supervisor means the agent is down too, rather than the
agent running unsupervised.

## Metrics

Prometheus text format, served from the stdlib — the supervisor sits inline in
a production path, so every dependency it carries is one it forces on whoever
adopts it.

| series | type | |
|---|---|---|
| `agent_supervisor_steps_total` | counter | steps observed |
| `agent_supervisor_verdicts_total` | counter | by decision |
| `agent_supervisor_freezes_total` | counter | by reason |
| `agent_supervisor_drift_score` | gauge | 0–1, current |
| `agent_supervisor_regime` | gauge | one-hot over regimes |
| `agent_supervisor_frozen` | gauge | 0/1 |
| `agent_supervisor_run_cost` | gauge | spend this run |
| `agent_supervisor_cost_saved_total` | counter | spend prevented by halting |

That last one is the series anyone paying for this will look at first.

## Use

```python
from supervisor import Supervisor, SupervisorPolicy, Step, Decision

sup = Supervisor(
    policy=SupervisorPolicy(persist_steps=3, max_cost=5.00),
    task="refund order A-4471",
)

for i, action in enumerate(agent_run()):
    verdict = sup.observe(Step(index=i, action=action.name, args=action.args,
                               ok=action.ok, observation_digest=action.state_hash,
                               cost=action.cost))
    if verdict.decision is Decision.FREEZE:
        agent.stop(verdict.note)
        break
```

## Design decisions worth knowing

**It can say "I don't know."** Below `min_steps` the detector returns
`UNKNOWN` rather than guessing. A supervisor that fabricates confidence is
worse than no supervisor.

**It works on a prefix.** Every signal is computable from the steps seen so
far. The point is calling a run bad at step 5 of 30, not explaining it
afterwards.

**Detection and policy are separate.** `DriftDetector.assess()` returns
evidence; `SupervisorPolicy` decides what that warrants. Thresholds are a
business choice — how much wasted spend is one wrongly-halted run worth —
and shouldn't require touching the measurement.

**Persistence over hair-trigger.** Just over half of all annotated runs loop
at some point, yet only ~72 % fail, so halting on first sight of repetition
would stop runs that were about to recover. `persist_steps` requires the
concern to hold for several consecutive steps, trading a little lead time for
far fewer false alarms. That trade is the product.

**The agent does not vouch for itself.** `attempt_unfreeze` takes *evidence*
— a reconciliation timestamp, a clean flag, a count of unresolved actions —
and checks it here, rather than accepting a boolean from the component that
may itself have failed. Stale reconciliation is refused.

**A frozen agent can still wind down.** Reducing actions (releasing locks,
closing sessions, cancelling pending requests) stay permitted while frozen.
A frozen agent holding resources forever is usually worse than a controlled
stop.

## Signals

| signal | catches |
|---|---|
| consecutive repetition | retry storms, stuck tool calls |
| cycle detection (period ≤ 4) | A→B→A→B navigation loops |
| repeat fraction | revisiting already-tried actions |
| action diversity | collapse toward a single behaviour |
| error rate and trend | a degrading environment |
| stall run | acting without changing the world |
| cost acceleration | spend running away |

Actions are identified by name *and* arguments, so `click(x=1), click(x=2)`
reads as progress while `click(x=1)` four times reads as a loop.

## What this is not

Not a permissions system. Tools like Codezero's Cordon let you declare what
an agent may not do, statically, in advance. This is the other axis:
behaviour observed at runtime, and authority that can be withdrawn
automatically. The two are complementary.

Not yet validated on production traffic. See below.

## Open measurement

The number that decides how good this is: replay real labelled trajectories,
cut at step 3 / 5 / 10, and measure how often the detector flags a run that
went on to fail — and how often it flags one that recovered. High early
precision means automatic intervention is safe. Low precision means you can
only advise a human, which is a much weaker product.

The data for this is AgentRewardBench's step-level trajectories on Hugging
Face. `Supervisor.replay()` and `DriftDetector.first_alarm()` exist to run
exactly that experiment.

## Status

v0.2.0. Detector, freeze manager, policy layer and metrics are implemented
and tested (22 tests, CI on Python 3.10-3.13). Containerised, with Prometheus
scraping and a provisioned Grafana dashboard.

Reconciliation, the promotion ladder and calibration are designed but not
built. Validation against production traffic has not been done.
