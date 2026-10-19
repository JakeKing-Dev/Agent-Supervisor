"""
Runs the supervisor as a long-lived service with a metrics endpoint.

This is what gets containerised: a process that supervises agent runs and
exposes Prometheus metrics on :9464/metrics for scraping.

For the demo it generates synthetic agent runs — a configurable fraction of
which degrade into retry loops — so the dashboard has something to show. In
real use, `observe()` is called from whatever is driving the agent.

  PYTHONPATH=. python3 examples/service.py
  curl localhost:9464/metrics
"""

from __future__ import annotations

import os
import random
import signal
import sys
import time

from supervisor import (
    Decision,
    MetricsRegistry,
    Step,
    Supervisor,
    SupervisorPolicy,
    serve_metrics,
)

PORT = int(os.environ.get("METRICS_PORT", "9464"))
STEP_DELAY = float(os.environ.get("STEP_DELAY", "0.35"))
FAIL_RATE = float(os.environ.get("FAIL_RATE", "0.4"))
SEED = os.environ.get("SEED")

TOOLS = [
    "lookup_order", "read_policy", "check_eligibility", "compute_amount",
    "notify_customer", "update_record", "fetch_invoice", "verify_identity",
]

_running = True


def _stop(*_):
    global _running
    _running = False


def synthetic_run(rng: random.Random) -> list[Step]:
    """A healthy run, or one that degrades into a retry loop partway."""
    steps: list[Step] = []
    doomed = rng.random() < FAIL_RATE
    length = rng.randint(8, 18)
    break_at = rng.randint(3, max(4, length // 2)) if doomed else length + 1

    state = 0
    for i in range(length):
        if i >= break_at:
            steps.append(Step(
                index=i, action="issue_refund", args={"id": "A-1"},
                ok=False, error="HTTP502",
                observation_digest=f"s{state}", cost=round(rng.uniform(0.05, 0.3), 3),
            ))
        else:
            state += 1
            steps.append(Step(
                index=i, action=rng.choice(TOOLS), args={"n": i},
                ok=True, observation_digest=f"s{state}",
                cost=round(rng.uniform(0.01, 0.06), 3),
            ))
    return steps


def main() -> None:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    rng = random.Random(int(SEED) if SEED else None)
    registry = MetricsRegistry()
    httpd = serve_metrics(registry, port=PORT)
    print(f"metrics on :{PORT}/metrics", flush=True)

    run_no = 0
    while _running:
        run_no += 1
        task = f"run-{run_no:04d}"
        sup = Supervisor(
            policy=SupervisorPolicy(persist_steps=3, max_cost=4.0),
            task=task,
            metrics=registry,
        )
        steps = synthetic_run(rng)
        for n, step in enumerate(steps):
            if not _running:
                break
            v = sup.observe(step)
            if v.decision is Decision.FREEZE:
                saved = sum(s.cost for s in steps[n + 1:])
                registry.record(
                    task, v, frozen=True,
                    run_cost=sup.trajectory.total_cost, cost_saved=saved,
                )
                print(f"{task}: FROZEN at step {step.index} — {v.note} "
                      f"(prevented ${saved:.2f})", flush=True)
                break
            time.sleep(STEP_DELAY)
        else:
            print(f"{task}: completed, {len(steps)} steps, "
                  f"${sup.trajectory.total_cost:.2f}", flush=True)

    httpd.shutdown()
    httpd.server_close()
    print("stopped", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
