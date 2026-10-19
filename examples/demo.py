"""
A support agent processing a refund gets stuck, and is caught mid-run.

Run:  PYTHONPATH=. python3 examples/demo.py
"""

import time

from supervisor import (
    Decision,
    RecoveryEvidence,
    Step,
    Supervisor,
    SupervisorPolicy,
)

# What the agent does. The first six steps are real work; then the payment
# API starts failing and the agent falls into a retry loop, which is the
# single most common way an autonomous run quietly burns money.
SCRIPT = [
    ("lookup_order",     {"id": "A-4471"}, True,  None,      "s1", 0.02),
    ("read_policy",      {"doc": "refunds"}, True, None,     "s2", 0.01),
    ("check_eligibility", {"id": "A-4471"}, True, None,      "s3", 0.02),
    ("compute_amount",   {"id": "A-4471"}, True,  None,      "s4", 0.01),
    ("notify_customer",  {"tmpl": "pending"}, True, None,    "s5", 0.03),
    ("issue_refund",     {"id": "A-4471"}, False, "HTTP502", "s5", 0.08),
    ("issue_refund",     {"id": "A-4471"}, False, "HTTP502", "s5", 0.09),
    ("issue_refund",     {"id": "A-4471"}, False, "HTTP502", "s5", 0.11),
    ("issue_refund",     {"id": "A-4471"}, False, "HTTP502", "s5", 0.14),
    ("issue_refund",     {"id": "A-4471"}, False, "HTTP502", "s5", 0.17),
    ("issue_refund",     {"id": "A-4471"}, False, "HTTP502", "s5", 0.21),
    ("issue_refund",     {"id": "A-4471"}, False, "HTTP502", "s5", 0.26),
]

BAR = "─" * 72


def main() -> None:
    sup = Supervisor(
        policy=SupervisorPolicy(persist_steps=3, freeze_score=0.7, max_cost=5.00),
        task="refund order A-4471",
    )

    print(BAR)
    print(f"task: {sup.trajectory.task}")
    print(BAR)
    print(f"{'#':>2}  {'action':<28} {'ok':<4} {'regime':<10} {'score':>5}  decision")
    print(BAR)

    for i, (action, args, ok, err, obs, cost) in enumerate(SCRIPT):
        v = sup.observe(
            Step(
                index=i,
                action=action,
                args=args,
                ok=ok,
                error=err,
                observation_digest=obs,
                cost=cost,
            )
        )
        mark = "ok" if ok else "ERR"
        flag = {
            Decision.CONTINUE: "",
            Decision.WARN: "  <- warn",
            Decision.FREEZE: "  <- FREEZE",
        }[v.decision]
        print(
            f"{i:>2}  {action + str(tuple(args.values())):<28.28} {mark:<4} "
            f"{v.assessment.regime.value:<10} {v.assessment.score:>5.2f}{flag}"
        )
        if v.decision is Decision.FREEZE:
            print(BAR)
            print(f"HALTED at step {i}: {v.note}")
            print("evidence:")
            for r in v.assessment.reasons:
                print(f"  - {r}")
            spent = sup.trajectory.total_cost
            remaining = sum(c for *_, c in SCRIPT[i + 1:])
            print(f"spent ${spent:.2f}; stopped ${remaining:.2f} of further spend")
            break
    else:
        print(BAR)
        print("ran to completion without intervention")
        return

    print(BAR)
    print("a frozen agent can still wind down, but cannot act:")
    print(f"  release_lock (reducing) -> allowed={sup.freezer.allows('release_lock', reducing=True)}")
    print(f"  issue_refund            -> allowed={sup.freezer.allows('issue_refund')}")

    print(BAR)
    print("the agent cannot vouch for itself; unfreezing takes evidence:")
    ok, why = sup.resume(RecoveryEvidence(reconcile_clean=True))
    print(f"  claims it's fine, no reconciliation  -> {ok}, {why}")
    ok, why = sup.resume(
        RecoveryEvidence(
            reconciled_at=time.time(), reconcile_clean=True, unresolved_actions=1
        )
    )
    print(f"  reconciled, 1 action unresolved      -> {ok}, {why}")
    ok, why = sup.resume(
        RecoveryEvidence(reconciled_at=time.time(), reconcile_clean=True)
    )
    print(f"  reconciled, clean, nothing pending   -> {ok}, {why}")

    print(BAR)
    print("audit trail:")
    for r in sup.freezer.audit:
        print(f"  {time.strftime('%H:%M:%S', time.localtime(r.ts))}  {r.event:<17} {r.reason} {r.detail}")
    print(BAR)


if __name__ == "__main__":
    main()
