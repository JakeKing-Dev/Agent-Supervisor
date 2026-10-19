"""
Where detection becomes a decision.

The detector says what it sees. This decides what to do about it, and the
separation is deliberate: thresholds are a business choice (how much wasted
spend is worth one wrongly-halted run?) and should be tunable without
touching the measurement.

One mechanism matters more than the thresholds: `persist_steps`. In the
AgentRewardBench annotations, slightly over half of all runs show looping at
some point, yet only ~72% fail — so a detector that halts on first sight of
repetition will stop runs that were about to recover. Requiring the concern
to hold for several consecutive steps trades a little lead time for a large
drop in false alarms. That trade is the product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .detector import Assessment, DetectorConfig, DriftDetector, Regime
from .freeze import FreezeManager, RecoveryEvidence
from .trajectory import Step, Trajectory


class Decision(str, Enum):
    CONTINUE = "continue"
    WARN = "warn"
    FREEZE = "freeze"


@dataclass
class SupervisorPolicy:
    warn_score: float = 0.5
    freeze_score: float = 0.7
    persist_steps: int = 3        # consecutive concerning steps before freezing
    max_cost: Optional[float] = None   # hard budget ceiling, freezes regardless
    freeze_on: tuple[Regime, ...] = (Regime.LOOPING, Regime.STALLED)


@dataclass
class Verdict:
    decision: Decision
    assessment: Assessment
    streak: int
    note: str = ""


class Supervisor:
    """Feed it steps as they happen; it tells you whether to keep going."""

    def __init__(
        self,
        policy: SupervisorPolicy | None = None,
        detector_config: DetectorConfig | None = None,
        freezer: FreezeManager | None = None,
        task: str = "",
        metrics=None,
    ) -> None:
        self.policy = policy or SupervisorPolicy()
        self.detector = DriftDetector(detector_config)
        self.freezer = freezer or FreezeManager()
        self.trajectory = Trajectory(task=task)
        self.metrics = metrics      # optional MetricsRegistry; duck-typed
        self._streak = 0
        self.history: list[Verdict] = []

    def _publish(self, v: Verdict, cost_saved: float = 0.0) -> None:
        if self.metrics is None:
            return
        self.metrics.record(
            self.trajectory.task or "unnamed",
            v,
            frozen=self.freezer.frozen,
            run_cost=self.trajectory.total_cost,
            cost_saved=cost_saved,
        )

    def observe(self, step: Step) -> Verdict:
        self.trajectory.add(step)
        a = self.detector.assess(self.trajectory)
        p = self.policy

        if p.max_cost is not None and self.trajectory.total_cost > p.max_cost:
            self.freezer.freeze(
                "BUDGET_EXCEEDED",
                f"spent {self.trajectory.total_cost:.2f} > {p.max_cost:.2f}",
            )
            v = Verdict(Decision.FREEZE, a, self._streak, "budget ceiling")
            self.history.append(v)
            self._publish(v)
            return v

        concerning = a.concerning and a.score >= p.warn_score
        self._streak = self._streak + 1 if concerning else 0

        decision = Decision.CONTINUE
        note = ""

        if concerning:
            decision = Decision.WARN
            note = f"{a.regime.value}: {a.reasons[0]}"
            if (
                a.regime in p.freeze_on
                and a.score >= p.freeze_score
                and self._streak >= p.persist_steps
            ):
                decision = Decision.FREEZE
                note = (
                    f"{a.regime.value} sustained for {self._streak} steps "
                    f"(score {a.score:.2f})"
                )
                self.freezer.freeze(a.regime.value.upper(), note)

        v = Verdict(decision, a, self._streak, note)
        self.history.append(v)
        self._publish(v)
        return v

    # ---- convenience -------------------------------------------------------

    @property
    def frozen(self) -> bool:
        return self.freezer.frozen

    def resume(self, evidence: RecoveryEvidence) -> tuple[bool, str]:
        ok, why = self.freezer.attempt_unfreeze(evidence)
        if ok:
            self._streak = 0
        return ok, why

    def replay(self, traj: Trajectory) -> Optional[int]:
        """Run a completed trajectory through and report the step index at
        which it would have been frozen, or None."""
        sup = Supervisor(self.policy, self.detector.config, FreezeManager(), traj.task)
        for s in traj.steps:
            v = sup.observe(s)
            if v.decision is Decision.FREEZE:
                return s.index
        return None
