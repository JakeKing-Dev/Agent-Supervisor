"""
Behavioural drift detection over an agent trajectory.

Design rules, in order of importance:

1. It must work on a PREFIX. The whole point is calling a run bad at step 5
   of 30, not explaining it afterwards. Every signal here is computable from
   the steps seen so far.

2. It must not need a model. An LLM judge is slow, expensive, and itself
   unreliable; worse, it cannot sit inline in a production path. Everything
   here is mechanical and runs in microseconds.

3. It must be able to say "I don't know". Below `min_steps` there is not
   enough evidence, and the detector returns UNKNOWN rather than guessing.
   A supervisor that fabricates confidence is worse than none.

4. Detection and policy are separate. `assess()` returns evidence and a
   verdict; whether that warrants freezing an agent is a policy decision
   made elsewhere, so thresholds can be tuned without touching detection.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .trajectory import Trajectory


class Regime(str, Enum):
    UNKNOWN = "unknown"      # not enough steps to judge
    HEALTHY = "healthy"      # varied, progressing
    WANDERING = "wandering"  # diversity falling, errors rising
    LOOPING = "looping"      # repeating actions or cycles
    STALLED = "stalled"      # acting, but the world is not changing


@dataclass(frozen=True)
class Signals:
    """Raw measurements. Exposed so a caller can log or re-threshold them."""

    n_steps: int
    window: int
    max_consecutive_repeat: int
    repeat_fraction: float      # share of windowed steps whose signature recurred
    cycle_period: Optional[int] # smallest repeating period found, if any
    cycle_repeats: int          # how many times that cycle repeated
    diversity: float            # unique signatures / window size
    error_rate: float           # errors in window
    error_trend: float          # window error rate minus earlier error rate
    stall_run: int              # consecutive steps with an unchanged observation
    cost_ratio: float           # window mean cost / overall mean cost

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Assessment:
    regime: Regime
    score: float                 # 0..1, how concerning; 0 when UNKNOWN
    reasons: tuple[str, ...]
    signals: Signals

    @property
    def concerning(self) -> bool:
        return self.regime in (Regime.LOOPING, Regime.STALLED, Regime.WANDERING)


@dataclass
class DetectorConfig:
    min_steps: int = 4           # below this we refuse to judge
    window: int = 8              # recent steps considered
    repeat_run_threshold: int = 3        # N identical actions in a row
    repeat_fraction_threshold: float = 0.6
    cycle_min_repeats: int = 2           # ABAB counts once period=2 repeats twice
    max_cycle_period: int = 4
    diversity_threshold: float = 0.45
    error_rate_threshold: float = 0.5
    error_trend_threshold: float = 0.25
    stall_threshold: int = 3
    cost_ratio_threshold: float = 2.0


class DriftDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    # ---- signal extraction -------------------------------------------------

    def _signals(self, traj: Trajectory) -> Signals:
        c = self.config
        steps = traj.steps
        n = len(steps)
        w = min(c.window, n)
        recent = steps[-w:]
        sigs = [s.signature() for s in recent]
        all_sigs = [s.signature() for s in steps]

        # longest run of identical consecutive signatures, within the window
        max_run = run = 1 if sigs else 0
        for a, b in zip(sigs, sigs[1:]):
            run = run + 1 if a == b else 1
            max_run = max(max_run, run)

        # share of windowed steps whose signature was already seen earlier
        seen_before = sum(
            1 for i, s in enumerate(sigs) if s in set(all_sigs[: n - w + i])
        )
        repeat_fraction = seen_before / w if w else 0.0

        cycle_period, cycle_repeats = self._find_cycle(sigs)

        diversity = len(set(sigs)) / w if w else 1.0

        errors = sum(1 for s in recent if not s.ok)
        error_rate = errors / w if w else 0.0
        earlier = steps[:-w]
        earlier_rate = (
            sum(1 for s in earlier if not s.ok) / len(earlier) if earlier else 0.0
        )

        # consecutive unchanged observations (acting without effect)
        stall_run = 0
        for a, b in zip(reversed(recent), reversed(recent[:-1])):
            if (
                a.observation_digest is not None
                and a.observation_digest == b.observation_digest
            ):
                stall_run += 1
            else:
                break
        if stall_run:
            stall_run += 1  # count both ends of the unchanged pair

        overall_mean = (traj.total_cost / n) if n else 0.0
        window_mean = (sum(s.cost for s in recent) / w) if w else 0.0
        cost_ratio = (window_mean / overall_mean) if overall_mean > 0 else 1.0

        return Signals(
            n_steps=n,
            window=w,
            max_consecutive_repeat=max_run,
            repeat_fraction=repeat_fraction,
            cycle_period=cycle_period,
            cycle_repeats=cycle_repeats,
            diversity=diversity,
            error_rate=error_rate,
            error_trend=error_rate - earlier_rate,
            stall_run=stall_run,
            cost_ratio=cost_ratio,
        )

    def _find_cycle(self, sigs: list[str]) -> tuple[Optional[int], int]:
        """Smallest period p whose block repeats at the tail, with count."""
        c = self.config
        n = len(sigs)
        for p in range(1, min(c.max_cycle_period, n // 2) + 1):
            block = sigs[-p:]
            repeats = 1
            i = n - 2 * p
            while i >= 0 and sigs[i : i + p] == block:
                repeats += 1
                i -= p
            if repeats >= c.cycle_min_repeats:
                return p, repeats
        return None, 0

    # ---- verdict -----------------------------------------------------------

    def assess(self, traj: Trajectory) -> Assessment:
        c = self.config
        s = self._signals(traj)

        if s.n_steps < c.min_steps:
            return Assessment(
                regime=Regime.UNKNOWN,
                score=0.0,
                reasons=(f"only {s.n_steps} steps; need {c.min_steps}",),
                signals=s,
            )

        reasons: list[str] = []
        score = 0.0
        regime = Regime.HEALTHY

        if s.max_consecutive_repeat >= c.repeat_run_threshold:
            reasons.append(
                f"same action {s.max_consecutive_repeat}x in a row"
            )
            score = max(score, 0.55 + 0.1 * (s.max_consecutive_repeat - c.repeat_run_threshold))
            regime = Regime.LOOPING

        if s.cycle_period and s.cycle_repeats >= c.cycle_min_repeats:
            reasons.append(
                f"cycle of {s.cycle_period} action(s) repeated {s.cycle_repeats}x"
            )
            score = max(score, 0.5 + 0.12 * s.cycle_repeats)
            regime = Regime.LOOPING

        if s.repeat_fraction >= c.repeat_fraction_threshold:
            reasons.append(f"{s.repeat_fraction:.0%} of recent actions already tried")
            score = max(score, s.repeat_fraction)
            if regime is Regime.HEALTHY:
                regime = Regime.WANDERING

        if s.stall_run >= c.stall_threshold:
            reasons.append(f"state unchanged across {s.stall_run} steps")
            score = max(score, 0.6)
            if regime is not Regime.LOOPING:
                regime = Regime.STALLED

        if s.diversity <= c.diversity_threshold:
            reasons.append(f"action diversity {s.diversity:.2f}")
            score = max(score, 1.0 - s.diversity)
            if regime is Regime.HEALTHY:
                regime = Regime.WANDERING

        if s.error_rate >= c.error_rate_threshold:
            reasons.append(f"error rate {s.error_rate:.0%} in window")
            score = max(score, s.error_rate)
            if regime is Regime.HEALTHY:
                regime = Regime.WANDERING

        if s.error_trend >= c.error_trend_threshold:
            reasons.append(f"errors rising (+{s.error_trend:.0%})")
            score = max(score, 0.5 + s.error_trend)
            if regime is Regime.HEALTHY:
                regime = Regime.WANDERING

        if s.cost_ratio >= c.cost_ratio_threshold:
            reasons.append(f"cost per step {s.cost_ratio:.1f}x the run average")
            score = max(score, 0.5)
            if regime is Regime.HEALTHY:
                regime = Regime.WANDERING

        if regime is Regime.HEALTHY:
            reasons.append("varied actions, no repetition, errors stable")

        return Assessment(
            regime=regime,
            score=min(1.0, round(score, 3)),
            reasons=tuple(reasons),
            signals=s,
        )

    def first_alarm(
        self, traj: Trajectory, threshold: float = 0.6
    ) -> Optional[int]:
        """Earliest step index at which this run would have been flagged.

        This is the measurement that matters commercially: how early would we
        have caught it. Returns None if the run never trips the threshold.
        """
        for n in range(self.config.min_steps, len(traj) + 1):
            a = self.assess(traj.prefix(n))
            if a.concerning and a.score >= threshold:
                return n - 1
        return None
