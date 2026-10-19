"""
Trajectory types.

A Step is one thing an agent did. We deliberately keep this small: the
supervisor must work for any agent framework, so it sees actions and
outcomes, not prompts or model internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Step:
    """One agent action and what came of it."""

    index: int
    action: str                       # tool/function name, e.g. "http_post"
    args: Mapping[str, Any] = field(default_factory=dict)
    ok: bool = True                   # did the action itself succeed
    error: Optional[str] = None       # error class, if any
    observation_digest: Optional[str] = None  # cheap hash of resulting state
    cost: float = 0.0                 # tokens, dollars, seconds — caller's unit

    def signature(self) -> str:
        """Identity of the action for repetition purposes.

        Args are included but normalised to their keys plus a short repr of
        values, so that `click(x=10)` and `click(x=11)` are different while
        huge payloads don't dominate.
        """
        if not self.args:
            return self.action
        parts = []
        for k in sorted(self.args):
            v = self.args[k]
            s = repr(v)
            if len(s) > 48:
                s = s[:48] + "…"
            parts.append(f"{k}={s}")
        return f"{self.action}({','.join(parts)})"


@dataclass
class Trajectory:
    """An ordered run of steps. Append-only."""

    task: str = ""
    steps: list[Step] = field(default_factory=list)

    def add(self, step: Step) -> None:
        if self.steps and step.index <= self.steps[-1].index:
            raise ValueError(
                f"steps must advance: got {step.index} after {self.steps[-1].index}"
            )
        self.steps.append(step)

    def prefix(self, n: int) -> "Trajectory":
        """The first n steps. Used to ask 'what would we have known by then?'"""
        return Trajectory(task=self.task, steps=self.steps[:n])

    @property
    def total_cost(self) -> float:
        return sum(s.cost for s in self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    @classmethod
    def from_actions(cls, actions: Sequence[str], task: str = "") -> "Trajectory":
        """Convenience for tests and quick analysis."""
        t = cls(task=task)
        for i, a in enumerate(actions):
            t.add(Step(index=i, action=a))
        return t
