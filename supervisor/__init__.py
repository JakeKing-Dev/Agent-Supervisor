"""
Runtime supervision for autonomous agents.

Watches an agent's behaviour as it runs, and stops it when the behaviour
stops making sense — mechanically, with no model in the loop.
"""

from .trajectory import Step, Trajectory
from .detector import (
    Assessment,
    DetectorConfig,
    DriftDetector,
    Regime,
    Signals,
)
from .freeze import (
    AuditRecord,
    FreezeManager,
    FrozenError,
    RecoveryEvidence,
)
from .supervisor import Decision, Supervisor, SupervisorPolicy, Verdict
from .metrics import MetricsRegistry, dashboard_html, serve_metrics

__all__ = [
    "Step",
    "Trajectory",
    "Assessment",
    "DetectorConfig",
    "DriftDetector",
    "Regime",
    "Signals",
    "AuditRecord",
    "FreezeManager",
    "FrozenError",
    "RecoveryEvidence",
    "Decision",
    "Supervisor",
    "SupervisorPolicy",
    "Verdict",
    "MetricsRegistry",
    "serve_metrics",
    "dashboard_html",
]

__version__ = "0.3.0"
