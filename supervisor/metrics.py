"""
Prometheus metrics, in the standard text exposition format.

Written against the stdlib on purpose. The supervisor sits inline in a
production path, so every dependency it carries is a dependency it forces on
whoever adopts it — and a metrics client is a poor reason to pull in a tree.
The exposition format is a documented, stable text protocol; implementing it
is about eighty lines.

Exposed series:

  agent_supervisor_steps_total{task}               counter
  agent_supervisor_verdicts_total{task,decision}   counter
  agent_supervisor_freezes_total{task,reason}      counter
  agent_supervisor_drift_score{task}               gauge   0..1
  agent_supervisor_regime{task,regime}             gauge   1 for current
  agent_supervisor_frozen{task}                    gauge   0/1
  agent_supervisor_run_cost{task}                  gauge
  agent_supervisor_cost_saved_total{task}          counter (est.)
"""

from __future__ import annotations

import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterable, Optional

try:  # py3.9+ has importlib.resources.files
    from importlib.resources import files as _res_files
except ImportError:  # pragma: no cover
    _res_files = None  # type: ignore[assignment]

from .detector import Regime
from .supervisor import Decision, Verdict

_ALL_REGIMES = tuple(r.value for r in Regime)


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(**kw: str) -> str:
    if not kw:
        return ""
    inner = ",".join(f'{k}="{_escape(str(v))}"' for k, v in sorted(kw.items()))
    return "{" + inner + "}"


class MetricsRegistry:
    """Thread-safe. Call `record()` for each verdict; scrape with `render()`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps: dict[str, int] = defaultdict(int)
        self._verdicts: dict[tuple[str, str], int] = defaultdict(int)
        self._freezes: dict[tuple[str, str], int] = defaultdict(int)
        self._score: dict[str, float] = {}
        self._regime: dict[str, str] = {}
        self._frozen: dict[str, bool] = {}
        self._cost: dict[str, float] = {}
        self._saved: dict[str, float] = defaultdict(float)

    def record(
        self,
        task: str,
        verdict: Verdict,
        *,
        frozen: bool = False,
        run_cost: float = 0.0,
        cost_saved: float = 0.0,
        count_freeze: bool = True,
    ) -> None:
        """Record one verdict.

        `count_freeze` exists because a halted run keeps producing FREEZE
        verdicts for as long as anyone keeps observing it. `freezes_total` is
        meant to count halts, so the caller passes False for the repeats —
        otherwise `rate()` on that counter measures how chatty the caller is
        rather than how often agents are being stopped.
        """
        with self._lock:
            self._steps[task] += 1
            self._verdicts[(task, verdict.decision.value)] += 1
            self._score[task] = verdict.assessment.score
            self._regime[task] = verdict.assessment.regime.value
            self._frozen[task] = frozen
            self._cost[task] = run_cost
            if verdict.decision is Decision.FREEZE and count_freeze:
                self._freezes[(task, verdict.assessment.regime.value)] += 1
                self._saved[task] += cost_saved

    def render(self) -> str:
        out: list[str] = []

        def emit(name: str, typ: str, help_: str, rows: Iterable[tuple[str, float]]):
            rows = list(rows)
            if not rows:
                return
            out.append(f"# HELP {name} {help_}")
            out.append(f"# TYPE {name} {typ}")
            for lbl, val in rows:
                out.append(f"{name}{lbl} {val}")

        with self._lock:
            emit(
                "agent_supervisor_steps_total", "counter",
                "Agent steps observed.",
                ((_labels(task=t), v) for t, v in self._steps.items()),
            )
            emit(
                "agent_supervisor_verdicts_total", "counter",
                "Verdicts issued, by decision.",
                ((_labels(task=t, decision=d), v) for (t, d), v in self._verdicts.items()),
            )
            emit(
                "agent_supervisor_freezes_total", "counter",
                "Freezes triggered, by reason.",
                ((_labels(task=t, reason=r), v) for (t, r), v in self._freezes.items()),
            )
            emit(
                "agent_supervisor_drift_score", "gauge",
                "Most recent drift score, 0 to 1.",
                ((_labels(task=t), v) for t, v in self._score.items()),
            )
            emit(
                "agent_supervisor_regime", "gauge",
                "Current regime; 1 on the active one.",
                (
                    (_labels(task=t, regime=r), 1.0 if r == cur else 0.0)
                    for t, cur in self._regime.items()
                    for r in _ALL_REGIMES
                ),
            )
            emit(
                "agent_supervisor_frozen", "gauge",
                "1 while the agent is frozen.",
                ((_labels(task=t), 1.0 if f else 0.0) for t, f in self._frozen.items()),
            )
            emit(
                "agent_supervisor_run_cost", "gauge",
                "Cost accumulated by the current run.",
                ((_labels(task=t), v) for t, v in self._cost.items()),
            )
            emit(
                "agent_supervisor_cost_saved_total", "counter",
                "Estimated spend prevented by freezing.",
                ((_labels(task=t), v) for t, v in self._saved.items()),
            )

        return "\n".join(out) + "\n"


_DASHBOARD_CACHE: Optional[bytes] = None


def dashboard_html() -> bytes:
    """The operator dashboard, as bytes. Read once, then cached.

    A static page served from the same endpoint as the metrics, so the
    supervisor has an operator view with no extra process to run and no
    frontend toolchain. It reads `/metrics` in the browser and parses the
    exposition format there — the same format Prometheus reads, so there is
    no second source of truth to drift.
    """
    global _DASHBOARD_CACHE
    if _DASHBOARD_CACHE is None:
        if _res_files is not None:
            _DASHBOARD_CACHE = (
                _res_files(__package__).joinpath("dashboard.html").read_bytes()
            )
        else:  # pragma: no cover
            import os

            path = os.path.join(os.path.dirname(__file__), "dashboard.html")
            with open(path, "rb") as fh:
                _DASHBOARD_CACHE = fh.read()
    return _DASHBOARD_CACHE


class _Handler(BaseHTTPRequestHandler):
    registry: MetricsRegistry

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/metrics":
            self._send(
                200,
                self.registry.render().encode(),
                "text/plain; version=0.0.4; charset=utf-8",
            )
        elif path in ("", "/ui"):
            self._send(200, dashboard_html(), "text/html; charset=utf-8")
        elif path == "/healthz":
            self._send(200, b"ok", "text/plain; charset=utf-8")
        else:
            self._send(404, b"not found\n", "text/plain; charset=utf-8")

    def log_message(self, *args):  # silence per-request logging
        pass


def serve_metrics(
    registry: MetricsRegistry, port: int = 9464, host: str = "0.0.0.0"
) -> HTTPServer:
    """Start a scrape endpoint on a daemon thread. Returns the server."""
    handler = type("Handler", (_Handler,), {"registry": registry})
    httpd = HTTPServer((host, port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
