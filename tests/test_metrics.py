import unittest
import urllib.request

from supervisor import (
    Decision,
    MetricsRegistry,
    Step,
    Supervisor,
    SupervisorPolicy,
    serve_metrics,
)


def _run(sup, actions):
    for i, a in enumerate(actions):
        v = sup.observe(Step(index=i, action=a, cost=0.1))
        if v.decision is Decision.FREEZE:
            return i
    return None


class TestMetrics(unittest.TestCase):
    def test_counters_and_gauges_render(self):
        reg = MetricsRegistry()
        sup = Supervisor(SupervisorPolicy(persist_steps=2), task="demo", metrics=reg)
        _run(sup, ["a", "b", "c"] + ["stuck"] * 8)
        text = reg.render()

        self.assertIn("agent_supervisor_steps_total", text)
        self.assertIn('agent_supervisor_freezes_total{reason="looping",task="demo"}', text)
        self.assertIn('agent_supervisor_frozen{task="demo"} 1.0', text)
        self.assertIn("# TYPE agent_supervisor_drift_score gauge", text)

    def test_regime_gauge_is_one_hot(self):
        reg = MetricsRegistry()
        sup = Supervisor(SupervisorPolicy(persist_steps=2), task="t", metrics=reg)
        _run(sup, ["a", "b", "c"] + ["stuck"] * 8)
        lines = [l for l in reg.render().splitlines() if l.startswith("agent_supervisor_regime")]
        hot = [l for l in lines if l.endswith(" 1.0")]
        self.assertEqual(len(hot), 1, f"expected exactly one active regime, got {hot}")

    def test_label_values_are_escaped(self):
        reg = MetricsRegistry()
        sup = Supervisor(task='weird "task"\nname', metrics=reg)
        sup.observe(Step(index=0, action="a"))
        text = reg.render()
        self.assertIn('\\"task\\"', text)
        self.assertNotIn("\nname", text.split("agent_supervisor_steps_total")[1][:40])

    def test_no_metrics_registry_is_fine(self):
        sup = Supervisor(task="t")  # metrics=None
        sup.observe(Step(index=0, action="a"))  # must not raise

    def test_http_endpoint_serves_metrics(self):
        reg = MetricsRegistry()
        sup = Supervisor(SupervisorPolicy(persist_steps=2), task="http", metrics=reg)
        _run(sup, ["a", "b", "c"] + ["stuck"] * 8)
        httpd = serve_metrics(reg, port=0, host="127.0.0.1")
        try:
            port = httpd.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
                self.assertEqual(r.status, 200)
                self.assertIn("text/plain", r.headers["Content-Type"])
                body = r.read().decode()
            self.assertIn("agent_supervisor_steps_total", body)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as r:
                self.assertEqual(r.read(), b"ok")
        finally:
            httpd.shutdown()
            httpd.server_close()


def _value(body: str, name: str) -> float:
    """Sum one series across its label sets."""
    total = 0.0
    for line in body.splitlines():
        if line.startswith("#") or not line.startswith(name):
            continue
        head, _, val = line.rpartition(" ")
        if head == name or head.startswith(name + "{"):
            total += float(val)
    return total


class TestHaltAccounting(unittest.TestCase):
    """A halt is one event, however long the caller keeps observing it."""

    def _frozen_run(self, extra_steps: int) -> str:
        reg = MetricsRegistry()
        sup = Supervisor(
            SupervisorPolicy(persist_steps=2, savings_horizon=20),
            task="loop",
            metrics=reg,
        )
        actions = ["a", "b"] + ["stuck"] * (6 + extra_steps)
        for i, a in enumerate(actions):
            sup.observe(Step(index=i, action=a, cost=0.1))
        self.assertTrue(sup.frozen, "fixture must freeze")
        return reg.render()

    def test_freeze_counted_once_however_many_verdicts_follow(self):
        self.assertEqual(_value(self._frozen_run(0), "agent_supervisor_freezes_total"), 1.0)
        self.assertEqual(_value(self._frozen_run(10), "agent_supervisor_freezes_total"), 1.0)

    def test_prevented_spend_is_not_re_added_on_repeat_freezes(self):
        short = _value(self._frozen_run(0), "agent_supervisor_cost_saved_total")
        long = _value(self._frozen_run(10), "agent_supervisor_cost_saved_total")
        self.assertGreater(short, 0.0, "a halt must report prevented spend")
        self.assertAlmostEqual(short, long, places=6)

    def test_prevented_spend_reflects_the_run_rate(self):
        """0.1 per step over a 20-step horizon is about $2."""
        saved = _value(self._frozen_run(0), "agent_supervisor_cost_saved_total")
        self.assertAlmostEqual(saved, 2.0, places=6)

    def test_horizon_of_zero_reports_nothing_rather_than_an_estimate(self):
        reg = MetricsRegistry()
        sup = Supervisor(
            SupervisorPolicy(persist_steps=2, savings_horizon=0),
            task="loop",
            metrics=reg,
        )
        for i, a in enumerate(["a", "b"] + ["stuck"] * 6):
            sup.observe(Step(index=i, action=a, cost=0.1))
        self.assertTrue(sup.frozen)
        self.assertEqual(_value(reg.render(), "agent_supervisor_cost_saved_total"), 0.0)


class TestDashboardRoute(unittest.TestCase):
    def test_dashboard_served_at_root_metrics_still_at_metrics(self):
        reg = MetricsRegistry()
        sup = Supervisor(SupervisorPolicy(persist_steps=2), task="ui", metrics=reg)
        _run(sup, ["a", "b", "c"])
        httpd = serve_metrics(reg, port=0, host="127.0.0.1")
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            with urllib.request.urlopen(base + "/", timeout=5) as r:
                self.assertIn("text/html", r.headers["Content-Type"])
                self.assertIn("Agent Supervisor", r.read().decode())
            with urllib.request.urlopen(base + "/metrics", timeout=5) as r:
                self.assertIn("version=0.0.4", r.headers["Content-Type"])
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
