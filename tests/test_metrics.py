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


if __name__ == "__main__":
    unittest.main(verbosity=2)
