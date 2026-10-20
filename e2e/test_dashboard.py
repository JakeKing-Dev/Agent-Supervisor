"""
End-to-end tests for the operator dashboard.

Kept out of `tests/` on purpose. The library has no runtime dependencies and
the unit suite has no test dependencies either; Playwright is needed only
here, so `python3 -m unittest discover -s tests` still runs on a bare
interpreter. CI runs this suite as a separate job.

What these actually assert: that the page is driven by live metrics, not that
it renders. A dashboard that looks right while showing stale or invented
numbers is worse than no dashboard — an operator deciding whether to halt a
fleet is relying on it. So each test seeds a known supervisor state, serves
it, and checks that what the browser shows matches it, including after the
state changes underneath an already-open page.

    pip install playwright && playwright install chromium
    PYTHONPATH=. python3 -m unittest discover -s e2e
"""

from __future__ import annotations

import unittest

from playwright.sync_api import expect, sync_playwright

from supervisor import (
    Decision,
    MetricsRegistry,
    Step,
    Supervisor,
    SupervisorPolicy,
    serve_metrics,
)

TIMEOUT = 8_000  # ms; the page polls every 2s, so assertions need headroom


def healthy_run(registry: MetricsRegistry, task: str, steps: int = 12) -> Supervisor:
    """An agent making genuine progress: every action distinct."""
    sup = Supervisor(task=task, metrics=registry)
    for i in range(steps):
        sup.observe(
            Step(
                index=i,
                action="read_file",
                args={"path": f"/docs/page-{i}.md"},
                ok=True,
                observation_digest=f"state-{i}",
                cost=0.02,
            )
        )
    return sup


def looping_run(registry: MetricsRegistry, task: str) -> Supervisor:
    """An agent stuck retrying one failing call. Should end frozen."""
    sup = Supervisor(
        policy=SupervisorPolicy(persist_steps=3),
        task=task,
        metrics=registry,
    )
    for i in range(20):
        v = sup.observe(
            Step(
                index=i,
                action="issue_refund",
                args={"order": "A-4471"},
                ok=False,
                error="HTTP 502",
                observation_digest="unchanged",
                cost=0.08,
            )
        )
        if v.decision is Decision.FREEZE:
            break
    return sup


class DashboardTest(unittest.TestCase):
    """Base: a metrics server on an ephemeral port and a headless browser."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._pw = sync_playwright().start()
        cls._browser = cls._pw.chromium.launch()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._browser.close()
        cls._pw.stop()

    def setUp(self) -> None:
        self.registry = MetricsRegistry()
        self.httpd = serve_metrics(self.registry, port=0, host="127.0.0.1")
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.context = self._browser.new_context()
        self.page = self.context.new_page()

    def stop_server(self) -> None:
        """Stop serving *and* release the socket.

        `shutdown()` alone only ends the serve loop; the listening socket
        stays bound, so a browser request is accepted and then hangs rather
        than failing. Closing it is what makes the client see the outage.
        """
        if getattr(self, "_stopped", False):
            return
        self._stopped = True
        self.httpd.shutdown()
        self.httpd.server_close()

    def tearDown(self) -> None:
        self.context.close()
        self.stop_server()

    def row(self, task: str):
        return self.page.locator(f'[data-testid="agent-row"][data-task="{task}"]')


class TestPageLoads(DashboardTest):
    def test_served_at_root_and_at_ui(self):
        for path in ("/", "/ui"):
            res = self.page.goto(self.base + path)
            self.assertEqual(res.status, 200, path)
            self.assertIn("text/html", res.header_value("content-type"))
            expect(self.page.locator("h1")).to_have_text("Agent Supervisor")

    def test_metrics_endpoint_still_serves_prometheus_format(self):
        """The dashboard took over `/`, so guard the scrape path explicitly."""
        healthy_run(self.registry, "scrape-check")
        res = self.page.request.get(self.base + "/metrics")
        self.assertEqual(res.status, 200)
        self.assertIn("version=0.0.4", res.headers["content-type"])
        self.assertIn("# TYPE agent_supervisor_steps_total counter", res.text())

    def test_reaches_live_state(self):
        self.page.goto(self.base)
        expect(self.page.get_by_test_id("status")).to_have_attribute(
            "data-state", "live", timeout=TIMEOUT
        )

    def test_empty_state_when_nothing_observed(self):
        self.page.goto(self.base)
        expect(self.page.get_by_test_id("empty")).to_be_visible(timeout=TIMEOUT)
        expect(self.page.get_by_test_id("agent-row")).to_have_count(0)
        expect(self.page.get_by_test_id("total-steps")).to_have_text("0")


class TestHealthyAgent(DashboardTest):
    def setUp(self) -> None:
        super().setUp()
        self.sup = healthy_run(self.registry, "summarise-inbox")
        self.page.goto(self.base)

    def test_renders_as_healthy_and_not_frozen(self):
        row = self.row("summarise-inbox")
        expect(row).to_be_visible(timeout=TIMEOUT)
        expect(row.get_by_test_id("regime")).to_have_attribute("data-regime", "healthy")
        expect(row.get_by_test_id("frozen")).to_have_text("—")

    def test_drift_score_matches_the_supervisor(self):
        expected = f"{self.sup.history[-1].assessment.score:.2f}"
        expect(
            self.row("summarise-inbox").get_by_test_id("score")
        ).to_have_text(expected, timeout=TIMEOUT)

    def test_step_count_and_empty_state_agree_with_reality(self):
        expect(self.page.get_by_test_id("total-steps")).to_have_text(
            "12", timeout=TIMEOUT
        )
        expect(self.page.get_by_test_id("empty")).to_be_hidden()


class TestFrozenAgent(DashboardTest):
    def setUp(self) -> None:
        super().setUp()
        self.sup = looping_run(self.registry, "refund-A-4471")
        self.assertTrue(self.sup.frozen, "fixture must actually freeze")
        self.page.goto(self.base)

    def test_renders_as_looping_and_frozen(self):
        row = self.row("refund-A-4471")
        expect(row).to_be_visible(timeout=TIMEOUT)
        expect(row.get_by_test_id("regime")).to_have_attribute("data-regime", "looping")
        expect(row.get_by_test_id("frozen")).to_have_text("FROZEN")

    def test_halt_is_counted_and_prevented_spend_reported(self):
        expect(self.page.get_by_test_id("total-freezes")).to_have_text(
            "1", timeout=TIMEOUT
        )
        expect(self.page.get_by_test_id("total-frozen")).to_have_text("1")
        saved = self.page.get_by_test_id("total-saved").inner_text()
        self.assertRegex(saved, r"^\$\d+\.\d{2}$")
        self.assertGreater(
            float(saved.lstrip("$")),
            0.0,
            "a halted run must report non-zero prevented spend",
        )

    def test_score_is_rendered_at_the_danger_level(self):
        fill = self.row("refund-A-4471").locator(".bar > i")
        expect(fill).to_have_attribute("data-level", "bad", timeout=TIMEOUT)


class TestLiveUpdates(DashboardTest):
    def test_a_degrading_agent_turns_frozen_without_a_reload(self):
        """The point of the page: an operator watching it sees the halt."""
        sup = Supervisor(
            policy=SupervisorPolicy(persist_steps=3),
            task="checkout-flow",
            metrics=self.registry,
        )
        sup.observe(
            Step(index=0, action="open_cart", ok=True, observation_digest="a", cost=0.05)
        )
        sup.observe(
            Step(index=1, action="apply_coupon", ok=True, observation_digest="b", cost=0.05)
        )

        self.page.goto(self.base)
        row = self.row("checkout-flow")
        expect(row).to_be_visible(timeout=TIMEOUT)
        expect(row.get_by_test_id("frozen")).to_have_text("—")

        # Now the agent degrades, with the page already open.
        for i in range(2, 20):
            v = sup.observe(
                Step(
                    index=i,
                    action="submit_payment",
                    args={"attempt": "retry"},
                    ok=False,
                    error="HTTP 502",
                    observation_digest="unchanged",
                    cost=0.05,
                )
            )
            if v.decision is Decision.FREEZE:
                break
        self.assertTrue(sup.frozen, "fixture must actually freeze")

        expect(row.get_by_test_id("frozen")).to_have_text("FROZEN", timeout=TIMEOUT)
        expect(row.get_by_test_id("regime")).to_have_attribute("data-regime", "looping")

    def test_reports_an_error_when_metrics_go_away(self):
        """Fail visibly rather than showing numbers that have stopped moving."""
        healthy_run(self.registry, "doomed")
        self.page.goto(self.base)
        expect(self.page.get_by_test_id("status")).to_have_attribute(
            "data-state", "live", timeout=TIMEOUT
        )

        self.stop_server()
        expect(self.page.get_by_test_id("status")).to_have_attribute(
            "data-state", "error", timeout=TIMEOUT
        )


class TestMultipleAgents(DashboardTest):
    def test_worst_agent_sorts_first(self):
        healthy_run(self.registry, "aaa-healthy")
        looping_run(self.registry, "zzz-looping")
        self.page.goto(self.base)

        rows = self.page.get_by_test_id("agent-row")
        expect(rows).to_have_count(2, timeout=TIMEOUT)
        self.assertEqual(rows.nth(0).get_attribute("data-task"), "zzz-looping")
        self.assertEqual(rows.nth(1).get_attribute("data-task"), "aaa-healthy")


class TestSmallScreen(DashboardTest):
    def test_no_horizontal_overflow_at_phone_width(self):
        self.page.set_viewport_size({"width": 390, "height": 780})
        looping_run(self.registry, "refund-A-4471")
        self.page.goto(self.base)
        expect(self.row("refund-A-4471")).to_be_visible(timeout=TIMEOUT)

        overflow = self.page.evaluate(
            "() => document.documentElement.scrollWidth - window.innerWidth"
        )
        self.assertLessEqual(overflow, 1, "page scrolls sideways on a phone")

    def test_regime_and_freeze_state_stay_visible_when_columns_collapse(self):
        self.page.set_viewport_size({"width": 390, "height": 780})
        looping_run(self.registry, "refund-A-4471")
        self.page.goto(self.base)
        row = self.row("refund-A-4471")
        expect(row.get_by_test_id("regime")).to_be_visible(timeout=TIMEOUT)
        expect(row.get_by_test_id("frozen")).to_be_visible()


if __name__ == "__main__":
    unittest.main()
