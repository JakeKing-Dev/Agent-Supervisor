import time
import unittest

from supervisor import (
    DetectorConfig,
    Decision,
    DriftDetector,
    FreezeManager,
    FrozenError,
    RecoveryEvidence,
    Regime,
    Step,
    Supervisor,
    SupervisorPolicy,
    Trajectory,
)


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.d = DriftDetector()

    def test_refuses_to_judge_short_runs(self):
        t = Trajectory.from_actions(["a", "b"])
        a = self.d.assess(t)
        self.assertIs(a.regime, Regime.UNKNOWN)
        self.assertEqual(a.score, 0.0)

    def test_healthy_varied_run(self):
        t = Trajectory.from_actions(["a", "b", "c", "d", "e", "f"])
        a = self.d.assess(t)
        self.assertIs(a.regime, Regime.HEALTHY)

    def test_consecutive_repetition_is_looping(self):
        t = Trajectory.from_actions(["a", "b", "c", "c", "c", "c"])
        a = self.d.assess(t)
        self.assertIs(a.regime, Regime.LOOPING)
        self.assertGreater(a.score, 0.5)

    def test_alternating_cycle_is_looping(self):
        t = Trajectory.from_actions(["x", "y", "x", "y", "x", "y"])
        a = self.d.assess(t)
        self.assertIs(a.regime, Regime.LOOPING)
        self.assertIsNotNone(a.signals.cycle_period)

    def test_args_distinguish_actions(self):
        """click(1), click(2)... is progress; click(1) x4 is a loop."""
        t = Trajectory()
        for i in range(6):
            t.add(Step(index=i, action="click", args={"x": i}))
        self.assertIs(self.d.assess(t).regime, Regime.HEALTHY)

        t2 = Trajectory()
        for i in range(6):
            t2.add(Step(index=i, action="click", args={"x": 1}))
        self.assertIs(self.d.assess(t2).regime, Regime.LOOPING)

    def test_stall_detected_from_unchanged_observations(self):
        t = Trajectory()
        for i, a in enumerate(["a", "b", "c", "d", "e"]):
            t.add(Step(index=i, action=a, observation_digest="same"))
        self.assertIs(self.d.assess(t).regime, Regime.STALLED)

    def test_rising_errors_are_wandering(self):
        t = Trajectory()
        for i in range(8):
            t.add(Step(index=i, action=f"a{i}", ok=True))
        for i in range(8, 16):
            t.add(Step(index=i, action=f"b{i}", ok=False, error="HTTP500"))
        a = self.d.assess(t)
        self.assertIn(a.regime, (Regime.WANDERING, Regime.LOOPING))
        self.assertGreater(a.signals.error_rate, 0.9)

    def test_first_alarm_is_early(self):
        actions = ["start", "search", "open"] + ["retry"] * 10
        t = Trajectory.from_actions(actions)
        idx = self.d.first_alarm(t)
        self.assertIsNotNone(idx)
        self.assertLess(idx, 7, "should fire well before the run ends")


class TestFreezeManager(unittest.TestCase):
    def test_freeze_blocks_normal_actions(self):
        f = FreezeManager()
        f.freeze("LOOPING")
        with self.assertRaises(FrozenError):
            f.check("http_post")

    def test_reducing_actions_allowed_while_frozen(self):
        f = FreezeManager()
        f.freeze("LOOPING")
        f.check("release_lock", reducing=True)  # must not raise
        self.assertTrue(f.allows("close_session", reducing=True))

    def test_first_reason_is_kept(self):
        f = FreezeManager()
        f.freeze("LOOPING")
        f.freeze("BUDGET_EXCEEDED")
        self.assertEqual(f.reason, "LOOPING")
        self.assertEqual(len(f.audit), 2)

    def test_unfreeze_requires_evidence(self):
        f = FreezeManager()
        f.freeze("LOOPING")

        ok, why = f.attempt_unfreeze(RecoveryEvidence())
        self.assertFalse(ok)
        self.assertEqual(why, "NEVER_RECONCILED")

        ok, why = f.attempt_unfreeze(
            RecoveryEvidence(reconciled_at=time.time() - 9999, reconcile_clean=True)
        )
        self.assertFalse(ok)
        self.assertIn("STALE", why)

        ok, why = f.attempt_unfreeze(
            RecoveryEvidence(reconciled_at=time.time(), reconcile_clean=False)
        )
        self.assertFalse(ok)
        self.assertEqual(why, "RECONCILIATION_NOT_CLEAN")

        ok, why = f.attempt_unfreeze(
            RecoveryEvidence(
                reconciled_at=time.time(), reconcile_clean=True, unresolved_actions=2
            )
        )
        self.assertFalse(ok)
        self.assertIn("UNRESOLVED", why)

        ok, why = f.attempt_unfreeze(
            RecoveryEvidence(reconciled_at=time.time(), reconcile_clean=True)
        )
        self.assertTrue(ok)
        self.assertFalse(f.frozen)

    def test_operator_ack_can_be_required(self):
        f = FreezeManager(require_operator_ack=True)
        f.freeze("LOOPING")
        ev = RecoveryEvidence(reconciled_at=time.time(), reconcile_clean=True)
        self.assertFalse(f.attempt_unfreeze(ev)[0])
        ev2 = RecoveryEvidence(
            reconciled_at=time.time(), reconcile_clean=True, operator_ack=True
        )
        self.assertTrue(f.attempt_unfreeze(ev2)[0])


class TestSupervisor(unittest.TestCase):
    def test_persistence_prevents_hair_trigger_freeze(self):
        """A run that repeats briefly then recovers must not be frozen."""
        sup = Supervisor(SupervisorPolicy(persist_steps=3))
        actions = ["a", "b", "c", "c", "c", "d", "e", "f", "g"]
        froze = False
        for i, a in enumerate(actions):
            if sup.observe(Step(index=i, action=a)).decision is Decision.FREEZE:
                froze = True
        self.assertFalse(froze, "brief repetition should not halt a recovering run")

    def test_sustained_loop_is_frozen(self):
        sup = Supervisor(SupervisorPolicy(persist_steps=3))
        frozen_at = None
        for i, a in enumerate(["a", "b", "c"] + ["stuck"] * 12):
            if sup.observe(Step(index=i, action=a)).decision is Decision.FREEZE:
                frozen_at = i
                break
        self.assertIsNotNone(frozen_at)
        self.assertTrue(sup.frozen)

    def test_budget_ceiling_freezes_immediately(self):
        sup = Supervisor(SupervisorPolicy(max_cost=1.0))
        v = None
        for i in range(5):
            v = sup.observe(Step(index=i, action=f"a{i}", cost=0.5))
            if v.decision is Decision.FREEZE:
                break
        self.assertIs(v.decision, Decision.FREEZE)
        self.assertEqual(sup.freezer.reason, "BUDGET_EXCEEDED")

    def test_replay_reports_freeze_point(self):
        sup = Supervisor(SupervisorPolicy(persist_steps=2))
        t = Trajectory.from_actions(["a", "b", "c"] + ["loop"] * 10)
        idx = sup.replay(t)
        self.assertIsNotNone(idx)
        self.assertLess(idx, 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
