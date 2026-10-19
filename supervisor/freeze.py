"""
Freeze manager.

The one idea worth stealing from running an unsupervised system against real
money: the thing being supervised does not get to vouch for itself.

`attempt_unfreeze` therefore takes *evidence* and checks it here, rather than
taking a boolean from a caller who may be the very component that failed.
A frozen agent may still take reducing actions (cleaning up, releasing locks,
closing connections) because the alternative — a frozen agent holding
resources forever — is usually worse than a controlled wind-down.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


@dataclass(frozen=True)
class AuditRecord:
    ts: float
    event: str          # FREEZE | UNFREEZE_OK | UNFREEZE_DENIED
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class RecoveryEvidence:
    """What a caller must show to lift a freeze.

    `reconciled_at` is a wall-clock timestamp from the reconciliation pass,
    not a claim that reconciliation happened.
    """

    reconciled_at: Optional[float] = None
    reconcile_clean: bool = False
    unresolved_actions: int = 0
    operator_ack: bool = False


class FrozenError(RuntimeError):
    """Raised when a non-reducing action is attempted while frozen."""


class FreezeManager:
    def __init__(
        self,
        max_reconcile_age_s: float = 120.0,
        require_operator_ack: bool = False,
        max_audit: int = 10_000,
    ) -> None:
        self._lock = threading.Lock()
        self._frozen = False
        self._reason: Optional[str] = None
        self.max_reconcile_age_s = max_reconcile_age_s
        self.require_operator_ack = require_operator_ack
        self.audit: Deque[AuditRecord] = deque(maxlen=max_audit)

    # ---- state -------------------------------------------------------------

    @property
    def frozen(self) -> bool:
        with self._lock:
            return self._frozen

    @property
    def reason(self) -> Optional[str]:
        with self._lock:
            return self._reason

    # ---- transitions -------------------------------------------------------

    def freeze(self, reason: str, detail: str = "") -> None:
        """Idempotent. The FIRST reason is kept; later ones are audited only,
        so the original cause isn't overwritten by downstream symptoms."""
        with self._lock:
            if not self._frozen:
                self._frozen = True
                self._reason = reason
            self.audit.append(AuditRecord(time.time(), "FREEZE", reason, detail))

    def attempt_unfreeze(self, evidence: RecoveryEvidence) -> tuple[bool, str]:
        with self._lock:
            if not self._frozen:
                return True, "NOT_FROZEN"

            if evidence.reconciled_at is None:
                return self._deny("NEVER_RECONCILED")

            age = time.time() - evidence.reconciled_at
            if age > self.max_reconcile_age_s:
                return self._deny(f"RECONCILIATION_STALE_{age:.0f}s")

            if not evidence.reconcile_clean:
                return self._deny("RECONCILIATION_NOT_CLEAN")

            if evidence.unresolved_actions > 0:
                return self._deny(
                    f"UNRESOLVED_ACTIONS_{evidence.unresolved_actions}"
                )

            if self.require_operator_ack and not evidence.operator_ack:
                return self._deny("OPERATOR_ACK_REQUIRED")

            self._frozen = False
            self._reason = None
            self.audit.append(
                AuditRecord(time.time(), "UNFREEZE_OK", "evidence accepted")
            )
            return True, "UNFROZEN"

    def _deny(self, why: str) -> tuple[bool, str]:
        """Caller must hold the lock."""
        self.audit.append(AuditRecord(time.time(), "UNFREEZE_DENIED", why))
        return False, why

    # ---- enforcement -------------------------------------------------------

    def check(self, action: str, *, reducing: bool = False) -> None:
        """Raise if this action may not proceed.

        `reducing` marks actions that only wind things down — releasing a
        lock, closing a session, cancelling a pending request. Those stay
        permitted while frozen.
        """
        with self._lock:
            if self._frozen and not reducing:
                raise FrozenError(
                    f"frozen ({self._reason}); '{action}' is not a reducing action"
                )

    def allows(self, action: str, *, reducing: bool = False) -> bool:
        try:
            self.check(action, reducing=reducing)
            return True
        except FrozenError:
            return False
