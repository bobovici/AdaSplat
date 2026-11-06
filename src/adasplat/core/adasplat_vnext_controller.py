"""PRODUCTION VNEXT controller: one phase owner, pure diagnosis, optional oracle.
Growth and objective re-exports are compatibility imports for existing tests.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import copy


class Phase(str, Enum):
    FIT = "FIT"
    DIAGNOSE = "DIAGNOSE"
    SNAPSHOT = "SNAPSHOT"
    PROPOSE = "PROPOSE"
    BRANCH_A = "BRANCH_A"
    RESTORE_S0 = "RESTORE_S0"
    BRANCH_B = "BRANCH_B"
    COMPARE = "COMPARE"
    REJECT = "REJECT"
    COMMIT_A = "COMMIT_A"
    COMMIT_B = "COMMIT_B"
    CONSOLIDATE = "CONSOLIDATE"
    SELECT = "SELECT"
    REPLAY = "REPLAY"
    FINAL = "FINAL"


EDGES = {
    "FIT": {"FIT", "DIAGNOSE", "SELECT"},
    "DIAGNOSE": {"FIT", "SNAPSHOT", "CONSOLIDATE", "SELECT"},
    "SNAPSHOT": {"PROPOSE"},
    "PROPOSE": {"BRANCH_A", "CONSOLIDATE"},
    "BRANCH_A": {"RESTORE_S0"},
    "RESTORE_S0": {"BRANCH_B"},
    "BRANCH_B": {"COMPARE"},
    "COMPARE": {"REJECT", "COMMIT_B"},
    "REJECT": {"COMMIT_A", "CONSOLIDATE"},
    "COMMIT_A": {"CONSOLIDATE"},
    "COMMIT_B": {"CONSOLIDATE"},
    "CONSOLIDATE": {"FIT", "SELECT"},
    "SELECT": {"REPLAY"},
    "REPLAY": {"FINAL"},
    "FINAL": set(),
}


class Controller:
    def __init__(self, state=None):
        self.data = copy.deepcopy(
            state
            or dict(
                phase="FIT",
                fit_updates=0,
                probes=0,
                last_probe_fit_update=None,
                history=[],
                evidence=[],
            )
        )
        Phase(self.data["phase"])

    @property
    def phase(self):
        return self.data["phase"]

    def transition(self, target, reason):
        target = Phase(target).value
        if target not in EDGES[self.phase] or not reason:
            raise ValueError(
                "illegal controller transition: " + self.phase + " -> " + target
            )
        self.data["history"].append(
            dict(source=self.phase, target=target, reason=reason)
        )
        self.data["phase"] = target
        return copy.deepcopy(self.data)

    def recovery(self, phase):
        # Unpublished oracle work is rolled back by TrialStore. Never resume a partial
        # counterfactual as normal FIT or silently rerun an expensive oracle.
        phase = Phase(phase).value
        if phase in {
            "PROPOSE",
            "BRANCH_A",
            "RESTORE_S0",
            "BRANCH_B",
            "COMPARE",
            "REJECT",
            "COMMIT_A",
            "COMMIT_B",
            "SNAPSHOT",
        }:
            self.data["phase"] = "CONSOLIDATE"
            self.data["history"].append(
                dict(
                    source=phase,
                    target="CONSOLIDATE",
                    reason="transaction recovery; oracle not automatically repeated",
                )
            )
        else:
            self.data["phase"] = phase
        return copy.deepcopy(self.data)


@dataclass(frozen=True)
class DiagnosisInput:
    trajectory: tuple = ()
    update_rms: float | None = None
    render_change: float | None = None
    residual_summary: dict | None = None
    candidate_preview: dict | None = None
    previous_evidence: tuple = ()
    remaining_updates: int = 0
    remaining_capacity: int = 0
    probe_permitted: bool = False


def diagnose(data, config):
    """Read-only, transparent gate. Uncalibrated thresholds abstain, never grow."""
    cfg = config["diagnosis"]
    if data.remaining_updates <= 0:
        return {"action": "TERMINATE", "reason": "compute budget exhausted"}
    if data.remaining_capacity <= 0:
        return {"action": "CONSOLIDATE", "reason": "capacity budget exhausted"}
    if len(data.trajectory) < cfg["window"]:
        return {"action": "UNRESOLVED", "reason": "insufficient fixed-topology history"}
    window = data.trajectory[-cfg["window"] :]
    gain = (window[0]["MSE"] - window[-1]["MSE"]) / max(window[0]["MSE"], 1e-12)
    threshold = cfg["significant_relative_improvement"]
    if threshold is not None and gain >= threshold:
        return {
            "action": "FIT_MORE",
            "reason": "fixed topology still improving",
            "relative_gain": gain,
        }
    limits = [
        cfg["plateau_relative_improvement"],
        cfg["max_update_rms"],
        cfg["max_render_change"],
    ]
    if (
        any(x is None for x in limits)
        or data.update_rms is None
        or data.render_change is None
    ):
        return {"action": "UNRESOLVED", "reason": "diagnosis thresholds uncalibrated"}
    stable = (
        0 <= gain <= limits[0]
        and data.update_rms <= limits[1]
        and data.render_change <= limits[2]
    )
    if stable and data.probe_permitted:
        return {
            "action": "PROBE_CAPACITY",
            "reason": "configured plateau and update/render gates; oracle permitted",
        }
    return {"action": "FIT_MORE", "reason": "probe gates not satisfied"}


# Backwards-compatible function imports; these modules contain no old policies.
from adasplat_vnext_growth import (
    micro_budget,
    core_grid,
    propose_from_residual,
    propose_micro_growth,
    apply_micro_proposal,
    CohortBank,
    zero_color_bridge,
    bank_optimizer_step,
)
from adasplat_vnext_objective import unified_loss, control_metrics, capacity_decision
from adasplat_vnext_oracle import run_capacity_probe, run_single_trial, proposal_regions
