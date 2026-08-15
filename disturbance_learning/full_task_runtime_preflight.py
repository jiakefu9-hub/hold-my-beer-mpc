"""Fail-fast runtime contract for formal full-task control experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, Sequence


FORMAL_FULL_TASK_RUNTIME_VERSION = "full_task_cpu7_single_thread_v1"
FORMAL_CONTROL_CPU = 7
RUN_SH_LAUNCH_MARKER = "disturbance_lab_run_sh"
THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


class FormalFullTaskEnvironmentError(RuntimeError):
    """The formal experiment environment is not the frozen CPU-7 setup."""


def _normalized_affinity(value: Sequence[int]) -> list[int]:
    return sorted({int(cpu) for cpu in value})


@dataclass(frozen=True)
class FormalFullTaskRuntimeEvidence:
    protocol_version: str
    launcher: str
    requested_control_cpu: str
    parent_cpu_affinity: list[int]
    worker_cpu_affinity: list[int] | None
    thread_environment: dict[str, str]
    torch_num_threads: int
    torch_num_interop_threads: int
    gc_disabled_during_control: bool | None
    dynamic_arming_enabled: bool
    startup_pd_duration_s: float
    mpc_handoff_anchor_index: int
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "launcher": self.launcher,
            "requested_control_cpu": self.requested_control_cpu,
            "parent_cpu_affinity": self.parent_cpu_affinity,
            "worker_cpu_affinity": self.worker_cpu_affinity,
            "thread_environment": self.thread_environment,
            "torch_num_threads": self.torch_num_threads,
            "torch_num_interop_threads": self.torch_num_interop_threads,
            "gc_disabled_during_control": self.gc_disabled_during_control,
            "dynamic_arming_enabled": self.dynamic_arming_enabled,
            "startup_pd_duration_s": self.startup_pd_duration_s,
            "mpc_handoff_anchor_index": self.mpc_handoff_anchor_index,
            "passed": self.passed,
        }


def validate_formal_full_task_runtime(
    *,
    parent_affinity: Sequence[int] | None = None,
    worker_affinity: Sequence[int] | None = None,
    environment: Mapping[str, str] | None = None,
    torch_num_threads: int,
    torch_num_interop_threads: int,
    gc_disabled_during_control: bool | None = None,
) -> FormalFullTaskRuntimeEvidence:
    """Validate the frozen launcher, affinity, threading, and startup mode.

    ``worker_affinity`` and ``gc_disabled_during_control`` are optional only for
    the early parent-side check.  The final pre-step check supplies both and is
    therefore the evidence written beside every formal rollout.
    """

    env = os.environ if environment is None else environment
    affinity = _normalized_affinity(
        os.sched_getaffinity(0) if parent_affinity is None else parent_affinity
    )
    errors: list[str] = []
    launcher = str(env.get("DISTURBANCE_LAB_FORMAL_LAUNCHER", ""))
    requested_cpu = str(env.get("MPC_CONTROL_CPU", ""))
    if launcher != RUN_SH_LAUNCH_MARKER:
        errors.append("formal full-task experiments must be launched through run.sh")
    if requested_cpu != str(FORMAL_CONTROL_CPU):
        errors.append("MPC_CONTROL_CPU must be explicitly set to 7")
    if affinity != [FORMAL_CONTROL_CPU]:
        errors.append(f"parent affinity must equal [7], got {affinity}")

    thread_environment = {
        name: str(env.get(name, "")) for name in THREAD_ENVIRONMENT_NAMES
    }
    invalid_threads = {
        name: value for name, value in thread_environment.items() if value != "1"
    }
    if invalid_threads:
        errors.append(f"numeric thread environment must all equal 1: {invalid_threads}")
    if int(torch_num_threads) != 1:
        errors.append(f"torch intra-op threads must equal 1, got {torch_num_threads}")
    if int(torch_num_interop_threads) != 1:
        errors.append(
            f"torch inter-op threads must equal 1, got {torch_num_interop_threads}"
        )

    normalized_worker = None
    if worker_affinity is not None:
        normalized_worker = _normalized_affinity(worker_affinity)
        if normalized_worker != [FORMAL_CONTROL_CPU]:
            errors.append(f"C++ worker affinity must equal [7], got {normalized_worker}")
        if normalized_worker != affinity:
            errors.append("parent and C++ worker affinities differ")
    if gc_disabled_during_control is not None and not bool(
        gc_disabled_during_control
    ):
        errors.append("Python GC must be disabled before the control loop")

    if errors:
        raise FormalFullTaskEnvironmentError(
            "formal full-task runtime preflight failed: " + "; ".join(errors)
        )
    return FormalFullTaskRuntimeEvidence(
        protocol_version=FORMAL_FULL_TASK_RUNTIME_VERSION,
        launcher=launcher,
        requested_control_cpu=requested_cpu,
        parent_cpu_affinity=affinity,
        worker_cpu_affinity=normalized_worker,
        thread_environment=thread_environment,
        torch_num_threads=int(torch_num_threads),
        torch_num_interop_threads=int(torch_num_interop_threads),
        gc_disabled_during_control=gc_disabled_during_control,
        dynamic_arming_enabled=False,
        startup_pd_duration_s=0.024,
        mpc_handoff_anchor_index=4,
        passed=True,
    )
