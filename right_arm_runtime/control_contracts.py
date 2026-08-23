"""Lightweight typed contracts shared by simulation and hardware adapters.

This module intentionally imports no predictor, MuJoCo, Unitree SDK, DDS, or
output implementation.  Keeping task-clock and state-capability identity here
prevents either platform adapter from becoming the owner of shared semantics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskClockEvent:
    """Authoritative locomotion event; never inferred from a state stream."""

    session_nonce: str
    task_epoch_id: str
    producer_sequence: int
    event_monotonic_timestamp_ns: int
    source_sample_id: int
    task_time_ns: int
    full_task_anchor: int
    planned_command_vx_vy_wz: tuple[float, float, float]
    runtime_command_vx_vy_wz: tuple[float, float, float]
    heading_reference_rad: float


@dataclass(frozen=True)
class ControlStateCapabilities:
    """Validity of state components; unknown quantities remain explicit."""

    right_arm_joint_state: bool
    torso_rotation: bool
    torso_angular_velocity: bool
    torso_linear_acceleration: bool
    torso_angular_acceleration: bool
    floating_base_translation: bool = False
    floating_base_velocity: bool = False
    foot_contacts: bool = False
    external_forces: bool = False

    @property
    def mpc_observation_complete(self) -> bool:
        return bool(
            self.right_arm_joint_state
            and self.torso_rotation
            and self.torso_angular_velocity
            and self.torso_linear_acceleration
            and self.torso_angular_acceleration
        )

    @property
    def hardware_torque_state_complete(self) -> bool:
        """Whether all state families needed for a hardware torque claim exist."""

        return bool(
            self.mpc_observation_complete
            and self.floating_base_translation
            and self.floating_base_velocity
            and self.foot_contacts
            and self.external_forces
        )


__all__ = ("ControlStateCapabilities", "TaskClockEvent")
