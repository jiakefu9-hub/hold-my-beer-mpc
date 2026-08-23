"""L1-A/B contracts for simulation-only MPC-result latency."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from right_arm_runtime.sim_mpc_latency import (
    FixedMpcResultDelayLine,
    MpcLatencyTraceRecorder,
    load_and_validate_trace,
    parse_mapper_candidate_trace,
    validate_experimental_latency_cli,
)


class FixedMpcResultDelayLineTest(unittest.TestCase):
    @staticmethod
    def vector(value: float, size: int = 5) -> np.ndarray:
        return np.full(size, value, dtype=np.float64)

    def line(self, delay: float) -> FixedMpcResultDelayLine:
        return FixedMpcResultDelayLine(
            step_dt=0.002, requested_delay_s=delay, mpc_dt=0.006
        )

    def publish(self, line, *, sample=12, anchor=4, value=1.0):
        return line.publish(
            source_time=sample * 0.002,
            source_sample_index=sample,
            source_anchor_index=anchor,
            q_ref=self.vector(value),
            dq_ref=self.vector(value + 1),
            ddq_raw=self.vector(value + 2),
            ddq_des=self.vector(value + 3),
            source_right_arm_q=self.vector(value + 4),
            source_right_arm_dq=self.vector(value + 5),
            source_torso_acc=self.vector(value + 6, 3),
            source_torso_omega=self.vector(value + 7, 3),
            diagnostics={"anchor": anchor, "values": self.vector(value)},
            packet_ready_wall_ns=1_000_000 + anchor,
        )

    def test_exact_0_2_4ms_activation_and_no_synthetic_packet(self):
        for delay, expected_sample in ((0.0, 12), (0.002, 13), (0.004, 14)):
            with self.subTest(delay=delay):
                line = self.line(delay)
                self.assertIsNone(line.activate_ready(now=0.0, sample_index=0).packet)
                packet = self.publish(line)
                for sample in range(12, expected_sample):
                    self.assertIsNone(
                        line.activate_ready(
                            now=sample * 0.002, sample_index=sample
                        ).packet
                    )
                activation = line.activate_ready(
                    now=expected_sample * 0.002, sample_index=expected_sample
                )
                self.assertTrue(activation.activated)
                self.assertEqual(activation.packet.command_id, packet.command_id)
                self.assertAlmostEqual(
                    activation.effective_delay, delay, places=12
                )

    def test_quantization_boundaries_and_six_ms_rejected(self):
        self.assertEqual(self.line(0.001999).delay_ticks, 1)
        self.assertEqual(self.line(0.002).delay_ticks, 1)
        self.assertEqual(self.line(0.002000001).delay_ticks, 2)
        self.assertEqual(self.line(0.004).delay_ticks, 2)
        with self.assertRaises(ValueError):
            self.line(0.006)

    def test_packet_is_copied_read_only_and_ids_are_consecutive(self):
        line = self.line(0.002)
        source = self.vector(1.0)
        packet = line.publish(
            source_time=0.0,
            source_sample_index=0,
            source_anchor_index=0,
            q_ref=source,
            dq_ref=source,
            ddq_raw=source,
            ddq_des=source,
            source_right_arm_q=source,
            source_right_arm_dq=source,
            source_torso_acc=np.ones(3),
            source_torso_omega=np.ones(3),
            diagnostics={"values": source},
            packet_ready_wall_ns=10,
        )
        source.fill(9.0)
        np.testing.assert_array_equal(packet.q_ref, np.ones(5))
        with self.assertRaises(ValueError):
            packet.q_ref[0] = 2.0
        next_packet = self.publish(line, sample=3, anchor=1)
        self.assertEqual(next_packet.command_id, packet.command_id + 1)
        with self.assertRaises(ValueError):
            self.publish(line, sample=9, anchor=3)

    def test_mapper_refresh_is_activation_then_four_ms(self):
        line = self.line(0.002)
        self.publish(line)
        activation = line.activate_ready(now=0.026, sample_index=13)
        self.assertTrue(
            line.mapping_update_due(
                sample_index=13, activated=activation.activated,
                mode="twice_per_interval",
            )
        )
        self.assertFalse(
            line.mapping_update_due(
                sample_index=14, activated=False, mode="twice_per_interval"
            )
        )
        self.assertTrue(
            line.mapping_update_due(
                sample_index=15, activated=False, mode="twice_per_interval"
            )
        )

    def test_time_backwards_and_task_boundaries(self):
        line = self.line(0.004)
        self.publish(line, sample=3198, anchor=1066)  # source 6.396 s
        activation = line.activate_ready(now=6.400, sample_index=3200)
        self.assertEqual(activation.packet.source_anchor_index, 1066)
        self.publish(line, sample=3201, anchor=1067)  # source 6.402 s
        activation = line.activate_ready(now=6.406, sample_index=3203)
        self.assertEqual(activation.packet.source_anchor_index, 1067)
        with self.assertRaises(ValueError):
            line.activate_ready(now=6.404, sample_index=3202)

        tail = self.line(0.004)
        self.publish(tail, sample=3999, anchor=1333)  # source 7.998 s
        activation = tail.activate_ready(now=8.002, sample_index=4001)
        self.assertEqual(activation.packet.source_time, 7.998)


class MpcLatencyTraceTest(unittest.TestCase):
    def test_no_safe_torque_candidate_trace_parser(self):
        trace = parse_mapper_candidate_trace(
            "C++执行失败[NO_SAFE_TORQUE]: ddq_torque_mapper_compute: "
            "NO_SAFE_TORQUE|CT1;B=53;BEST=12;HL=14.5;SH=78.5;"
            "MIN=R2@0.125:8.75;F@1=11/12;R2@0.125=8.75/10.6"
        )
        self.assertEqual(trace["schema_version"], "mapper_candidate_trace_v1")
        self.assertEqual(
            trace["minimum_predicted_candidate_type"],
            "rescue_2_scale_0.125",
        )
        self.assertAlmostEqual(
            trace["minimum_predicted_max_abs_qacc_rad_s2"], 8.75
        )
        self.assertEqual(len(trace["predicted_candidates"]), 2)
        self.assertAlmostEqual(
            trace["predicted_candidates"][1]["validated_max_abs_qacc_rad_s2"],
            10.6,
        )
        self.assertIsNone(parse_mapper_candidate_trace("ordinary error"))

    def test_schema_checksum_roundtrip_and_timing(self):
        recorder = MpcLatencyTraceRecorder(metadata={"mode": "off"})
        recorder.begin_source(
            source_sample_index=0,
            source_anchor_index=0,
            source_time=0.0,
            source_sample_wall_ns=100,
        )
        recorder.mark_packet_ready(source_anchor_index=0, wall_ns=200)
        recorder.mark_first_certified(source_anchor_index=0, wall_ns=350)
        recorder.record_execution(
            sample_index=0,
            simulation_time_s=0.0,
            final_output_certified=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = recorder.save(Path(directory))
            payload = load_and_validate_trace(Path(paths["json"]))
            self.assertEqual(payload["schema_version"], "mpc_latency_trace_v1")
            self.assertAlmostEqual(payload["anchors"][0]["source_to_packet_ready_ms"], 0.0001)
            wrapper = json.loads(Path(paths["json"]).read_text())
            wrapper["payload"]["anchors"][0]["source_time_s"] = 1.0
            Path(paths["json"]).write_text(json.dumps(wrapper))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_and_validate_trace(Path(paths["json"]))

    def test_fail_closed_trace_and_state_survive_exception_path(self):
        recorder = MpcLatencyTraceRecorder(metadata={"mode": "fixed"})
        recorder.begin_source(
            source_sample_index=12,
            source_anchor_index=4,
            source_time=0.024,
            source_sample_wall_ns=100,
        )
        recorder.mark_packet_ready(source_anchor_index=4, wall_ns=200)
        recorder.record_execution(
            sample_index=12,
            simulation_time_s=0.024,
            final_output_certified=True,
        )
        recorder.record_failure(
            sample_index=13,
            simulation_time_s=0.026,
            status="NO_SAFE_TORQUE",
            right_arm_d_ctrl_written=False,
            mj_step_performed=False,
        )
        with self.assertRaisesRegex(ValueError, "already contains"):
            recorder.record_failure(sample_index=14)
        with tempfile.TemporaryDirectory() as directory:
            paths = recorder.save_failure_snapshot(
                Path(directory),
                arrays={
                    "qpos": np.arange(4, dtype=np.float64),
                    "diagnostic_nonfinite": np.array([np.nan]),
                },
            )
            payload = load_and_validate_trace(Path(paths["json"]))
            self.assertEqual(payload["failure"]["sample_index"], 13)
            self.assertFalse(payload["failure"]["right_arm_d_ctrl_written"])
            self.assertFalse(payload["failure"]["mj_step_performed"])
            self.assertIn("state_snapshot", payload["failure"])
            summary = json.loads(Path(paths["summary"]).read_text())
            self.assertTrue(summary["failed_pre_step"])
            self.assertEqual(summary["execution_count"], 1)
            with np.load(paths["failure_state"]) as snapshot:
                np.testing.assert_array_equal(
                    snapshot["qpos"], np.arange(4, dtype=np.float64)
                )
                self.assertTrue(np.isnan(snapshot["diagnostic_nonfinite"][0]))

    def test_trace_rejects_nonmonotonic_and_invalid_wall_times(self):
        recorder = MpcLatencyTraceRecorder(metadata={})
        recorder.begin_source(
            source_sample_index=0, source_anchor_index=0,
            source_time=0.0, source_sample_wall_ns=100,
        )
        with self.assertRaises(ValueError):
            recorder.begin_source(
                source_sample_index=6, source_anchor_index=2,
                source_time=0.012, source_sample_wall_ns=110,
            )
        with self.assertRaises(ValueError):
            recorder.mark_packet_ready(source_anchor_index=0, wall_ns=99)

    def test_cli_contract_stops_at_l1c(self):
        validate_experimental_latency_cli(
            mode="fixed", delay_ms=4.0, capture_path=None,
            full_task=True, predictor="full_task_template", runtime_mode="process",
        )
        for invalid in (1.0, 6.0):
            with self.assertRaises(ValueError):
                validate_experimental_latency_cli(
                    mode="fixed", delay_ms=invalid, capture_path=None,
                    full_task=True, predictor="full_task_template", runtime_mode="process",
                )
        with self.assertRaises(ValueError):
            validate_experimental_latency_cli(
                mode="trace", delay_ms=None, capture_path="trace.json",
                full_task=True, predictor="full_task_template", runtime_mode="process",
            )


if __name__ == "__main__":
    unittest.main()
