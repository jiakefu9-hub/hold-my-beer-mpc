import copy
import json
import unittest
from pathlib import Path

from tools.build_final_freeze_evidence import (
    RUN_EVIDENCE_FILES,
    EvidenceError,
    validate_formal_environment,
)
from tools.build_final_freeze_run_evidence import (
    AGGREGATE_JSON,
    DEFAULT_OUTPUT,
    DEFAULT_R1_OUTPUT,
    FILE_MANIFEST,
    FINAL_RUNS,
    FINAL_RUN_EVIDENCE_FILES,
    FINAL_RUN_SOURCE_ROOT,
    R1_FINAL_RUNS,
    REPOSITORY_ROOT,
    verify,
)


class BuildFinalFreezeRunEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = REPOSITORY_ROOT / DEFAULT_OUTPUT
        cls.aggregate = json.loads((cls.output / AGGREGATE_JSON).read_text())
        cls.manifest = json.loads((cls.output / FILE_MANIFEST).read_text())
        cls.r1_output = REPOSITORY_ROOT / DEFAULT_R1_OUTPUT
        cls.r1_aggregate = json.loads(
            (cls.r1_output / AGGREGATE_JSON).read_text()
        )
        cls.r1_manifest = json.loads(
            (cls.r1_output / FILE_MANIFEST).read_text()
        )

    def test_source_selection_is_exact_and_not_latest_discovery(self):
        self.assertEqual(
            FINAL_RUNS,
            (
                ("20260815_231454_final_freeze", "nominal"),
                (
                    "20260815_231555_final_freeze_heldout_pair_02_minus",
                    "heldout_pair_02_minus",
                ),
            ),
        )
        self.assertEqual(FINAL_RUN_SOURCE_ROOT.as_posix(), "evaluation/t2_full_task_closed_loop")
        self.assertEqual(FINAL_RUN_EVIDENCE_FILES, RUN_EVIDENCE_FILES)
        self.assertEqual(len(FINAL_RUN_EVIDENCE_FILES), 13)
        self.assertEqual(
            self.manifest["source_selection"]["selection_method"],
            "fixed constants; no directory scan",
        )
        self.assertEqual(
            R1_FINAL_RUNS,
            (
                ("20260816_011925_final_freeze", "nominal"),
                (
                    "20260816_012007_final_freeze_heldout_pair_02_minus",
                    "heldout_pair_02_minus",
                ),
            ),
        )
        self.assertEqual(
            self.r1_manifest["source_selection"]["selection_method"],
            "fixed constants; no directory scan",
        )

    def test_package_verifies_without_requiring_archived_sources(self):
        result = verify(REPOSITORY_ROOT, self.output, require_sources=False)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["copied_file_count"], 48)
        self.assertGreaterEqual(result["missing_source_count"], 0)

    def test_paths_are_relocatable_and_large_arrays_are_not_copied(self):
        banned = {
            "trajectory.npz",
            "metrics.npz",
            "full_task_nominal_raw.npz",
            "control_preview.csv",
            "startup_pd_handoff_trace.npz",
        }
        for item in self.manifest["copied_files"]:
            source = Path(item["source_repository_path"])
            output = Path(item["output_package_path"])
            self.assertFalse(source.is_absolute())
            self.assertFalse(output.is_absolute())
            self.assertTrue(banned.isdisjoint(source.parts))
            self.assertTrue(banned.isdisjoint(output.parts))
            self.assertNotIn("source_absolute_path", item)
        for item in self.manifest["derived_only_source_files"]:
            self.assertFalse(Path(item["source_repository_path"]).is_absolute())
            self.assertFalse(item["copied_to_evidence_package"])

    def test_legacy_smoke_false_is_preserved_but_certification_gates_pass(self):
        self.assertFalse(
            self.aggregate["two_run_aggregate"][
                "legacy_smoke_status_is_acceptance_gate"
            ]
        )
        expected_fallbacks = {
            "nominal": (3, 2, 0),
            "heldout_pair_02_minus": (5, 3, 1),
        }
        for run in self.aggregate["runs"]:
            self.assertEqual(run["source_smoke_status"]["status"], "FAIL")
            self.assertFalse(run["source_smoke_status"]["smoke_passed"])
            self.assertFalse(
                run["source_smoke_status"]["is_final_freeze_acceptance_gate"]
            )
            self.assertTrue(run["safety"]["certified_control_gate_pass"])
            fallback = run["safety"]["certified_fallbacks"]
            self.assertEqual(
                (
                    fallback["legacy_runtime_mapping_safety_fallback_count"],
                    fallback["rescue_used_count"],
                    fallback["hold_last_succeeded_count"],
                ),
                expected_fallbacks[run["scenario"]],
            )
            self.assertEqual(fallback["final_output_uncertified_count"], 0)

    def test_environment_validator_fails_closed_on_affinity_or_thread_drift(self):
        run_id = FINAL_RUNS[0][0]
        packaged = self.output / "runs" / run_id
        metadata = json.loads((packaged / "run_metadata.json").read_text())
        preflight = json.loads(
            (packaged / "formal_full_task_runtime_preflight.json").read_text()
        )
        bad_affinity = copy.deepcopy(preflight)
        bad_affinity["parent_cpu_affinity"] = [6, 7]
        with self.assertRaises(EvidenceError):
            validate_formal_environment(bad_affinity, metadata, run_id)
        bad_threads = copy.deepcopy(preflight)
        bad_threads["thread_environment"]["OMP_NUM_THREADS"] = "2"
        with self.assertRaises(EvidenceError):
            validate_formal_environment(bad_threads, metadata, run_id)

    def test_r1_smoke_summary_uses_certified_fallback_warning_semantics(self):
        result = verify(
            REPOSITORY_ROOT,
            self.r1_output,
            # The full sources are deliberately removed after their external
            # archive is verified; the compact package must remain sufficient.
            require_sources=False,
            revision="r1",
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(
            self.r1_aggregate["two_run_aggregate"][
                "all_task_protocol_smoke_summaries_pass"
            ]
        )
        expected_fallbacks = {"nominal": 3, "heldout_pair_02_minus": 5}
        for run in self.r1_aggregate["runs"]:
            smoke = run["source_smoke_status"]
            self.assertEqual(smoke["status"], "PASS")
            self.assertTrue(smoke["smoke_passed"])
            self.assertFalse(smoke["nominal_mapping_path_passed"])
            self.assertEqual(smoke["warnings"], ["MAPPING_SAFETY_FALLBACK_USED"])
            self.assertEqual(
                smoke["runtime_mapping_safety_fallback_count"],
                expected_fallbacks[run["scenario"]],
            )
            self.assertTrue(run["safety"]["certified_control_gate_pass"])


if __name__ == "__main__":
    unittest.main()
