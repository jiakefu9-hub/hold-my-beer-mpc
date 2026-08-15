import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.prepare_data_cleanup import (
    CleanupError,
    FINAL_RUN_R1_SOURCE_PATHS,
    FINAL_RUN_SOURCE_PATHS,
    _assert_group_deleted,
    _audit_archive_summary,
    _audit_compact_evidence,
    assert_exact_archive_members,
    assert_safe_tar_member_types,
    assert_safe_cleanup_path,
    delete_exact_manifest_targets,
    normalize_tar_listing,
    verify_group_archive,
)


class PrepareDataCleanupTest(unittest.TestCase):
    def _repo(self, root: Path) -> Path:
        for relative in (
            "evaluation",
            "disturbance_learning/data",
            "disturbance_learning/artifacts",
        ):
            (root / relative).mkdir(parents=True, exist_ok=True)
        return root

    def _add_final_run_evidence(
        self,
        repo: Path,
        evidence: Path,
        *,
        package_name: str = "final_runs",
        schema_version: str = "full_task_template_v2_final_freeze_two_run_files_v1",
        source_paths=FINAL_RUN_SOURCE_PATHS,
        archive_filename: str = "final_freeze_full_runs.tar.zst",
    ):
        final_runs = evidence / package_name
        copied = final_runs / "runs" / "final" / "result.json"
        generated = final_runs / "aggregate.json"
        final_builder = repo / "tools" / "final_builder.py"
        copied.parent.mkdir(parents=True)
        final_builder.parent.mkdir(parents=True, exist_ok=True)
        copied.write_text('{"pass": true}\n', encoding="utf-8")
        generated.write_text('{"aggregate": true}\n', encoding="utf-8")
        final_builder.write_text("# final builder\n", encoding="utf-8")
        file_manifest = {
            "schema_version": schema_version,
            "status": "PASS",
            "output_repository_path": (
                f"evaluation_summary/full_task_template_v2_final_freeze/{package_name}"
            ),
            "builder": {
                "repository_path": "tools/final_builder.py",
                "sha256": hashlib.sha256(final_builder.read_bytes()).hexdigest(),
            },
            "validation": {"all_test_gates_pass": True},
            "copied_files": [
                {
                    "output_package_path": "runs/final/result.json",
                    "bytes": copied.stat().st_size,
                    "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
                }
            ],
            "generated_files": [
                {
                    "output_package_path": "aggregate.json",
                    "bytes": generated.stat().st_size,
                    "sha256": hashlib.sha256(generated.read_bytes()).hexdigest(),
                }
            ],
        }
        (final_runs / "final_freeze_file_manifest.json").write_text(
            json.dumps(file_manifest), encoding="utf-8"
        )
        archive = repo.parent / archive_filename
        archive.write_bytes(b"verified final archive")
        expected_realpaths = [
            str((repo / relative).resolve(strict=False))
            for relative in source_paths
        ]
        archive_manifest = {
            "schema_version": "disturbance-lab-final-freeze-run-archive-v1",
            "status": "VERIFIED_ARCHIVE_SOURCE_DELETED",
            "archive_absolute_path": str(archive),
            "source_relative_paths": list(source_paths),
            "allowed_delete_realpaths": expected_realpaths,
            "archive_verification": {
                "status": "verified",
                "archive_size_bytes": archive.stat().st_size,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "zstd_test_passed": True,
                "tar_listing_passed": True,
                "member_paths_exact": True,
                "member_types_exact": True,
                "archived_file_size_sha256_exact": True,
            },
            "deletion": {
                "status": "completed",
                "deleted_realpaths": expected_realpaths,
            },
        }
        (final_runs / "final_freeze_archive_manifest.json").write_text(
            json.dumps(archive_manifest), encoding="utf-8"
        )
        return archive, copied

    def _add_both_final_run_evidence(self, repo: Path, evidence: Path):
        legacy_archive, legacy_output = self._add_final_run_evidence(repo, evidence)
        r1_archive, r1_output = self._add_final_run_evidence(
            repo,
            evidence,
            package_name="final_runs_r1",
            schema_version="full_task_template_v2_final_freeze_two_run_files_r1_v1",
            source_paths=FINAL_RUN_R1_SOURCE_PATHS,
            archive_filename="final_freeze_full_runs_r1.tar.zst",
        )
        return legacy_archive, legacy_output, r1_archive, r1_output

    def test_cleanup_root_and_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = self._repo(Path(temporary).resolve())
            with self.assertRaises(CleanupError):
                assert_safe_cleanup_path(repo, "evaluation", allow_missing=False)
            with self.assertRaises(CleanupError):
                assert_safe_cleanup_path(repo, "evaluation/../outside", allow_missing=True)

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = self._repo(Path(temporary).resolve())
            outside = repo / "outside"
            outside.mkdir()
            (repo / "evaluation" / "escape").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(CleanupError):
                assert_safe_cleanup_path(repo, "evaluation/escape/file", allow_missing=True)

    def test_archive_listing_requires_exact_safe_paths(self):
        files, directories = normalize_tar_listing(
            [
                "./evaluation/run/\n",
                "./evaluation/run/result.json\n",
                "disturbance_learning/data/raw.npz\n",
            ]
        )
        self.assertEqual(files, ["evaluation/run/result.json", "disturbance_learning/data/raw.npz"])
        self.assertEqual(directories, ["evaluation/run"])
        assert_exact_archive_members(files, list(reversed(files)))
        with self.assertRaises(CleanupError):
            normalize_tar_listing(["../escape\n"])
        with self.assertRaises(CleanupError):
            assert_exact_archive_members(files, ["evaluation/run/result.json"])

    def test_archive_member_types_reject_links(self):
        assert_safe_tar_member_types(
            ["evaluation/run/", "evaluation/run/result.json"],
            ["drwxr-xr-x owner/group 0 date evaluation/run/", "-rw-r--r-- owner/group 1 date evaluation/run/result.json"],
        )
        with self.assertRaises(CleanupError):
            assert_safe_tar_member_types(
                ["evaluation/run/result.json"],
                ["lrwxrwxrwx owner/group 0 date evaluation/run/result.json -> elsewhere"],
            )

    def test_exact_delete_preserves_unlisted_file_and_cleanup_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = self._repo(Path(temporary).resolve())
            run = repo / "evaluation" / "run"
            keep = repo / "evaluation" / "keep.txt"
            run.mkdir()
            (run / "delete.json").write_text("delete", encoding="utf-8")
            keep.write_text("keep", encoding="utf-8")
            deleted_files, deleted_directories = delete_exact_manifest_targets(
                repo,
                ["evaluation/run/delete.json"],
                ["evaluation/run"],
            )
            self.assertEqual((deleted_files, deleted_directories), (1, 1))
            self.assertTrue((repo / "evaluation").is_dir())
            self.assertEqual(keep.read_text(encoding="utf-8"), "keep")

    def test_tar_zstd_verification_checks_archive_and_exact_members(self):
        zstd = shutil.which("zstd") or "/home/fjk/miniforge3/bin/zstd"
        tar = shutil.which("tar")
        if not tar or not Path(zstd).is_file():
            self.skipTest("tar/zstd is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            archive_root = root / "archives"
            result = source / "evaluation" / "run" / "result.json"
            result.parent.mkdir(parents=True)
            archive_root.mkdir()
            result.write_text('{"ok": true}\n', encoding="utf-8")
            archive = archive_root / "test.tar.zst"
            subprocess.run(
                [
                    tar,
                    f"--use-compress-program={zstd}",
                    "-cf",
                    str(archive),
                    "-C",
                    str(source),
                    "evaluation/run",
                ],
                check=True,
            )
            payload = result.read_bytes()
            manifest = {
                "archive_root": str(archive_root),
                "archive_groups": [
                    {
                        "group_id": "test",
                        "archive_path": str(archive),
                        "source_files": [
                            {
                                "path": "evaluation/run/result.json",
                                "size_bytes": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest(),
                            }
                        ],
                    }
                ],
            }
            verification = verify_group_archive(
                manifest,
                "test",
                zstd_executable=zstd,
                tar_executable=tar,
            )
            self.assertEqual(verification["status"], "verified")
            self.assertTrue(verification["zstd_test_passed"])
            self.assertTrue(verification["member_paths_exact"])
            self.assertTrue(verification["member_types_safe"])
            self.assertEqual(verification["listed_regular_file_count"], 1)

    def test_post_audit_requires_deleted_targets_to_remain_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = self._repo(Path(temporary).resolve())
            group = {
                "group_id": "test",
                "cleanup_file_count": 1,
                "cleanup_total_bytes": 7,
                "cleanup_files": ["evaluation/run/result.json"],
                "cleanup_directories_deepest_first": ["evaluation/run"],
                "deletion": {
                    "status": "completed",
                    "completed_at_utc": "2026-08-15T00:00:00Z",
                    "deleted_file_count": 1,
                    "deleted_directory_count": 1,
                },
            }
            record = _assert_group_deleted(repo, group)
            self.assertEqual(record["status"], "PASS")
            (repo / "evaluation" / "run").mkdir()
            (repo / "evaluation" / "run" / "result.json").write_text(
                "restored", encoding="utf-8"
            )
            with self.assertRaises(CleanupError):
                _assert_group_deleted(repo, group)

    def test_post_archive_audit_rechecks_current_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_root = Path(temporary).resolve()
            archive = archive_root / "archive.tar.zst"
            archive.write_bytes(b"verified archive")
            checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
            manifest = {
                "archive_root": str(archive_root),
                "archive_groups": [
                    {
                        "group_id": "test",
                        "archive_path": str(archive),
                        "source_file_count": 1,
                        "source_total_bytes": 3,
                        "archive_verification": {
                            "status": "verified",
                            "archive_sha256": checksum,
                            "archive_size_bytes": archive.stat().st_size,
                            "zstd_test_passed": True,
                            "tar_listing_passed": True,
                            "member_paths_exact": True,
                            "member_types_safe": True,
                            "listed_regular_file_count": 1,
                            "listed_directory_entry_count": 0,
                        },
                    }
                ],
            }
            result = _audit_archive_summary(manifest)
            self.assertEqual(result["status"], "PASS")
            archive.write_bytes(b"changed")
            with self.assertRaises(CleanupError):
                _audit_archive_summary(manifest)

    def test_compact_evidence_audit_rehashes_all_declared_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary).resolve()
            evidence = repo / "evaluation_summary" / "full_task_template_v2_final_freeze"
            output = evidence / "controlled_runs" / "summary.json"
            builder = repo / "tools" / "builder.py"
            output.parent.mkdir(parents=True)
            builder.parent.mkdir(parents=True)
            output.write_text('{"pass": true}\n', encoding="utf-8")
            builder.write_text("# builder\n", encoding="utf-8")
            output_bytes = output.read_bytes()
            validation = {
                "all_16074_mapper_outputs_certified": True,
                "all_7974_complete_intervals_within_6ms": True,
                "all_copy_checksums_match": True,
                "all_six_certified_control_gates_pass": True,
                "all_six_runtime_preflights_pass": True,
                "offline_online_parity_pass": True,
            }
            evidence_manifest = {
                "schema_version": "full_task_template_v2_compact_evidence_files_v1",
                "status": "PASS",
                "repository_root": str(repo),
                "output_absolute_path": str(evidence),
                "builder": {
                    "repository_path": "tools/builder.py",
                    "sha256": hashlib.sha256(builder.read_bytes()).hexdigest(),
                },
                "validation": validation,
                "copied_files": [
                    {
                        "output_repository_path": (
                            "evaluation_summary/full_task_template_v2_final_freeze/"
                            "controlled_runs/summary.json"
                        ),
                        "bytes": len(output_bytes),
                        "sha256": hashlib.sha256(output_bytes).hexdigest(),
                    }
                ],
                "generated_files": [],
                "frozen_assets": {},
            }
            (evidence / "evidence_file_manifest.json").write_text(
                json.dumps(evidence_manifest), encoding="utf-8"
            )
            (evidence / "cleanup_manifest.json").write_text("{}\n", encoding="utf-8")
            final_archive, _, r1_archive, _ = self._add_both_final_run_evidence(
                repo, evidence
            )
            result = _audit_compact_evidence(
                repo,
                evidence,
                final_run_archive_path=final_archive,
                final_run_r1_archive_path=r1_archive,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["verified_output_file_count"], 1)
            self.assertEqual(
                result["additional_final_run_evidence"]["status"], "PASS"
            )
            output.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(CleanupError):
                _audit_compact_evidence(
                    repo,
                    evidence,
                    final_run_archive_path=final_archive,
                    final_run_r1_archive_path=r1_archive,
                )

    def test_compact_evidence_audit_rejects_final_run_tamper_and_extra_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary).resolve()
            evidence = repo / "evaluation_summary" / "full_task_template_v2_final_freeze"
            output = evidence / "controlled_runs" / "summary.json"
            builder = repo / "tools" / "builder.py"
            output.parent.mkdir(parents=True)
            builder.parent.mkdir(parents=True)
            output.write_text('{"pass": true}\n', encoding="utf-8")
            builder.write_text("# builder\n", encoding="utf-8")
            payload = output.read_bytes()
            evidence_manifest = {
                "schema_version": "full_task_template_v2_compact_evidence_files_v1",
                "status": "PASS",
                "repository_root": str(repo),
                "output_absolute_path": str(evidence),
                "builder": {
                    "repository_path": "tools/builder.py",
                    "sha256": hashlib.sha256(builder.read_bytes()).hexdigest(),
                },
                "validation": {
                    "all_16074_mapper_outputs_certified": True,
                    "all_7974_complete_intervals_within_6ms": True,
                    "all_copy_checksums_match": True,
                    "all_six_certified_control_gates_pass": True,
                    "all_six_runtime_preflights_pass": True,
                    "offline_online_parity_pass": True,
                },
                "copied_files": [{
                    "output_repository_path": (
                        "evaluation_summary/full_task_template_v2_final_freeze/"
                        "controlled_runs/summary.json"
                    ),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }],
                "generated_files": [],
                "frozen_assets": {},
            }
            (evidence / "evidence_file_manifest.json").write_text(
                json.dumps(evidence_manifest), encoding="utf-8"
            )
            (evidence / "cleanup_manifest.json").write_text("{}\n", encoding="utf-8")
            final_archive, final_output, r1_archive, _ = (
                self._add_both_final_run_evidence(repo, evidence)
            )
            _audit_compact_evidence(
                repo,
                evidence,
                final_run_archive_path=final_archive,
                final_run_r1_archive_path=r1_archive,
            )
            final_output.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(CleanupError):
                _audit_compact_evidence(
                    repo,
                    evidence,
                    final_run_archive_path=final_archive,
                    final_run_r1_archive_path=r1_archive,
                )
            final_output.write_text('{"pass": true}\n', encoding="utf-8")
            (evidence / "final_runs" / "unexpected.txt").write_text(
                "unexpected\n", encoding="utf-8"
            )
            with self.assertRaises(CleanupError):
                _audit_compact_evidence(
                    repo,
                    evidence,
                    final_run_archive_path=final_archive,
                    final_run_r1_archive_path=r1_archive,
                )


if __name__ == "__main__":
    unittest.main()
