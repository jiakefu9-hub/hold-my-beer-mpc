import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.archive_final_freeze_runs import (
    CONFIRMATION_ONE,
    CONFIRMATION_TWO,
    SOURCE_RELATIVE_PATHS,
    ArchiveContract,
    FinalFreezeArchiveError,
    create_archive,
    delete_archived_sources,
    inventory_sources,
    verify_manifest_archive,
)


class ArchiveFinalFreezeRunsTest(unittest.TestCase):
    def setUp(self):
        self.tar = shutil.which("tar")
        self.zstd = shutil.which("zstd") or "/home/fjk/miniforge3/bin/zstd"
        if not self.tar or not Path(self.zstd).is_file():
            self.skipTest("tar/zstd is unavailable")

    def _fixture(self, root: Path) -> tuple[ArchiveContract, Path]:
        repo = root / "repo"
        parent = repo / "evaluation" / "t2_full_task_closed_loop"
        parent.mkdir(parents=True)
        for index, relative in enumerate(SOURCE_RELATIVE_PATHS):
            run = repo / relative
            (run / "nested").mkdir(parents=True)
            (run / "run_metadata.json").write_text(
                json.dumps({"run": index, "passed": True}) + "\n",
                encoding="utf-8",
            )
            (run / "nested" / "perf_intervals.csv").write_text(
                "time_ms,total_ms\n0,3.4\n", encoding="utf-8"
            )
        sibling = parent / "keep_this_unlisted_run"
        sibling.mkdir()
        (sibling / "keep.txt").write_text("keep\n", encoding="utf-8")
        contract = ArchiveContract(
            repo_root=repo,
            source_relative_paths=SOURCE_RELATIVE_PATHS,
            archive_path=root / "archives" / "final_freeze_full_runs.tar.zst",
            manifest_path=(
                repo
                / "evaluation_summary"
                / "full_task_template_v2_final_freeze"
                / "final_runs"
                / "final_freeze_archive_manifest.json"
            ),
        )
        return contract, sibling

    def _create(self, contract: ArchiveContract):
        return create_archive(
            contract,
            enforce_production_contract=False,
            tar_executable=self.tar,
            zstd_executable=self.zstd,
        )

    def _delete(self, contract: ArchiveContract, first: str, second: str):
        return delete_archived_sources(
            contract,
            confirmation_one=first,
            confirmation_two=second,
            enforce_production_contract=False,
            tar_executable=self.tar,
            zstd_executable=self.zstd,
        )

    def test_create_records_each_file_and_retains_both_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, sibling = self._fixture(Path(temporary).resolve())
            expected = inventory_sources(contract)
            manifest = self._create(contract)

            self.assertEqual(manifest["status"], "VERIFIED_SOURCE_RETAINED")
            self.assertEqual(manifest["inventory"], expected)
            self.assertEqual(manifest["inventory"]["file_count"], 4)
            self.assertTrue(contract.archive_path.is_file())
            self.assertTrue(contract.manifest_path.is_file())
            self.assertTrue(sibling.is_dir())
            for relative in SOURCE_RELATIVE_PATHS:
                self.assertTrue((contract.repo_root / relative).is_dir())
            for record in manifest["inventory"]["files"]:
                payload = (contract.repo_root / record["path"]).read_bytes()
                self.assertEqual(record["size_bytes"], len(payload))
                self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
            verification = manifest["archive_verification"]
            self.assertTrue(verification["zstd_test_passed"])
            self.assertTrue(verification["member_paths_exact"])
            self.assertTrue(verification["member_types_exact"])
            self.assertTrue(verification["archived_file_size_sha256_exact"])

            repeated = verify_manifest_archive(
                contract,
                enforce_production_contract=False,
                tar_executable=self.tar,
                zstd_executable=self.zstd,
            )
            self.assertEqual(repeated["status"], "verified")

    def test_create_refuses_to_overwrite_archive_or_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, _ = self._fixture(Path(temporary).resolve())
            self._create(contract)
            archive_before = contract.archive_path.read_bytes()
            manifest_before = contract.manifest_path.read_bytes()
            with self.assertRaises(FinalFreezeArchiveError):
                self._create(contract)
            self.assertEqual(contract.archive_path.read_bytes(), archive_before)
            self.assertEqual(contract.manifest_path.read_bytes(), manifest_before)

    def test_delete_requires_both_exact_confirmations(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, sibling = self._fixture(Path(temporary).resolve())
            self._create(contract)
            with self.assertRaises(FinalFreezeArchiveError):
                self._delete(contract, "wrong", CONFIRMATION_TWO)
            with self.assertRaises(FinalFreezeArchiveError):
                self._delete(contract, CONFIRMATION_ONE, "wrong")
            for relative in SOURCE_RELATIVE_PATHS:
                self.assertTrue((contract.repo_root / relative).is_dir())
            self.assertTrue(sibling.is_dir())

    def test_verified_delete_removes_only_the_two_exact_realpaths(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, sibling = self._fixture(Path(temporary).resolve())
            self._create(contract)
            result = self._delete(contract, CONFIRMATION_ONE, CONFIRMATION_TWO)
            self.assertEqual(result["status"], "VERIFIED_ARCHIVE_SOURCE_DELETED")
            self.assertEqual(result["deletion"]["deleted_directory_count"], 2)
            for relative in SOURCE_RELATIVE_PATHS:
                self.assertFalse((contract.repo_root / relative).exists())
            self.assertTrue(sibling.is_dir())
            self.assertEqual((sibling / "keep.txt").read_text(), "keep\n")
            self.assertTrue(contract.archive_path.is_file())
            self.assertTrue(contract.manifest_path.is_file())

    def test_source_mutation_after_archive_blocks_all_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, sibling = self._fixture(Path(temporary).resolve())
            self._create(contract)
            changed = contract.repo_root / SOURCE_RELATIVE_PATHS[0] / "run_metadata.json"
            changed.write_text('{"tampered": true}\n', encoding="utf-8")
            with self.assertRaises(FinalFreezeArchiveError):
                self._delete(contract, CONFIRMATION_ONE, CONFIRMATION_TWO)
            for relative in SOURCE_RELATIVE_PATHS:
                self.assertTrue((contract.repo_root / relative).is_dir())
            self.assertTrue(sibling.is_dir())

    def test_archive_mutation_blocks_all_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, sibling = self._fixture(Path(temporary).resolve())
            self._create(contract)
            with contract.archive_path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaises(FinalFreezeArchiveError):
                self._delete(contract, CONFIRMATION_ONE, CONFIRMATION_TWO)
            for relative in SOURCE_RELATIVE_PATHS:
                self.assertTrue((contract.repo_root / relative).is_dir())
            self.assertTrue(sibling.is_dir())

    def test_archive_extra_path_or_link_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            contract, sibling = self._fixture(root)
            self._create(contract)

            def replace_archive(member_paths):
                contract.archive_path.unlink()
                subprocess.run(
                    [
                        self.tar,
                        f"--use-compress-program={self.zstd}",
                        "--create",
                        "--file",
                        str(contract.archive_path),
                        "--directory",
                        str(contract.repo_root),
                        "--",
                        *member_paths,
                    ],
                    check=True,
                )
                manifest = json.loads(contract.manifest_path.read_text())
                payload = contract.archive_path.read_bytes()
                manifest["archive_verification"]["archive_size_bytes"] = len(payload)
                manifest["archive_verification"]["archive_sha256"] = hashlib.sha256(
                    payload
                ).hexdigest()
                contract.manifest_path.write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )

            replace_archive([*SOURCE_RELATIVE_PATHS, sibling.relative_to(contract.repo_root).as_posix()])
            with self.assertRaises(FinalFreezeArchiveError):
                verify_manifest_archive(
                    contract,
                    enforce_production_contract=False,
                    tar_executable=self.tar,
                    zstd_executable=self.zstd,
                )

            staging = root / "staging"
            for relative in SOURCE_RELATIVE_PATHS:
                shutil.copytree(contract.repo_root / relative, staging / relative)
            replaced = staging / SOURCE_RELATIVE_PATHS[0] / "run_metadata.json"
            replaced.unlink()
            replaced.symlink_to("nested/perf_intervals.csv")
            contract.archive_path.unlink()
            subprocess.run(
                [
                    self.tar,
                    f"--use-compress-program={self.zstd}",
                    "--create",
                    "--file",
                    str(contract.archive_path),
                    "--directory",
                    str(staging),
                    "--",
                    *SOURCE_RELATIVE_PATHS,
                ],
                check=True,
            )
            manifest = json.loads(contract.manifest_path.read_text())
            payload = contract.archive_path.read_bytes()
            manifest["archive_verification"]["archive_size_bytes"] = len(payload)
            manifest["archive_verification"]["archive_sha256"] = hashlib.sha256(
                payload
            ).hexdigest()
            contract.manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaises(FinalFreezeArchiveError):
                verify_manifest_archive(
                    contract,
                    enforce_production_contract=False,
                    tar_executable=self.tar,
                    zstd_executable=self.zstd,
                )
            for relative in SOURCE_RELATIVE_PATHS:
                self.assertTrue((contract.repo_root / relative).is_dir())

    def test_symlink_or_broad_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            contract, _ = self._fixture(root)
            outside = root / "outside.txt"
            outside.write_text("outside\n", encoding="utf-8")
            link = contract.repo_root / SOURCE_RELATIVE_PATHS[0] / "unsafe-link"
            link.symlink_to(outside)
            with self.assertRaises(FinalFreezeArchiveError):
                inventory_sources(contract)
            link.unlink()

            broad = ArchiveContract(
                repo_root=contract.repo_root,
                source_relative_paths=("evaluation/t2_full_task_closed_loop",),
                archive_path=contract.archive_path,
                manifest_path=contract.manifest_path,
            )
            with self.assertRaises(FinalFreezeArchiveError):
                inventory_sources(broad)

    def test_production_guard_rejects_injected_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, _ = self._fixture(Path(temporary).resolve())
            with self.assertRaises(FinalFreezeArchiveError):
                create_archive(
                    contract,
                    tar_executable=self.tar,
                    zstd_executable=self.zstd,
                )
            self.assertFalse(contract.archive_path.exists())


if __name__ == "__main__":
    unittest.main()
