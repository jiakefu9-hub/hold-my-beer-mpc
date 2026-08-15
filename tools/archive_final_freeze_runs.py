#!/usr/bin/env python3
"""Archive, verify, and explicitly retire the two frozen full-task runs.

The command-line contract is intentionally not configurable: it can only read
the two final-freeze run directories, create the one declared external archive
and write the one declared evidence manifest.  Archive creation never removes
source data.  Deletion is a separate command which requires two exact phrases
and repeats the archive, member, content, source-hash, and realpath checks before
removing either source directory.

Examples (do not run ``delete`` until the archive manifest has been reviewed)::

    python3 tools/archive_final_freeze_runs.py create
    python3 tools/archive_final_freeze_runs.py verify
    python3 tools/archive_final_freeze_runs.py delete \
      --confirm DELETE_VERIFIED_FINAL_FREEZE_ARCHIVE \
      --confirm-again DELETE_ONLY_TWO_FIXED_FINAL_FREEZE_RUNS
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RELATIVE_PATHS = (
    "evaluation/t2_full_task_closed_loop/20260815_231454_final_freeze",
    (
        "evaluation/t2_full_task_closed_loop/"
        "20260815_231555_final_freeze_heldout_pair_02_minus"
    ),
)
ARCHIVE_PATH = Path(
    "/home/fjk/g1_ws/disturbance-lab-archives/20260815_pre_cleanup/"
    "final_freeze_full_runs.tar.zst"
)
MANIFEST_PATH = (
    REPO_ROOT
    / "evaluation_summary"
    / "full_task_template_v2_final_freeze"
    / "final_runs"
    / "final_freeze_archive_manifest.json"
)
SCHEMA_VERSION = "disturbance-lab-final-freeze-run-archive-v1"
CONFIRMATION_ONE = "DELETE_VERIFIED_FINAL_FREEZE_ARCHIVE"
CONFIRMATION_TWO = "DELETE_ONLY_TWO_FIXED_FINAL_FREEZE_RUNS"


class FinalFreezeArchiveError(RuntimeError):
    """A fail-closed archive or deletion contract violation."""


@dataclass(frozen=True)
class ArchiveContract:
    repo_root: Path
    source_relative_paths: tuple[str, ...]
    archive_path: Path
    manifest_path: Path


def production_contract() -> ArchiveContract:
    return ArchiveContract(
        repo_root=REPO_ROOT,
        source_relative_paths=SOURCE_RELATIVE_PATHS,
        archive_path=ARCHIVE_PATH,
        manifest_path=MANIFEST_PATH,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_record(repo_root: Path, path: Path) -> dict[str, Any]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise FinalFreezeArchiveError(f"source is not a unique regular file: {path}")
    checksum = _sha256_file(path)
    after = path.lstat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise FinalFreezeArchiveError(f"source changed while hashing: {path}")
    try:
        relative = path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise FinalFreezeArchiveError(f"source escaped repository: {path}") from exc
    return {
        "path": relative,
        "size_bytes": before.st_size,
        "sha256": checksum,
    }


def _validate_relative_path(value: str) -> PurePosixPath:
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
        or any(character in value for character in "\n\r\0")
    ):
        raise FinalFreezeArchiveError(f"unsafe relative path: {value!r}")
    return pure


def _assert_no_overlapping_sources(source_relative_paths: Sequence[str]) -> None:
    parsed = [_validate_relative_path(value) for value in source_relative_paths]
    if len(set(parsed)) != len(parsed):
        raise FinalFreezeArchiveError("source directories must be unique")
    for index, left in enumerate(parsed):
        for right in parsed[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise FinalFreezeArchiveError("source directories may not overlap")


def _safe_source_roots(contract: ArchiveContract) -> list[Path]:
    """Resolve and validate the exact directory roots declared by a contract."""

    _assert_no_overlapping_sources(contract.source_relative_paths)
    repo_root = contract.repo_root.resolve(strict=True)
    allowed_parent = (repo_root / "evaluation" / "t2_full_task_closed_loop").resolve(
        strict=True
    )
    roots: list[Path] = []
    for relative in contract.source_relative_paths:
        pure = _validate_relative_path(relative)
        lexical = repo_root.joinpath(*pure.parts)
        source_stat = lexical.lstat()
        if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISDIR(source_stat.st_mode):
            raise FinalFreezeArchiveError(f"source must be a real directory: {lexical}")
        resolved = lexical.resolve(strict=True)
        if resolved.parent != allowed_parent:
            raise FinalFreezeArchiveError(
                f"source is not an immediate child of the fixed run root: {resolved}"
            )
        roots.append(resolved)
    return roots


def _walk_directory(
    repo_root: Path,
    directory: Path,
    *,
    directories: list[str],
    files: list[dict[str, Any]],
) -> None:
    directory_stat = directory.lstat()
    if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
        raise FinalFreezeArchiveError(f"archive member is not a real directory: {directory}")
    directories.append(directory.relative_to(repo_root).as_posix())
    with os.scandir(directory) as entries:
        ordered = sorted(entries, key=lambda entry: entry.name)
    for entry in ordered:
        if any(character in entry.name for character in "\n\r\0"):
            raise FinalFreezeArchiveError(f"unsupported source filename: {entry.path!r}")
        path = Path(entry.path)
        entry_stat = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(entry_stat.st_mode):
            raise FinalFreezeArchiveError(f"symlinks are forbidden in source: {path}")
        if stat.S_ISDIR(entry_stat.st_mode):
            _walk_directory(
                repo_root,
                path,
                directories=directories,
                files=files,
            )
        elif stat.S_ISREG(entry_stat.st_mode):
            files.append(_stable_file_record(repo_root, path))
        else:
            raise FinalFreezeArchiveError(
                f"only regular files and directories may be archived: {path}"
            )


def inventory_sources(contract: ArchiveContract) -> dict[str, Any]:
    repo_root = contract.repo_root.resolve(strict=True)
    roots = _safe_source_roots(contract)
    directories: list[str] = []
    files: list[dict[str, Any]] = []
    for root in roots:
        _walk_directory(repo_root, root, directories=directories, files=files)
    directories.sort()
    files.sort(key=lambda record: record["path"])
    return {
        "source_relative_paths": list(contract.source_relative_paths),
        "directories": directories,
        "files": files,
        "file_count": len(files),
        "directory_count": len(directories),
        "total_file_bytes": sum(record["size_bytes"] for record in files),
    }


def _assert_same_inventory(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    keys = (
        "source_relative_paths",
        "directories",
        "files",
        "file_count",
        "directory_count",
        "total_file_bytes",
    )
    if any(expected.get(key) != actual.get(key) for key in keys):
        raise FinalFreezeArchiveError("source size/path/SHA inventory changed")


def _resolve_executable(name: str, fallback: str | None = None) -> str:
    resolved = shutil.which(name)
    if resolved:
        return resolved
    if fallback and Path(fallback).is_file():
        return fallback
    raise FinalFreezeArchiveError(f"required executable is unavailable: {name}")


def _run_checked(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "")
        raise FinalFreezeArchiveError(
            f"command failed: {' '.join(command)}; stderr={stderr!r}"
        ) from exc


def _normalize_tar_member(raw_name: str) -> str:
    name = raw_name.strip()
    while name.startswith("./"):
        name = name[2:]
    name = name.rstrip("/")
    pure = _validate_relative_path(name)
    return pure.as_posix()


def _list_archive_members(
    archive_path: Path,
    *,
    tar_executable: str,
    zstd_executable: str,
) -> tuple[list[str], list[str]]:
    common = [
        tar_executable,
        f"--use-compress-program={zstd_executable}",
    ]
    names_result = _run_checked([*common, "-tf", str(archive_path)])
    verbose_result = _run_checked([*common, "-tvf", str(archive_path)])
    raw_names = names_result.stdout.splitlines()
    verbose_lines = verbose_result.stdout.splitlines()
    if len(raw_names) != len(verbose_lines):
        raise FinalFreezeArchiveError("tar member and type listings disagree")
    names = [_normalize_tar_member(name) for name in raw_names]
    if len(set(names)) != len(names):
        raise FinalFreezeArchiveError("archive contains duplicate member paths")
    types: list[str] = []
    for line in verbose_lines:
        if not line:
            raise FinalFreezeArchiveError("tar returned an empty verbose member line")
        member_type = line[0]
        if member_type not in ("-", "d"):
            raise FinalFreezeArchiveError(
                f"archive contains a forbidden member type: {member_type!r}"
            )
        types.append("file" if member_type == "-" else "directory")
    return names, types


def _inventory_extracted_archive(
    extraction_root: Path,
    source_relative_paths: Sequence[str],
) -> dict[str, Any]:
    # Use the same strict walker without allowing production paths to influence
    # extraction verification.
    directories: list[str] = []
    files: list[dict[str, Any]] = []
    for relative in source_relative_paths:
        pure = _validate_relative_path(relative)
        root = extraction_root.joinpath(*pure.parts)
        if not root.is_dir() or root.is_symlink():
            raise FinalFreezeArchiveError(f"extracted source root is invalid: {relative}")
        _walk_directory(
            extraction_root,
            root,
            directories=directories,
            files=files,
        )
    directories.sort()
    files.sort(key=lambda record: record["path"])
    return {
        "source_relative_paths": list(source_relative_paths),
        "directories": directories,
        "files": files,
        "file_count": len(files),
        "directory_count": len(directories),
        "total_file_bytes": sum(record["size_bytes"] for record in files),
    }


def verify_archive_against_inventory(
    archive_path: Path,
    inventory: dict[str, Any],
    *,
    expected_archive_sha256: str | None = None,
    expected_archive_size: int | None = None,
    tar_executable: str | None = None,
    zstd_executable: str | None = None,
) -> dict[str, Any]:
    tar_program = tar_executable or _resolve_executable("tar")
    zstd_program = zstd_executable or _resolve_executable(
        "zstd", "/home/fjk/miniforge3/bin/zstd"
    )
    archive_stat = archive_path.lstat()
    if stat.S_ISLNK(archive_stat.st_mode) or not stat.S_ISREG(archive_stat.st_mode):
        raise FinalFreezeArchiveError(f"archive is not a regular file: {archive_path}")
    archive_sha256 = _sha256_file(archive_path)
    if expected_archive_sha256 and archive_sha256 != expected_archive_sha256:
        raise FinalFreezeArchiveError("archive SHA256 does not match the manifest")
    if expected_archive_size is not None and archive_stat.st_size != expected_archive_size:
        raise FinalFreezeArchiveError("archive size does not match the manifest")

    _run_checked([zstd_program, "-t", str(archive_path)])
    names, types = _list_archive_members(
        archive_path,
        tar_executable=tar_program,
        zstd_executable=zstd_program,
    )
    expected_types = {
        **{path: "directory" for path in inventory["directories"]},
        **{record["path"]: "file" for record in inventory["files"]},
    }
    actual_types = dict(zip(names, types, strict=True))
    if actual_types != expected_types:
        raise FinalFreezeArchiveError("archive member paths/types are not exact")

    with tempfile.TemporaryDirectory(prefix="final-freeze-archive-verify-") as temporary:
        extraction_root = Path(temporary).resolve()
        _run_checked(
            [
                tar_program,
                f"--use-compress-program={zstd_program}",
                "--extract",
                "--file",
                str(archive_path),
                "--directory",
                str(extraction_root),
                "--no-same-owner",
                "--no-same-permissions",
            ]
        )
        extracted = _inventory_extracted_archive(
            extraction_root, inventory["source_relative_paths"]
        )
        _assert_same_inventory(inventory, extracted)

    return {
        "status": "verified",
        "verified_at_utc": _utc_now(),
        "archive_size_bytes": archive_stat.st_size,
        "archive_sha256": archive_sha256,
        "zstd_test_passed": True,
        "tar_listing_passed": True,
        "member_paths_exact": True,
        "member_types_exact": True,
        "archived_file_size_sha256_exact": True,
        "listed_file_count": inventory["file_count"],
        "listed_directory_count": inventory["directory_count"],
    }


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FinalFreezeArchiveError(f"refusing to overwrite manifest: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FinalFreezeArchiveError(f"refusing to overwrite manifest: {path}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _replace_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _assert_production_contract(contract: ArchiveContract) -> None:
    expected = production_contract()
    if contract.source_relative_paths != expected.source_relative_paths:
        raise FinalFreezeArchiveError("production source list is not exact")
    if contract.repo_root.resolve(strict=True) != expected.repo_root.resolve(strict=True):
        raise FinalFreezeArchiveError("production repository root is not exact")
    if contract.archive_path != expected.archive_path:
        raise FinalFreezeArchiveError("production archive target is not exact")
    if contract.manifest_path != expected.manifest_path:
        raise FinalFreezeArchiveError("production manifest target is not exact")


def create_archive(
    contract: ArchiveContract,
    *,
    enforce_production_contract: bool = True,
    tar_executable: str | None = None,
    zstd_executable: str | None = None,
) -> dict[str, Any]:
    if enforce_production_contract:
        _assert_production_contract(contract)
    inventory = inventory_sources(contract)
    tar_program = tar_executable or _resolve_executable("tar")
    zstd_program = zstd_executable or _resolve_executable(
        "zstd", "/home/fjk/miniforge3/bin/zstd"
    )
    archive_path = contract.archive_path
    manifest_path = contract.manifest_path
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() or archive_path.is_symlink():
        raise FinalFreezeArchiveError(f"refusing to overwrite archive: {archive_path}")
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FinalFreezeArchiveError(f"refusing to overwrite manifest: {manifest_path}")
    partial = archive_path.with_name(f".{archive_path.name}.partial.{os.getpid()}")
    if partial.exists() or partial.is_symlink():
        raise FinalFreezeArchiveError(f"partial archive already exists: {partial}")

    installed = False
    try:
        _run_checked(
            [
                tar_program,
                f"--use-compress-program={zstd_program}",
                "--create",
                "--file",
                str(partial),
                "--directory",
                str(contract.repo_root.resolve(strict=True)),
                "--",
                *contract.source_relative_paths,
            ]
        )
        verification = verify_archive_against_inventory(
            partial,
            inventory,
            tar_executable=tar_program,
            zstd_executable=zstd_program,
        )
        current_inventory = inventory_sources(contract)
        _assert_same_inventory(inventory, current_inventory)
        os.replace(partial, archive_path)
        installed = True
        final_sha256 = _sha256_file(archive_path)
        if final_sha256 != verification["archive_sha256"]:
            raise FinalFreezeArchiveError("archive changed during final installation")
        _run_checked([zstd_program, "-t", str(archive_path)])
        verification["archive_size_bytes"] = archive_path.stat().st_size
        verification["archive_sha256"] = final_sha256
        verification["installed_path_retest_passed"] = True

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "VERIFIED_SOURCE_RETAINED",
            "created_at_utc": _utc_now(),
            "repository_root": str(contract.repo_root.resolve(strict=True)),
            "archive_absolute_path": str(archive_path),
            "manifest_absolute_path": str(manifest_path),
            "source_relative_paths": list(contract.source_relative_paths),
            "allowed_delete_realpaths": [
                str(path) for path in _safe_source_roots(contract)
            ],
            "inventory": inventory,
            "archive_verification": verification,
            "deletion": {
                "status": "not_requested",
                "required_confirmation_one": CONFIRMATION_ONE,
                "required_confirmation_two": CONFIRMATION_TWO,
            },
        }
        _write_json_new(manifest_path, manifest)
        return manifest
    except Exception:
        if partial.exists():
            partial.unlink()
        if installed and archive_path.exists():
            archive_path.unlink()
        raise


def _load_manifest(contract: ArchiveContract) -> dict[str, Any]:
    path = contract.manifest_path
    if path.is_symlink() or not path.is_file():
        raise FinalFreezeArchiveError(f"archive manifest is missing or unsafe: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalFreezeArchiveError(f"could not parse archive manifest: {path}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise FinalFreezeArchiveError("archive manifest schema mismatch")
    if manifest.get("archive_absolute_path") != str(contract.archive_path):
        raise FinalFreezeArchiveError("archive manifest target mismatch")
    if manifest.get("manifest_absolute_path") != str(contract.manifest_path):
        raise FinalFreezeArchiveError("archive manifest self-path mismatch")
    if manifest.get("source_relative_paths") != list(contract.source_relative_paths):
        raise FinalFreezeArchiveError("archive manifest source list mismatch")
    inventory = manifest.get("inventory")
    verification = manifest.get("archive_verification")
    if not isinstance(inventory, dict) or not isinstance(verification, dict):
        raise FinalFreezeArchiveError("archive manifest is incomplete")
    return manifest


def verify_manifest_archive(
    contract: ArchiveContract,
    *,
    enforce_production_contract: bool = True,
    tar_executable: str | None = None,
    zstd_executable: str | None = None,
) -> dict[str, Any]:
    if enforce_production_contract:
        _assert_production_contract(contract)
    manifest = _load_manifest(contract)
    recorded = manifest["archive_verification"]
    result = verify_archive_against_inventory(
        contract.archive_path,
        manifest["inventory"],
        expected_archive_sha256=recorded.get("archive_sha256"),
        expected_archive_size=recorded.get("archive_size_bytes"),
        tar_executable=tar_executable,
        zstd_executable=zstd_executable,
    )
    return result


def delete_archived_sources(
    contract: ArchiveContract,
    *,
    confirmation_one: str,
    confirmation_two: str,
    enforce_production_contract: bool = True,
    tar_executable: str | None = None,
    zstd_executable: str | None = None,
) -> dict[str, Any]:
    if confirmation_one != CONFIRMATION_ONE or confirmation_two != CONFIRMATION_TWO:
        raise FinalFreezeArchiveError("both exact deletion confirmations are required")
    if enforce_production_contract:
        _assert_production_contract(contract)
    manifest = _load_manifest(contract)
    if manifest.get("status") != "VERIFIED_SOURCE_RETAINED":
        raise FinalFreezeArchiveError("manifest does not authorize source deletion")

    roots = _safe_source_roots(contract)
    expected_realpaths = [str(path) for path in roots]
    if manifest.get("allowed_delete_realpaths") != expected_realpaths:
        raise FinalFreezeArchiveError("manifest delete realpaths are not exact")
    verify_manifest_archive(
        contract,
        enforce_production_contract=enforce_production_contract,
        tar_executable=tar_executable,
        zstd_executable=zstd_executable,
    )
    current_inventory = inventory_sources(contract)
    _assert_same_inventory(manifest["inventory"], current_inventory)

    # Re-resolve every target only after every archive/source check has passed.
    roots = _safe_source_roots(contract)
    if [str(path) for path in roots] != expected_realpaths:
        raise FinalFreezeArchiveError("source realpaths changed before deletion")
    for root in roots:
        shutil.rmtree(root)
    if any(root.exists() or root.is_symlink() for root in roots):
        raise FinalFreezeArchiveError("one or more fixed source directories remain")

    manifest["status"] = "VERIFIED_ARCHIVE_SOURCE_DELETED"
    manifest["deletion"] = {
        "status": "completed",
        "completed_at_utc": _utc_now(),
        "deleted_realpaths": expected_realpaths,
        "deleted_directory_count": len(roots),
        "deleted_file_count": manifest["inventory"]["file_count"],
        "deleted_file_bytes": manifest["inventory"]["total_file_bytes"],
        "confirmation_one": confirmation_one,
        "confirmation_two": confirmation_two,
    }
    _replace_json(contract.manifest_path, manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("create", help="create and fully verify the fixed archive")
    subparsers.add_parser("verify", help="reverify the archive without changing data")
    delete = subparsers.add_parser(
        "delete", help="delete only the two verified source directories"
    )
    delete.add_argument("--confirm", required=True)
    delete.add_argument("--confirm-again", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    contract = production_contract()
    try:
        if arguments.command == "create":
            result = create_archive(contract)
        elif arguments.command == "verify":
            result = verify_manifest_archive(contract)
        else:
            result = delete_archived_sources(
                contract,
                confirmation_one=arguments.confirm,
                confirmation_two=arguments.confirm_again,
            )
    except FinalFreezeArchiveError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
