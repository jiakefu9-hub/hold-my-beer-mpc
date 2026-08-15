#!/usr/bin/env python3
"""Prepare and enforce the repository-local data cleanup contract.

The default ``plan`` command is read-only with respect to experiment data.  It
hashes every source file and writes a cleanup manifest inside
``evaluation_summary``.  Archives are intentionally created outside this tool:
``verify-archive`` only validates an already-created ``tar.zst``.  Destruction
requires both ``--execute-delete`` and an exact confirmation phrase, and is
allowed only after archive integrity, member paths, current source hashes, and
Git tracking state have all been checked again.

Archive source roots and cleanup targets are deliberately enumerated below.
There is no wildcard-driven deletion path, and the top-level ``evaluation``,
``disturbance_learning/data``, and ``disturbance_learning/artifacts``
directories can never be deleted by this program.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "evaluation_summary"
    / "full_task_template_v2_final_freeze"
    / "cleanup_manifest.json"
)
DEFAULT_ARCHIVE_ROOT = Path(
    "/home/fjk/g1_ws/disturbance-lab-archives/20260815_pre_cleanup"
)
DEFAULT_EVIDENCE_ROOT = (
    REPO_ROOT / "evaluation_summary" / "full_task_template_v2_final_freeze"
)
DEFAULT_FINAL_RUN_ARCHIVE = Path(
    "/home/fjk/g1_ws/disturbance-lab-archives/20260815_pre_cleanup/"
    "final_freeze_full_runs.tar.zst"
)
FINAL_RUN_SOURCE_PATHS = (
    "evaluation/t2_full_task_closed_loop/20260815_231454_final_freeze",
    (
        "evaluation/t2_full_task_closed_loop/"
        "20260815_231555_final_freeze_heldout_pair_02_minus"
    ),
)
DELETE_CONFIRMATION = "DELETE_VERIFIED_ARCHIVED_SOURCES"
SCHEMA_VERSION = "disturbance-lab-data-cleanup-v1"
CHECKPOINT_BRANCH = "archive/pre-cleanup-full-task-v2-20260815"
CHECKPOINT_COMMIT = "70eb33b51656b958648ea013bc9bd45aa72dfa73"
CHECKPOINT_TAG = "checkpoint/full-task-v2-24ms-20260815"

CORE_ASSETS = (
    "disturbance_learning/data/full_task_template_v2/20260815_162850/"
    "full_task_template.npz",
    "disturbance_learning/data/full_task_template_v2/20260815_162850/"
    "full_task_template_manifest.json",
    "disturbance_learning/data/full_task_template_v2/20260815_162850/episodes/"
    "heldout_pair_02_minus/episode_manifest.json",
)

EVALUATION_SOURCE_MEMBERS = (
    "evaluation/fixed_startup_pd_cpu7_controlled",
    "evaluation/fixed_startup_pd_handoff_full",
    "evaluation/fixed_startup_pd_handoff_full_final",
    "evaluation/fixed_startup_pd_handoff_report",
    "evaluation/fixed_startup_pd_handoff_short",
    "evaluation/fixed_startup_pd_handoff_short_final",
    "evaluation/generalization_legacy_smoke_20260811",
    "evaluation/hybrid_generalization_isolation_20260811",
    "evaluation/hybrid_generalization_pilot_20260811",
    "evaluation/hybrid_generalization_validation_20260811_1750",
    "evaluation/irq_monitor_smoke_20260811",
    "evaluation/left_fixed_right_mpc",
    "evaluation/neural_closed_loop_20260811_first",
    "evaluation/neural_closed_loop_20260811_optimized",
    "evaluation/neural_predictor_smoke",
    "evaluation/payload_model_smoke",
    "evaluation/readiness_affinity_pilot_20260811",
    "evaluation/readiness_blocker_diagnostics_final_20260811",
    "evaluation/readiness_blockers_20260811",
    "evaluation/readiness_cpu_pilot_20260811",
    "evaluation/readiness_gate_diagnosis_20260811",
    "evaluation/readiness_gate_v2_20260811",
    "evaluation/readiness_profile_pilot_20260811",
    "evaluation/readiness_timing_performance_20260811",
    "evaluation/readiness_timing_performance_worker5_20260811",
    "evaluation/readiness_timing_powersave_20260811",
    "evaluation/real_robot_readiness_20260811",
    "evaluation/real_robot_readiness_final_20260811",
    "evaluation/real_robot_readiness_timing_gate_final_20260811",
    "evaluation/realtime_rr10_validation_20260811",
    "evaluation/realtime_runtime_smoke",
    "evaluation/t1a_full_task_smoke",
    "evaluation/t2_full_task_closed_loop",
    "evaluation/t2_full_task_template_online",
    "evaluation/t2_safe_hold_acceptance",
    "evaluation/target_rt_preempt_rt_20260811",
    "evaluation/target_rt_preempt_rt_irq_verified_20260811",
    "evaluation/target_runtime_preparation_smoke",
)


@dataclass(frozen=True)
class ArchiveGroupSpec:
    group_id: str
    archive_filename: str
    source_members: tuple[str, ...]
    reason: str


ARCHIVE_GROUP_SPECS = (
    ArchiveGroupSpec(
        group_id="neural_disturbance_inputs",
        archive_filename="neural_disturbance_inputs.tar.zst",
        source_members=(
            "disturbance_learning/data/b1_validation_seed0_raw.npz",
            "disturbance_learning/data/b1_validation_seed0_validation.json",
            "disturbance_learning/data/b1_validation_seed0_windows.npz",
            "disturbance_learning/data/b2_mlp_episodes",
            "disturbance_learning/artifacts/b2_mlp_baseline",
            "disturbance_learning/artifacts/hybrid_residual_mlp",
        ),
        reason=(
            "Frozen neural-disturbance datasets and checkpoints are no longer "
            "used by the final full-task-template runtime."
        ),
    ),
    ArchiveGroupSpec(
        group_id="full_task_template_v2_build_source",
        archive_filename="full_task_template_v2_build_source.tar.zst",
        source_members=(
            "disturbance_learning/data/full_task_template_v2/20260815_162850",
        ),
        reason=(
            "Preserve the complete 11-build/4-held-out source evidence outside "
            "the repository; the three Git-tracked runtime/reproduction assets "
            "remain active and are never cleanup targets."
        ),
    ),
    ArchiveGroupSpec(
        group_id="obsolete_full_task_v1_v3",
        archive_filename="obsolete_full_task_v1_v3.tar.zst",
        source_members=(
            "disturbance_learning/data/full_task_template_v1",
            "disturbance_learning/data/full_task_template_v3",
        ),
        reason=(
            "Archive superseded v1 and failed dynamic-arming/v3 evidence before "
            "removing it from the active data tree."
        ),
    ),
    ArchiveGroupSpec(
        group_id="evaluation_pre_cleanup",
        archive_filename="evaluation_pre_cleanup.tar.zst",
        source_members=EVALUATION_SOURCE_MEMBERS,
        reason=(
            "Archive the complete pre-cleanup evaluation tree after compact "
            "final evidence has been independently built and validated."
        ),
    ),
)

ALLOWED_CLEANUP_ROOTS = (
    "evaluation",
    "disturbance_learning/data",
    "disturbance_learning/artifacts",
)


class CleanupError(RuntimeError):
    """A fail-closed cleanup-contract violation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise CleanupError(f"path escapes repository: {path}") from exc


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_declared_relative_path(relative_path: str) -> PurePosixPath:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise CleanupError(f"unsafe relative path: {relative_path!r}")
    return pure


def assert_safe_cleanup_path(
    repo_root: Path,
    relative_path: str,
    *,
    allow_missing: bool,
) -> Path:
    """Resolve one manifest target and prove it is below an allowed leaf root."""

    pure = _validate_declared_relative_path(relative_path)
    lexical = repo_root.joinpath(*pure.parts)
    if not allow_missing and not lexical.exists():
        raise CleanupError(f"cleanup target is missing: {relative_path}")

    resolved_repo = repo_root.resolve(strict=True)
    resolved_target = lexical.resolve(strict=not allow_missing)
    allowed_roots = tuple((resolved_repo / root).resolve(strict=True) for root in ALLOWED_CLEANUP_ROOTS)
    forbidden = {resolved_repo, *allowed_roots}
    if resolved_target in forbidden:
        raise CleanupError(f"cleanup root deletion is forbidden: {relative_path}")
    if not any(_is_relative_to(resolved_target, root) for root in allowed_roots):
        raise CleanupError(f"cleanup target is outside allowed roots: {relative_path}")
    return resolved_target


def _walk_source_member(repo_root: Path, relative_member: str) -> tuple[list[Path], list[Path]]:
    pure = _validate_declared_relative_path(relative_member)
    member = repo_root.joinpath(*pure.parts)
    if not member.exists():
        raise CleanupError(f"declared source member is missing: {relative_member}")
    if member.is_symlink():
        raise CleanupError(f"source member may not be a symlink: {relative_member}")
    if member.is_file():
        mode = member.stat().st_mode
        if not stat.S_ISREG(mode):
            raise CleanupError(f"source member is not a regular file: {relative_member}")
        return [member], []
    if not member.is_dir():
        raise CleanupError(f"unsupported source member type: {relative_member}")

    files: list[Path] = []
    directories: list[Path] = [member]
    for current, dirnames, filenames in os.walk(member, followlinks=False):
        current_path = Path(current)
        dirnames.sort()
        filenames.sort()
        for dirname in dirnames:
            candidate = current_path / dirname
            if candidate.is_symlink():
                raise CleanupError(
                    f"source tree contains directory symlink: {_relative_path(repo_root, candidate)}"
                )
            directories.append(candidate)
        for filename in filenames:
            candidate = current_path / filename
            if candidate.is_symlink() or not candidate.is_file():
                raise CleanupError(
                    f"source tree contains non-regular file: {_relative_path(repo_root, candidate)}"
                )
            files.append(candidate)
    return files, directories


def _file_record(repo_root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _relative_path(repo_root, path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _contains_preserved_asset(directory: Path, preserved: Sequence[Path]) -> bool:
    return any(path == directory or _is_relative_to(path, directory) for path in preserved)


def _tracked_paths(repo_root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        entry.decode("utf-8")
        for entry in result.stdout.split(b"\0")
        if entry
    }


def _git_text(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _git_tree_usage(repo_root: Path, commit: str, relative_root: str) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "-l", "--full-tree", commit, "--", relative_root],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    file_count = 0
    file_bytes = 0
    for line in result.stdout.splitlines():
        metadata, _ = line.split("\t", 1)
        fields = metadata.split()
        if len(fields) != 4 or fields[1] != "blob" or fields[3] == "-":
            raise CleanupError(f"unexpected git ls-tree record: {line}")
        file_count += 1
        file_bytes += int(fields[3])
    return {
        "relative_path": relative_root,
        "measurement": "Git blob bytes at recovery checkpoint",
        "file_count": file_count,
        "regular_file_bytes": file_bytes,
    }


def _tree_usage(
    root: Path,
    *,
    excluded_files: Iterable[Path] = (),
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise CleanupError(f"usage root is missing, unsafe, or not a directory: {root}")
    excluded = {path.resolve(strict=False) for path in excluded_files}
    file_count = 0
    directory_count = 1
    regular_file_bytes = 0
    apparent_bytes = root.stat().st_size
    allocated_bytes = root.stat().st_blocks * 512
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        dirnames.sort()
        filenames.sort()
        for dirname in dirnames:
            path = current_path / dirname
            if path.is_symlink() or not path.is_dir():
                raise CleanupError(f"usage tree contains unsafe directory: {path}")
            info = path.stat()
            directory_count += 1
            apparent_bytes += info.st_size
            allocated_bytes += info.st_blocks * 512
        for filename in filenames:
            path = current_path / filename
            if path.resolve(strict=False) in excluded:
                continue
            if path.is_symlink() or not path.is_file():
                raise CleanupError(f"usage tree contains unsafe file: {path}")
            info = path.stat()
            file_count += 1
            regular_file_bytes += info.st_size
            apparent_bytes += info.st_size
            allocated_bytes += info.st_blocks * 512
    return {
        "absolute_path": str(root.resolve(strict=True)),
        "file_count": file_count,
        "directory_count_including_root": directory_count,
        "regular_file_bytes": regular_file_bytes,
        "apparent_bytes_including_directories": apparent_bytes,
        "allocated_bytes": allocated_bytes,
    }


def build_cleanup_plan(repo_root: Path, archive_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    archive_root = archive_root.expanduser().resolve(strict=False)
    core_paths = [repo_root / path for path in CORE_ASSETS]
    core_records = []
    for path in core_paths:
        if not path.is_file() or path.is_symlink():
            raise CleanupError(f"required core asset is missing or unsafe: {path}")
        record = _file_record(repo_root, path)
        record["classification"] = "preserve_in_git"
        record["cleanup_eligible"] = False
        core_records.append(record)

    tracked = _tracked_paths(repo_root)
    untracked_core = sorted(set(CORE_ASSETS) - tracked)
    if untracked_core:
        raise CleanupError(f"core assets are not tracked by Git: {untracked_core}")

    groups: list[dict[str, Any]] = []
    globally_seen_files: set[str] = set()
    globally_seen_members: set[str] = set()
    all_cleanup_files: set[str] = set()
    all_source_files: set[str] = set()

    for spec in ARCHIVE_GROUP_SPECS:
        source_files: list[Path] = []
        source_directories: list[Path] = []
        source_member_records: list[dict[str, Any]] = []
        for member in spec.source_members:
            if member in globally_seen_members:
                raise CleanupError(f"duplicate archive source member: {member}")
            globally_seen_members.add(member)
            files, directories = _walk_source_member(repo_root, member)
            source_files.extend(files)
            source_directories.extend(directories)
            member_path = repo_root / member
            source_member_records.append(
                {
                    "relative_path": member,
                    "absolute_path": str(member_path.resolve(strict=True)),
                    "kind": "directory" if member_path.is_dir() else "file",
                    "file_count": len(files),
                    "total_file_bytes": sum(path.stat().st_size for path in files),
                    "classification": (
                        "archive_then_prune_to_preserved_core"
                        if any(
                            core == member_path or _is_relative_to(core, member_path)
                            for core in core_paths
                        )
                        else "archive_then_delete"
                    ),
                    "reason": spec.reason,
                }
            )

        records = [_file_record(repo_root, path) for path in sorted(set(source_files))]
        group_paths = {record["path"] for record in records}
        overlap = globally_seen_files.intersection(group_paths)
        if overlap:
            raise CleanupError(f"archive groups overlap: {sorted(overlap)[:5]}")
        globally_seen_files.update(group_paths)
        all_source_files.update(group_paths)

        cleanup_files = sorted(group_paths - set(CORE_ASSETS))
        tracked_cleanup = sorted(set(cleanup_files).intersection(tracked))
        if tracked_cleanup:
            raise CleanupError(
                f"planned cleanup includes Git-tracked files in {spec.group_id}: "
                f"{tracked_cleanup[:10]}"
            )
        all_cleanup_files.update(cleanup_files)

        cleanup_directories: list[str] = []
        for directory in sorted(
            set(source_directories),
            key=lambda path: (len(path.parts), path.as_posix()),
            reverse=True,
        ):
            if _contains_preserved_asset(directory, core_paths):
                continue
            relative = _relative_path(repo_root, directory)
            assert_safe_cleanup_path(repo_root, relative, allow_missing=False)
            cleanup_directories.append(relative)

        archive_path = archive_root / spec.archive_filename
        groups.append(
            {
                "group_id": spec.group_id,
                "classification": "archive_then_delete_except_preserved_core",
                "reason": spec.reason,
                "archive_path": str(archive_path),
                "source_members": list(spec.source_members),
                "source_member_records": source_member_records,
                "source_file_count": len(records),
                "source_total_bytes": sum(record["size_bytes"] for record in records),
                "source_files": records,
                "cleanup_file_count": len(cleanup_files),
                "cleanup_total_bytes": sum(
                    record["size_bytes"]
                    for record in records
                    if record["path"] in set(cleanup_files)
                ),
                "cleanup_files": cleanup_files,
                "cleanup_directories_deepest_first": cleanup_directories,
                "archive_creation_contract": {
                    "working_directory": str(repo_root),
                    "required_relative_members": list(spec.source_members),
                    "command_argv_template": [
                        "tar",
                        "--zstd",
                        "-cf",
                        str(archive_path),
                        "-C",
                        str(repo_root),
                        *spec.source_members,
                    ],
                    "note": (
                        "Create outside this tool. Do not add or remove members; "
                        "verification requires an exact regular-file path set."
                    ),
                },
                "archive_verification": {
                    "status": "not_verified",
                    "verified_at_utc": None,
                    "archive_sha256": None,
                    "archive_size_bytes": None,
                    "zstd_test_passed": False,
                    "tar_listing_passed": False,
                    "listed_regular_file_count": None,
                    "listed_directory_entry_count": None,
                    "member_paths_exact": False,
                    "member_types_safe": False,
                },
                "deletion": {
                    "status": "not_started",
                    "completed_at_utc": None,
                    "deleted_file_count": 0,
                    "deleted_directory_count": 0,
                },
            }
        )

    allowed_files: set[str] = set()
    for allowed_root in ALLOWED_CLEANUP_ROOTS:
        root = repo_root / allowed_root
        if not root.exists():
            continue
        files, _ = _walk_source_member(repo_root, allowed_root)
        allowed_files.update(_relative_path(repo_root, path) for path in files)
    unclassified = sorted(allowed_files - all_source_files)
    if unclassified:
        raise CleanupError(
            "ignored data/evaluation files are not classified by the explicit plan: "
            f"{unclassified[:20]}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "repo_root": str(repo_root),
        "archive_root": str(archive_root),
        "git_context": {
            "cleanup_branch": _git_text(repo_root, "branch", "--show-current"),
            "plan_head": _git_text(repo_root, "rev-parse", "HEAD"),
            "recovery_checkpoint_branch": CHECKPOINT_BRANCH,
            "recovery_checkpoint_commit": CHECKPOINT_COMMIT,
            "recovery_checkpoint_tag": CHECKPOINT_TAG,
            "recovery_checkpoint_reachable": (
                _git_text(repo_root, "rev-parse", CHECKPOINT_TAG + "^{commit}")
                == CHECKPOINT_COMMIT
            ),
        },
        "policy": {
            "plan_is_non_destructive": True,
            "archive_creation_is_external": True,
            "archive_verification_required_before_delete": True,
            "source_hash_recheck_required_before_delete": True,
            "tracked_cleanup_files_forbidden": True,
            "symlinks_and_special_files_forbidden": True,
            "allowed_cleanup_roots": list(ALLOWED_CLEANUP_ROOTS),
            "forbidden_directory_targets": [
                str(repo_root),
                *(str(repo_root / path) for path in ALLOWED_CLEANUP_ROOTS),
            ],
            "delete_confirmation_phrase": DELETE_CONFIRMATION,
        },
        "core_assets": core_records,
        "archive_groups": groups,
        "summary": {
            "archive_group_count": len(groups),
            "source_file_count": len(all_source_files),
            "source_total_bytes": sum(
                group["source_total_bytes"] for group in groups
            ),
            "cleanup_file_count": len(all_cleanup_files),
            "cleanup_total_bytes": sum(
                group["cleanup_total_bytes"] for group in groups
            ),
            "preserved_core_file_count": len(core_records),
            "unclassified_file_count": 0,
        },
        "unclassified_files": [],
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=False)
        stream.write("\n")
    temporary.replace(path)


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CleanupError(
            f"unsupported cleanup manifest schema: {manifest.get('schema_version')!r}"
        )
    if Path(manifest.get("repo_root", "")).resolve(strict=True) != REPO_ROOT.resolve(strict=True):
        raise CleanupError("cleanup manifest repository does not match this checkout")
    return manifest


def _select_group(manifest: dict[str, Any], group_id: str) -> dict[str, Any]:
    matches = [group for group in manifest["archive_groups"] if group["group_id"] == group_id]
    if len(matches) != 1:
        raise CleanupError(f"unknown or duplicate archive group: {group_id}")
    return matches[0]


def _find_executable(explicit: str | None, candidates: Sequence[str]) -> str:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise CleanupError(f"executable is not available: {explicit}")
        return str(path.resolve())
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    raise CleanupError(f"required executable was not found: {candidates}")


def normalize_tar_listing(lines: Iterable[str]) -> tuple[list[str], list[str]]:
    files: list[str] = []
    directories: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        entry = raw.rstrip("\r\n")
        if entry.startswith("./"):
            entry = entry[2:]
        is_directory = entry.endswith("/")
        normalized = entry.rstrip("/")
        if not normalized:
            continue
        pure = _validate_declared_relative_path(normalized)
        rendered = pure.as_posix()
        key = rendered + ("/" if is_directory else "")
        if key in seen:
            raise CleanupError(f"archive contains a duplicate member: {key}")
        seen.add(key)
        if is_directory:
            directories.append(rendered)
        else:
            files.append(rendered)
    return files, directories


def assert_exact_archive_members(actual_files: Sequence[str], expected_files: Sequence[str]) -> None:
    actual = set(actual_files)
    expected = set(expected_files)
    if len(actual_files) != len(actual):
        raise CleanupError("archive file listing contains duplicate paths")
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise CleanupError(
            "archive member mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )


def assert_safe_tar_member_types(
    listing_lines: Sequence[str],
    verbose_lines: Sequence[str],
) -> None:
    """Reject archive links/devices and prove every member is file or directory."""

    if len(listing_lines) != len(verbose_lines):
        raise CleanupError(
            "tar normal and verbose listings disagree on member count: "
            f"{len(listing_lines)} != {len(verbose_lines)}"
        )
    for listing, verbose in zip(listing_lines, verbose_lines):
        entry = listing.rstrip("\r\n")
        if not verbose:
            raise CleanupError(f"tar verbose listing is empty for member: {entry}")
        expected_type = "d" if entry.endswith("/") else "-"
        if verbose[0] != expected_type:
            raise CleanupError(
                "archive member is not a regular file/directory: "
                f"member={entry!r}, tar_type={verbose[0]!r}"
            )


def _run_checked(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def verify_group_archive(
    manifest: dict[str, Any],
    group_id: str,
    *,
    zstd_executable: str | None,
    tar_executable: str | None,
) -> dict[str, Any]:
    group = _select_group(manifest, group_id)
    archive_root = Path(manifest["archive_root"]).resolve(strict=True)
    archive = Path(group["archive_path"])
    if archive.is_symlink() or not archive.is_file():
        raise CleanupError(f"archive is missing, non-regular, or a symlink: {archive}")
    resolved_archive = archive.resolve(strict=True)
    if not _is_relative_to(resolved_archive, archive_root) or resolved_archive.parent != archive_root:
        raise CleanupError(f"archive path is outside the declared archive root: {archive}")

    zstd = _find_executable(
        zstd_executable,
        ("zstd", "/home/fjk/miniforge3/bin/zstd"),
    )
    tar = _find_executable(tar_executable, ("tar",))
    try:
        zstd_result = _run_checked((zstd, "-t", str(resolved_archive)))
        tar_result = _run_checked(
            (
                tar,
                f"--use-compress-program={zstd}",
                "-tf",
                str(resolved_archive),
            )
        )
        tar_verbose_result = _run_checked(
            (
                tar,
                f"--use-compress-program={zstd}",
                "-tvf",
                str(resolved_archive),
            )
        )
    except subprocess.CalledProcessError as exc:
        raise CleanupError(
            f"archive verification command failed ({exc.returncode}): "
            f"{' '.join(exc.cmd)}\n{exc.stderr}"
        ) from exc

    listing_lines = tar_result.stdout.splitlines()
    verbose_lines = tar_verbose_result.stdout.splitlines()
    assert_safe_tar_member_types(listing_lines, verbose_lines)
    listed_files, listed_directories = normalize_tar_listing(listing_lines)
    expected_files = [record["path"] for record in group["source_files"]]
    assert_exact_archive_members(listed_files, expected_files)
    return {
        "status": "verified",
        "verified_at_utc": _utc_now(),
        "archive_sha256": sha256_file(resolved_archive),
        "archive_size_bytes": resolved_archive.stat().st_size,
        "zstd_test_passed": zstd_result.returncode == 0,
        "tar_listing_passed": tar_result.returncode == 0,
        "listed_regular_file_count": len(listed_files),
        "listed_directory_entry_count": len(listed_directories),
        "member_paths_exact": True,
        "member_types_safe": True,
        "zstd_executable": zstd,
        "tar_executable": tar,
    }


def _assert_current_group_matches_plan(repo_root: Path, group: dict[str, Any]) -> None:
    expected_records = {record["path"]: record for record in group["source_files"]}
    actual_paths: set[str] = set()
    for member in group["source_members"]:
        files, _ = _walk_source_member(repo_root, member)
        actual_paths.update(_relative_path(repo_root, path) for path in files)
    if actual_paths != set(expected_records):
        raise CleanupError(
            f"source membership changed for {group['group_id']}; regenerate and rearchive"
        )
    for relative_path, expected in expected_records.items():
        path = repo_root / relative_path
        if path.stat().st_size != expected["size_bytes"]:
            raise CleanupError(f"source size changed: {relative_path}")
        if sha256_file(path) != expected["sha256"]:
            raise CleanupError(f"source checksum changed: {relative_path}")


def _assert_core_assets_unchanged(repo_root: Path, manifest: dict[str, Any]) -> None:
    for record in manifest["core_assets"]:
        path = repo_root / record["path"]
        if not path.is_file() or path.is_symlink():
            raise CleanupError(f"core asset is missing or unsafe: {record['path']}")
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise CleanupError(f"core asset changed after planning: {record['path']}")


def delete_exact_manifest_targets(
    repo_root: Path,
    cleanup_files: Sequence[str],
    cleanup_directories_deepest_first: Sequence[str],
) -> tuple[int, int]:
    """Delete only explicit manifest entries after all paths pass prefix guards."""

    file_targets = [
        assert_safe_cleanup_path(repo_root, path, allow_missing=False)
        for path in cleanup_files
    ]
    directory_targets = [
        assert_safe_cleanup_path(repo_root, path, allow_missing=False)
        for path in cleanup_directories_deepest_first
    ]
    for path in file_targets:
        if path.is_symlink() or not path.is_file():
            raise CleanupError(f"refusing non-regular cleanup file: {path}")
    for path in directory_targets:
        if path.is_symlink() or not path.is_dir():
            raise CleanupError(f"refusing non-directory cleanup path: {path}")

    for path in file_targets:
        path.unlink()
    removed_directories = 0
    for path in directory_targets:
        try:
            path.rmdir()
        except OSError as exc:
            raise CleanupError(
                f"planned directory is not empty after exact file deletion: {path}"
            ) from exc
        removed_directories += 1
    return len(file_targets), removed_directories


def execute_group_delete(
    manifest: dict[str, Any],
    group_id: str,
    *,
    zstd_executable: str | None,
    tar_executable: str | None,
) -> dict[str, int]:
    repo_root = Path(manifest["repo_root"]).resolve(strict=True)
    group = _select_group(manifest, group_id)
    recorded_verification = group["archive_verification"]
    if recorded_verification.get("status") != "verified":
        raise CleanupError(f"archive group has not been verified: {group_id}")

    current_verification = verify_group_archive(
        manifest,
        group_id,
        zstd_executable=zstd_executable,
        tar_executable=tar_executable,
    )
    if current_verification["archive_sha256"] != recorded_verification.get("archive_sha256"):
        raise CleanupError(f"archive checksum changed after verification: {group_id}")
    _assert_core_assets_unchanged(repo_root, manifest)
    _assert_current_group_matches_plan(repo_root, group)

    tracked = _tracked_paths(repo_root)
    tracked_cleanup = sorted(set(group["cleanup_files"]).intersection(tracked))
    if tracked_cleanup:
        raise CleanupError(f"refusing to delete Git-tracked files: {tracked_cleanup[:10]}")

    deleted_files, deleted_directories = delete_exact_manifest_targets(
        repo_root,
        group["cleanup_files"],
        group["cleanup_directories_deepest_first"],
    )
    return {
        "deleted_file_count": deleted_files,
        "deleted_directory_count": deleted_directories,
    }


def _pre_cleanup_usage(manifest: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    roots: dict[str, dict[str, Any]] = {}
    for relative_root in ALLOWED_CLEANUP_ROOTS:
        prefix = relative_root + "/"
        records = [
            record
            for group in manifest["archive_groups"]
            for record in group["source_files"]
            if record["path"].startswith(prefix)
        ]
        roots[relative_root] = {
            "absolute_path": str((repo_root / relative_root).resolve(strict=True)),
            "measurement": "regular-file bytes reconstructed from the pre-delete source manifest",
            "file_count": len(records),
            "regular_file_bytes": sum(int(record["size_bytes"]) for record in records),
        }
    roots["evaluation_summary"] = _git_tree_usage(
        repo_root,
        manifest["git_context"]["recovery_checkpoint_commit"],
        "evaluation_summary",
    )
    return {
        "measurement_basis": (
            "Exact regular-file content bytes. Deleted-directory inode sizes are not "
            "reconstructed; evaluation_summary is measured from the recovery checkpoint."
        ),
        "roots": roots,
        "cleanup_scope_regular_file_bytes": sum(
            roots[root]["regular_file_bytes"] for root in ALLOWED_CLEANUP_ROOTS
        ),
        "cleanup_scope_file_count": sum(
            roots[root]["file_count"] for root in ALLOWED_CLEANUP_ROOTS
        ),
        "selected_repository_regular_file_bytes": sum(
            record["regular_file_bytes"] for record in roots.values()
        ),
        "selected_repository_file_count": sum(record["file_count"] for record in roots.values()),
    }


def _assert_group_deleted(repo_root: Path, group: dict[str, Any]) -> dict[str, Any]:
    deletion = group.get("deletion", {})
    if deletion.get("status") != "completed":
        raise CleanupError(f"cleanup group is not completed: {group['group_id']}")
    expected_file_count = int(group["cleanup_file_count"])
    expected_directory_count = len(group["cleanup_directories_deepest_first"])
    if int(deletion.get("deleted_file_count", -1)) != expected_file_count:
        raise CleanupError(f"deleted-file count mismatch: {group['group_id']}")
    if int(deletion.get("deleted_directory_count", -1)) != expected_directory_count:
        raise CleanupError(f"deleted-directory count mismatch: {group['group_id']}")

    for relative_path in group["cleanup_files"]:
        path = assert_safe_cleanup_path(repo_root, relative_path, allow_missing=True)
        if path.exists() or path.is_symlink():
            raise CleanupError(f"planned cleanup file still exists: {relative_path}")
    for relative_path in group["cleanup_directories_deepest_first"]:
        path = assert_safe_cleanup_path(repo_root, relative_path, allow_missing=True)
        if path.exists() or path.is_symlink():
            raise CleanupError(f"planned cleanup directory still exists: {relative_path}")
    return {
        "group_id": group["group_id"],
        "status": "PASS",
        "completed_at_utc": deletion["completed_at_utc"],
        "planned_cleanup_file_count": expected_file_count,
        "planned_cleanup_directory_count": expected_directory_count,
        "deleted_file_count": int(deletion["deleted_file_count"]),
        "deleted_directory_count": int(deletion["deleted_directory_count"]),
        "deleted_regular_file_bytes": int(group["cleanup_total_bytes"]),
        "all_planned_cleanup_files_absent": True,
        "all_planned_cleanup_directories_absent": True,
    }


def _audit_archive_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    archive_root = Path(manifest["archive_root"])
    if archive_root.is_symlink() or not archive_root.is_dir():
        raise CleanupError(f"archive root is missing or unsafe: {archive_root}")
    resolved_root = archive_root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    for group in manifest["archive_groups"]:
        verification = group.get("archive_verification", {})
        required_flags = (
            "zstd_test_passed",
            "tar_listing_passed",
            "member_paths_exact",
            "member_types_safe",
        )
        if verification.get("status") != "verified" or not all(
            verification.get(flag) is True for flag in required_flags
        ):
            raise CleanupError(f"archive group is not fully verified: {group['group_id']}")
        archive = Path(group["archive_path"])
        if archive.is_symlink() or not archive.is_file():
            raise CleanupError(f"verified archive is missing or unsafe: {archive}")
        resolved_archive = archive.resolve(strict=True)
        if resolved_archive.parent != resolved_root:
            raise CleanupError(f"verified archive moved outside archive root: {archive}")
        current_size = resolved_archive.stat().st_size
        current_sha256 = sha256_file(resolved_archive)
        if current_size != int(verification["archive_size_bytes"]):
            raise CleanupError(f"verified archive size changed: {archive}")
        if current_sha256 != verification["archive_sha256"]:
            raise CleanupError(f"verified archive checksum changed: {archive}")
        records.append(
            {
                "group_id": group["group_id"],
                "archive_absolute_path": str(resolved_archive),
                "archive_size_bytes": current_size,
                "archive_sha256": current_sha256,
                "source_file_count": int(group["source_file_count"]),
                "source_regular_file_bytes": int(group["source_total_bytes"]),
                "listed_regular_file_count": int(
                    verification["listed_regular_file_count"]
                ),
                "listed_directory_entry_count": int(
                    verification["listed_directory_entry_count"]
                ),
                "zstd_test_passed": True,
                "tar_listing_passed": True,
                "member_paths_exact": True,
                "member_types_safe": True,
                "archive_still_present": True,
                "archive_checksum_unchanged": True,
            }
        )
    return {
        "status": "PASS",
        "archive_root": str(resolved_root),
        "archive_count": len(records),
        "archive_total_bytes": sum(record["archive_size_bytes"] for record in records),
        "source_regular_file_total_bytes": sum(
            record["source_regular_file_bytes"] for record in records
        ),
        "all_archives_present_and_checksum_unchanged": True,
        "archives": records,
    }


def _audit_additional_final_run_evidence(
    repo_root: Path,
    final_runs_root: Path,
    *,
    expected_archive_path: Path,
) -> dict[str, Any]:
    """Strictly audit the separately-built two-run freeze evidence package."""

    final_runs_root = final_runs_root.resolve(strict=True)
    file_manifest_path = final_runs_root / "final_freeze_file_manifest.json"
    archive_manifest_path = final_runs_root / "final_freeze_archive_manifest.json"
    for path in (file_manifest_path, archive_manifest_path):
        if path.is_symlink() or not path.is_file():
            raise CleanupError(f"final-run manifest is missing or unsafe: {path}")

    file_manifest = json.loads(file_manifest_path.read_text(encoding="utf-8"))
    if (
        file_manifest.get("schema_version")
        != "full_task_template_v2_final_freeze_two_run_files_v1"
    ):
        raise CleanupError("final-run file manifest schema mismatch")
    if file_manifest.get("status") != "PASS":
        raise CleanupError("final-run file manifest is not PASS")
    if file_manifest.get("output_repository_path") != _relative_path(
        repo_root, final_runs_root
    ):
        raise CleanupError("final-run output root mismatch")
    validation = file_manifest.get("validation", {})
    if not validation or not all(value is True for value in validation.values()):
        raise CleanupError("final-run validation gates are not all PASS")

    builder = file_manifest.get("builder", {})
    builder_relative = builder.get("repository_path", "")
    builder_path = repo_root / builder_relative
    if (
        builder_path.is_symlink()
        or not builder_path.is_file()
        or not _is_relative_to(builder_path.resolve(strict=True), repo_root)
        or sha256_file(builder_path) != builder.get("sha256")
    ):
        raise CleanupError("final-run evidence builder checksum mismatch")

    declared_paths: set[str] = set()
    for record in [
        *file_manifest.get("copied_files", []),
        *file_manifest.get("generated_files", []),
    ]:
        package_path = record.get("output_package_path", "")
        pure = _validate_declared_relative_path(package_path)
        relative_path = (PurePosixPath(_relative_path(repo_root, final_runs_root)) / pure).as_posix()
        if relative_path in declared_paths:
            raise CleanupError(f"duplicate final-run output: {relative_path}")
        lexical_path = repo_root.joinpath(*PurePosixPath(relative_path).parts)
        if lexical_path.is_symlink() or not lexical_path.is_file():
            raise CleanupError(f"final-run output is missing or unsafe: {relative_path}")
        path = lexical_path.resolve(strict=True)
        if not _is_relative_to(path, final_runs_root):
            raise CleanupError(f"final-run output escapes package: {relative_path}")
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise CleanupError(f"final-run output size mismatch: {relative_path}")
        if sha256_file(path) != record.get("sha256"):
            raise CleanupError(f"final-run output checksum mismatch: {relative_path}")
        declared_paths.add(relative_path)

    archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
    if (
        archive_manifest.get("schema_version")
        != "disturbance-lab-final-freeze-run-archive-v1"
    ):
        raise CleanupError("final-run archive manifest schema mismatch")
    if archive_manifest.get("status") != "VERIFIED_ARCHIVE_SOURCE_DELETED":
        raise CleanupError("final-run archive manifest status mismatch")
    if archive_manifest.get("source_relative_paths") != list(FINAL_RUN_SOURCE_PATHS):
        raise CleanupError("final-run archive source list mismatch")

    archive_path = Path(archive_manifest.get("archive_absolute_path", ""))
    if archive_path != expected_archive_path:
        raise CleanupError("final-run archive path mismatch")
    if archive_path.is_symlink() or not archive_path.is_file():
        raise CleanupError("final-run archive is missing or unsafe")
    archive_verification = archive_manifest.get("archive_verification", {})
    required_archive_gates = (
        "zstd_test_passed",
        "tar_listing_passed",
        "member_paths_exact",
        "member_types_exact",
        "archived_file_size_sha256_exact",
    )
    if archive_verification.get("status") != "verified" or not all(
        archive_verification.get(field) is True for field in required_archive_gates
    ):
        raise CleanupError("final-run archive verification gates are incomplete")
    if archive_path.stat().st_size != int(
        archive_verification.get("archive_size_bytes", -1)
    ):
        raise CleanupError("final-run archive size mismatch")
    if sha256_file(archive_path) != archive_verification.get("archive_sha256"):
        raise CleanupError("final-run archive checksum mismatch")

    expected_realpaths = [str((repo_root / path).resolve(strict=False)) for path in FINAL_RUN_SOURCE_PATHS]
    deletion = archive_manifest.get("deletion", {})
    if deletion.get("status") != "completed":
        raise CleanupError("final-run archive deletion is not completed")
    if archive_manifest.get("allowed_delete_realpaths") != expected_realpaths:
        raise CleanupError("final-run allowed delete realpaths mismatch")
    if deletion.get("deleted_realpaths") != expected_realpaths:
        raise CleanupError("final-run deleted realpaths mismatch")
    for relative_path in FINAL_RUN_SOURCE_PATHS:
        source = repo_root / relative_path
        if source.exists() or source.is_symlink():
            raise CleanupError(f"archived final-run source still exists: {source}")

    manifest_paths = {
        _relative_path(repo_root, file_manifest_path),
        _relative_path(repo_root, archive_manifest_path),
    }
    expected_files = declared_paths | manifest_paths
    actual_files, _ = _walk_source_member(
        repo_root, _relative_path(repo_root, final_runs_root)
    )
    actual_paths = {_relative_path(repo_root, path) for path in actual_files}
    if actual_paths != expected_files:
        raise CleanupError(
            "final-run evidence file set is not exact: "
            f"missing={sorted(expected_files - actual_paths)[:10]}, "
            f"unexpected={sorted(actual_paths - expected_files)[:10]}"
        )

    return {
        "status": "PASS",
        "root_repository_path": _relative_path(repo_root, final_runs_root),
        "file_manifest_schema_version": file_manifest["schema_version"],
        "archive_manifest_schema_version": archive_manifest["schema_version"],
        "declared_output_file_count": len(declared_paths),
        "verified_package_file_count": len(actual_paths),
        "builder_checksum_matches": True,
        "all_output_sizes_and_checksums_match": True,
        "exact_file_set_matches": True,
        "archive_size_and_checksum_match": True,
        "archive_absolute_path": str(archive_path),
        "archive_sha256": archive_verification["archive_sha256"],
        "deletion_completed": True,
        "fixed_sources_absent": True,
        "verified_repository_paths": sorted(expected_files),
    }


def _audit_compact_evidence(
    repo_root: Path,
    evidence_root: Path,
    *,
    final_run_archive_path: Path = DEFAULT_FINAL_RUN_ARCHIVE,
) -> dict[str, Any]:
    evidence_root = evidence_root.resolve(strict=True)
    evidence_manifest_path = evidence_root / "evidence_file_manifest.json"
    if evidence_manifest_path.is_symlink() or not evidence_manifest_path.is_file():
        raise CleanupError(f"compact evidence manifest is missing or unsafe: {evidence_manifest_path}")
    evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
    if evidence_manifest.get("schema_version") != "full_task_template_v2_compact_evidence_files_v1":
        raise CleanupError("compact evidence manifest schema mismatch")
    if evidence_manifest.get("status") != "PASS":
        raise CleanupError("compact evidence manifest is not PASS")
    if Path(evidence_manifest.get("repository_root", "")).resolve(strict=True) != repo_root:
        raise CleanupError("compact evidence repository root mismatch")
    if Path(evidence_manifest.get("output_absolute_path", "")).resolve(strict=True) != evidence_root:
        raise CleanupError("compact evidence output root mismatch")

    validation = evidence_manifest.get("validation", {})
    required_validation = (
        "all_16074_mapper_outputs_certified",
        "all_7974_complete_intervals_within_6ms",
        "all_copy_checksums_match",
        "all_six_certified_control_gates_pass",
        "all_six_runtime_preflights_pass",
        "offline_online_parity_pass",
    )
    if not all(validation.get(field) is True for field in required_validation):
        raise CleanupError("compact evidence validation gates are not all PASS")

    builder = evidence_manifest.get("builder", {})
    builder_path = repo_root / builder.get("repository_path", "")
    if not builder_path.is_file() or sha256_file(builder_path) != builder.get("sha256"):
        raise CleanupError("compact evidence builder checksum mismatch")

    declared: dict[str, dict[str, Any]] = {}
    for record in [
        *evidence_manifest.get("copied_files", []),
        *evidence_manifest.get("generated_files", []),
    ]:
        relative_path = record["output_repository_path"]
        if relative_path in declared:
            raise CleanupError(f"duplicate compact evidence output: {relative_path}")
        lexical_path = repo_root / relative_path
        if lexical_path.is_symlink() or not lexical_path.is_file():
            raise CleanupError(f"compact evidence output is missing or unsafe: {relative_path}")
        path = lexical_path.resolve(strict=True)
        if not _is_relative_to(path, evidence_root):
            raise CleanupError(f"compact evidence output escapes package: {relative_path}")
        if path.stat().st_size != int(record["bytes"]):
            raise CleanupError(f"compact evidence size mismatch: {relative_path}")
        if sha256_file(path) != record["sha256"]:
            raise CleanupError(f"compact evidence checksum mismatch: {relative_path}")
        declared[relative_path] = record

    for asset in evidence_manifest.get("frozen_assets", {}).values():
        lexical_path = repo_root / asset["repository_path"]
        if lexical_path.is_symlink() or not lexical_path.is_file():
            raise CleanupError(f"frozen evidence asset is missing or unsafe: {lexical_path}")
        path = lexical_path.resolve(strict=True)
        if path.stat().st_size != int(asset["bytes"]) or sha256_file(path) != asset["sha256"]:
            raise CleanupError(f"frozen evidence asset checksum mismatch: {path}")

    additional_final_run_evidence = _audit_additional_final_run_evidence(
        repo_root,
        evidence_root / "final_runs",
        expected_archive_path=final_run_archive_path,
    )

    excluded = {
        (evidence_root / "cleanup_manifest.json").resolve(strict=False),
        evidence_manifest_path.resolve(strict=True),
    }
    actual_files, _ = _walk_source_member(repo_root, _relative_path(repo_root, evidence_root))
    actual_declared = {
        _relative_path(repo_root, path)
        for path in actual_files
        if path.resolve(strict=False) not in excluded
    }
    additional_paths = set(
        additional_final_run_evidence.pop("verified_repository_paths")
    )
    expected_all = set(declared) | additional_paths
    if actual_declared != expected_all:
        raise CleanupError(
            "compact evidence contains unmanifested or missing outputs: "
            f"missing={sorted(expected_all - actual_declared)[:10]}, "
            f"unexpected={sorted(actual_declared - expected_all)[:10]}"
        )

    return {
        "status": "PASS",
        "root_absolute_path": str(evidence_root),
        "evidence_manifest_repository_path": _relative_path(
            repo_root, evidence_manifest_path
        ),
        "evidence_manifest_size_bytes": evidence_manifest_path.stat().st_size,
        "evidence_manifest_sha256": sha256_file(evidence_manifest_path),
        "evidence_manifest_schema_version": evidence_manifest["schema_version"],
        "declared_output_file_count": len(declared),
        "verified_output_file_count": len(declared),
        "all_declared_output_sizes_and_checksums_match": True,
        "all_frozen_asset_sizes_and_checksums_match": True,
        "builder_checksum_matches": True,
        "validation": validation,
        "additional_final_run_evidence": additional_final_run_evidence,
        "usage_excluding_cleanup_manifest": _tree_usage(
            evidence_root,
            excluded_files=(evidence_root / "cleanup_manifest.json",),
        ),
    }


def audit_post_cleanup(
    manifest: dict[str, Any],
    *,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
) -> dict[str, Any]:
    repo_root = Path(manifest["repo_root"]).resolve(strict=True)
    _assert_core_assets_unchanged(repo_root, manifest)
    tracked = _tracked_paths(repo_root)
    if not set(CORE_ASSETS).issubset(tracked):
        raise CleanupError("one or more core assets are no longer tracked by Git")

    deletion_entries = [
        _assert_group_deleted(repo_root, group)
        for group in manifest["archive_groups"]
    ]
    archive_summary = _audit_archive_summary(manifest)
    compact_evidence = _audit_compact_evidence(
        repo_root,
        evidence_root,
    )
    cleanup_manifest_path = Path(manifest.get("_manifest_path", DEFAULT_MANIFEST))
    post_roots = {
        relative_root: _tree_usage(repo_root / relative_root)
        for relative_root in ALLOWED_CLEANUP_ROOTS
    }
    post_roots["evaluation_summary"] = _tree_usage(
        repo_root / "evaluation_summary",
        excluded_files=(cleanup_manifest_path,),
    )
    pre_cleanup = _pre_cleanup_usage(manifest, repo_root)
    post_cleanup_scope_bytes = sum(
        post_roots[root]["regular_file_bytes"] for root in ALLOWED_CLEANUP_ROOTS
    )
    post_cleanup_scope_files = sum(
        post_roots[root]["file_count"] for root in ALLOWED_CLEANUP_ROOTS
    )
    post_cleanup = {
        "measured_at_utc": _utc_now(),
        "measurement_basis": (
            "Current regular-file, directory apparent, and allocated bytes. The "
            "cleanup manifest itself is excluded from evaluation_summary usage so "
            "repeated finalization remains stable."
        ),
        "roots": post_roots,
        "cleanup_scope_regular_file_bytes": post_cleanup_scope_bytes,
        "cleanup_scope_file_count": post_cleanup_scope_files,
        "selected_repository_regular_file_bytes_excluding_cleanup_manifest": sum(
            record["regular_file_bytes"] for record in post_roots.values()
        ),
        "selected_repository_file_count_excluding_cleanup_manifest": sum(
            record["file_count"] for record in post_roots.values()
        ),
    }
    cleanup_reduction_bytes = (
        pre_cleanup["cleanup_scope_regular_file_bytes"] - post_cleanup_scope_bytes
    )
    if cleanup_reduction_bytes != int(manifest["summary"]["cleanup_total_bytes"]):
        raise CleanupError(
            "post-cleanup byte reduction does not match the planned cleanup total: "
            f"{cleanup_reduction_bytes} != {manifest['summary']['cleanup_total_bytes']}"
        )
    return {
        "status": "PASS",
        "finalized_at_utc": _utc_now(),
        "core_assets_verified": True,
        "all_archive_groups_verified_and_completed": True,
        "all_external_archives_present_and_checksum_unchanged": True,
        "compact_evidence_verified": True,
        "cleanup_scope_regular_file_reduction_bytes": cleanup_reduction_bytes,
        "pre_cleanup": pre_cleanup,
        "post_cleanup": post_cleanup,
        "archive_summary": archive_summary,
        "compact_evidence": compact_evidence,
        "deletion_log": {
            "status": "PASS",
            "group_count": len(deletion_entries),
            "deleted_file_count": sum(
                entry["deleted_file_count"] for entry in deletion_entries
            ),
            "deleted_directory_count": sum(
                entry["deleted_directory_count"] for entry in deletion_entries
            ),
            "deleted_regular_file_bytes": sum(
                entry["deleted_regular_file_bytes"] for entry in deletion_entries
            ),
            "entries": deletion_entries,
        },
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="hash sources and write a non-destructive plan")
    plan.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    plan.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)

    verify = subparsers.add_parser("verify-archive", help="verify one existing tar.zst")
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify.add_argument("--group", required=True)
    verify.add_argument("--zstd-executable")
    verify.add_argument("--tar-executable")

    delete = subparsers.add_parser("delete", help="delete exact verified manifest targets")
    delete.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    selector = delete.add_mutually_exclusive_group(required=True)
    selector.add_argument("--group", action="append", dest="groups")
    selector.add_argument("--all-groups", action="store_true")
    delete.add_argument("--execute-delete", action="store_true")
    delete.add_argument("--confirm", default="")
    delete.add_argument("--zstd-executable")
    delete.add_argument("--tar-executable")

    audit = subparsers.add_parser(
        "audit-post",
        aliases=["finalize"],
        help="fail-closed post-cleanup archive/evidence/occupancy audit",
    )
    audit.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    audit.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "plan":
            manifest = build_cleanup_plan(REPO_ROOT, args.archive_root)
            _write_json_atomic(args.manifest, manifest)
            print(json.dumps(manifest["summary"], indent=2))
            print(f"cleanup plan: {args.manifest.resolve()}")
            return 0

        manifest = _load_manifest(args.manifest)
        if args.command == "verify-archive":
            verification = verify_group_archive(
                manifest,
                args.group,
                zstd_executable=args.zstd_executable,
                tar_executable=args.tar_executable,
            )
            group = _select_group(manifest, args.group)
            group["archive_verification"] = verification
            _write_json_atomic(args.manifest, manifest)
            print(json.dumps(verification, indent=2))
            return 0

        if args.command in ("audit-post", "finalize"):
            manifest["_manifest_path"] = str(args.manifest.resolve())
            audit = audit_post_cleanup(
                manifest,
                evidence_root=args.evidence_root,
            )
            manifest.pop("_manifest_path", None)
            manifest["pre_cleanup"] = audit.pop("pre_cleanup")
            manifest["post_cleanup"] = audit.pop("post_cleanup")
            manifest["archive_summary"] = audit.pop("archive_summary")
            manifest["compact_evidence"] = audit.pop("compact_evidence")
            manifest["deletion_log"] = audit.pop("deletion_log")
            manifest["finalization"] = audit
            _write_json_atomic(args.manifest, manifest)
            print(json.dumps(audit, indent=2))
            print(f"post-cleanup audit: {args.manifest.resolve()}")
            return 0

        if not args.execute_delete or args.confirm != DELETE_CONFIRMATION:
            raise CleanupError(
                "deletion is locked; pass --execute-delete and "
                f"--confirm {DELETE_CONFIRMATION}"
            )
        groups = (
            [group["group_id"] for group in manifest["archive_groups"]]
            if args.all_groups
            else args.groups
        )
        for group_id in groups:
            result = execute_group_delete(
                manifest,
                group_id,
                zstd_executable=args.zstd_executable,
                tar_executable=args.tar_executable,
            )
            group = _select_group(manifest, group_id)
            group["deletion"] = {
                "status": "completed",
                "completed_at_utc": _utc_now(),
                **result,
            }
            _write_json_atomic(args.manifest, manifest)
            print(f"deleted verified group {group_id}: {result}")
        return 0
    except (CleanupError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
