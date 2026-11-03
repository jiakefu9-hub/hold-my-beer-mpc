"""Validate shipped simulation and optional field evidence; no robot access."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "docs/simulation/experiments/20261103_showcase"


def verify_hash(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Hash mismatch: {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-archive", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((SHOWCASE / "manifest.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "showcase_replay", ROOT / "tools/experiments/simulation_showcase/replay.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    checked = {}
    for name, case in manifest["cases"].items():
        verify_hash(ROOT / case["config"], case["config_sha256"])
        _, frames, model = replay.load_case(name)
        metrics = json.loads((SHOWCASE / case["metrics"]).read_text())
        if not metrics["complete"]:
            raise ValueError(f"Incomplete replay: {name}")
        checked[name] = {"frames": len(frames["time"]), "model_nq": model.nq}
    configs = [yaml.safe_load((ROOT / manifest["cases"][name]["config"]).read_text())
               for name in ("D02", "D03")]
    difference = sorted(key for key in configs[0].keys() | configs[1].keys()
                        if configs[0].get(key) != configs[1].get(key))
    if difference != manifest["mpc_pair_configuration_difference"]:
        raise ValueError(f"Unexpected MPC pair configuration differences: {difference}")
    result = {"simulation_cases": checked, "mpc_pair_difference": difference}
    predictor = ROOT / "assets/g1_hardware_mpc_predictor"
    bank_manifest = json.loads((predictor / "manifest.json").read_text())
    verify_hash(predictor / "bank.npz", bank_manifest["bank_sha256"])
    result["predictor_bank_sha256"] = bank_manifest["bank_sha256"]
    if args.hardware_archive:
        directory = ROOT / "docs/g1_field_validation/evidence/20261014_final_archive"
        archive = json.loads((directory / "manifest.json").read_text())
        for item in archive["files"]:
            path = directory / item["path"]
            if path.stat().st_size != item["bytes"]:
                raise ValueError(f"Size mismatch: {path.relative_to(ROOT)}")
            verify_hash(path, item["sha256"])
        result["hardware_archive"] = {
            "verified_files": len(archive["files"]),
            "full_raw_runs": archive["full_raw_runs"],
            "all_run_records": archive["all_run_records"],
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
