"""Restore the original 11-build/4-held-out design for the 13 s task.

The shared collector and template mathematics are reused unchanged. Only the
task protocol and output paths differ from the original 8 s acquisition.
"""
from dataclasses import asdict
import hashlib
import json

import numpy as np

from protocol import PROTOCOL, ROOT
from disturbance_template.full_task_fixed_pd_collector import (
    collection_specs, collect_fixed_pd_episode)
from disturbance_template.full_task_protocol import FullTaskContinuousHeadingFrame
from disturbance_template.full_task_recording import validate_full_task_raw
from disturbance_template.full_task_template_asset import (
    TEMPLATE_SCHEMA_VERSION_V2, validate_full_task_template)
from disturbance_template.full_task_template_builder import (
    _json_value, build_full_task_template, causal_h_metrics,
    evaluate_heldout_template, portable_asset)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2,
                               ensure_ascii=False) + '\n')


def assess_episode(raw):
    """Original fixed-PD gates, with stop/distance derived from the new task."""
    validation = validate_full_task_raw(raw, PROTOCOL, require_complete=True)
    h = causal_h_metrics(raw)
    rotation = raw['torso_rotation_world']
    roll = np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2])
    pitch = np.arcsin(np.clip(-rotation[:, 2, 0], -1., 1.))
    fallen = bool(np.min(raw['torso_position_world'][:, 2]) < .45
                  or max(np.max(np.abs(roll)), np.max(np.abs(pitch)))
                  > np.deg2rad(60.))
    end = int(round(PROTOCOL.headline_end / PROTOCOL.physics_dt))
    xy = raw['torso_position_world'][:end + 1, :2]
    displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    expected = float(np.linalg.norm(raw['nominal_command'][:2])) * PROTOCOL.stop_time
    distance_gate = [expected - .4, expected + .4]
    stop = int(round(PROTOCOL.stop_time / PROTOCOL.physics_dt))
    direct = bool(np.allclose(raw['planned_command'][stop - 1, :2], raw['nominal_command'][:2])
                  and np.allclose(raw['planned_command'][stop:stop + 2, :2], 0.))
    heading = bool(np.any(np.isfinite(raw['heading_reference_world'])))
    saturation = int(np.count_nonzero(raw['right_arm_pd_saturated']))
    h_ok = (h['min_circular_concentration'] >= .95
            and h['max_adjacent_h_yaw_jump_rad'] <= .15
            and h['max_cycle_boundary_h_yaw_jump_rad'] <= .15)
    passed = bool(validation['strict_pre_step'] and validation['tail_complete']
                  and direct and heading and not fallen and not saturation and h_ok
                  and distance_gate[0] <= displacement <= distance_gate[1])
    return dict(status='PASS' if passed else 'FAIL', fallen=fallen,
                xy_displacement_m=displacement, distance_gate_m=distance_gate,
                direct_step_effective=direct, heading_enabled=heading,
                right_arm_pd_saturation_sample_count=saturation,
                torso_height_min_m=float(np.min(raw['torso_position_world'][:, 2])),
                **validation, **h)


def build_original_design(config, config_path, out):
    out.mkdir(parents=True, exist_ok=True)
    build_specs, heldout_specs = collection_specs()
    reference = ROOT / 'docs/simulation/experiments/20261117_walk10_stop3/template/acquisition_design.json'
    reference_manifest = json.loads(reference.read_text())
    design = {role: [_json_value(asdict(spec)) for spec in specs]
              for role, specs in (('build', build_specs), ('heldout', heldout_specs))}
    # Compare actual offsets and seeds, not just the number of episodes.
    for role in design:
        if design[role] != reference_manifest['perturbation_design'][role]:
            raise RuntimeError('original perturbation design has changed: ' + role)
    config_hash = sha(config_path)
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in (
        ROOT / 'disturbance_template/full_task_fixed_pd_collector.py',
        ROOT / 'disturbance_template/full_task_template_builder.py',
        ROOT / 'tools/experiments/thesis_walk13/protocol.py',
        ROOT / 'tools/experiments/thesis_walk13/full_template.py')}
    write_json(out / 'acquisition_design.json', dict(
        build_episode_count=11, heldout_episode_count=4,
        perturbation_design=design, reference=portable_asset(reference, ROOT),
        matches_original_perturbations=True, config_sha256=config_hash,
        source_sha256=source_hashes, protocol=asdict(PROTOCOL)))
    raws, paths, reports = [], [], []
    specs = build_specs + heldout_specs
    for index, spec in enumerate(specs, 1):
        path = out / 'episodes' / spec.episode_id / 'raw.npz'
        meta_path = path.with_name('metadata.json')
        expected_meta = dict(episode=_json_value(asdict(spec)),
                             config_sha256=config_hash, source_sha256=source_hashes)
        if path.exists():
            metadata = json.loads(meta_path.read_text())
            if any(metadata.get(k) != value for k, value in expected_meta.items()):
                raise RuntimeError('cannot resume mismatched episode: ' + spec.episode_id)
            if metadata['raw_sha256'] != sha(path):
                raise RuntimeError('raw checksum mismatch: ' + spec.episode_id)
            with np.load(path, allow_pickle=False) as archive:
                raw = {k: archive[k] for k in archive.files}
        else:
            print(f'COLLECT {index}/15 {spec.episode_id}', flush=True)
            raw, details = collect_fixed_pd_episode(
                config, config_path=config_path, spec=spec, protocol=PROTOCOL,
                heading_frame_version=FullTaskContinuousHeadingFrame.DEFINITION_VERSION)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **raw)
            write_json(meta_path, dict(**expected_meta, raw_sha256=sha(path),
                fixed_pd=details['fixed_pd'], heading_control=details['heading_control']))
        report = assess_episode(raw)
        write_json(path.with_name('validation.json'), report)
        print(f'COLLECTED {index}/15 {spec.episode_id}: {report["status"]}, '
              f'distance={report["xy_displacement_m"]:.3f} m', flush=True)
        if report['status'] != 'PASS':
            raise RuntimeError('fixed-PD episode failed; raw evidence retained: ' + spec.episode_id)
        raws.append(raw); paths.append(path); reports.append(report)
    print('BUILD TEMPLATE: 11 build episodes; 4 excluded held-out episodes', flush=True)
    template = build_full_task_template(raws[:11], [s.episode_id for s in build_specs],
        PROTOCOL, template_schema_version=TEMPLATE_SCHEMA_VERSION_V2)
    validation = validate_full_task_template(template, PROTOCOL)
    heldout, _ = evaluate_heldout_template(template, raws[11:],
        [s.episode_id for s in heldout_specs], PROTOCOL)
    asset = out / 'full_task_template.npz'
    np.savez_compressed(asset, **template)
    write_json(out / 'heldout_metrics.json', heldout)
    manifest_path = out / 'full_task_template_manifest.json'
    write_json(manifest_path, dict(
        template_schema_version=TEMPLATE_SCHEMA_VERSION_V2,
        template_validation=validation, template=portable_asset(asset, ROOT),
        collection=dict(protocol=asdict(PROTOCOL), build_episode_count=11,
            heldout_episode_count=4, perturbation_design=design,
            heading_frame_version=FullTaskContinuousHeadingFrame.DEFINITION_VERSION,
            build_episode_summaries=reports[:11], heldout_episode_summaries=reports[11:],
            config=portable_asset(config_path, ROOT)),
        build_raw_episodes=[portable_asset(p, ROOT) for p in paths[:11]],
        heldout_raw_episodes=[portable_asset(p, ROOT) for p in paths[11:]],
        reference_design=portable_asset(reference, ROOT), source_sha256=source_hashes,
        scope='13 s absolute-task-time template; original 11+4 perturbations; '
              'no cycle folding, repetition or splicing; held-outs excluded from construction'))
    print('TEMPLATE COMPLETE: 11+4, SHA256=' + sha(asset), flush=True)
    return asset, manifest_path
