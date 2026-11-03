"""Validate and export selected numerical records and small portable replays."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'mpc_comprehensive'))
from metrics import summarize

STUDY = ROOT / 'evaluation/simulation_showcase_20261103'
DEST = ROOT / 'docs/simulation/experiments/20261103_showcase'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cases = json.loads((HERE / 'cases.json').read_text())
    configs = []
    for case in cases:
        config = yaml.safe_load((ROOT / case['base_config']).read_text())
        config.update(case.get('overrides', {}))
        configs.append(config)
    difference = [k for k in configs[1].keys() | configs[2].keys()
                  if configs[1].get(k) != configs[2].get(k)]
    assert difference == ['research_hold_future_disturbance'], difference
    common = ('xml_path', 'simulation_dt', 'control_decimation', 'arm_control_decimation',
              'arm_waist_kps', 'arm_waist_kds', 'arm_waist_target',
              'kps', 'kds', 'default_angles', 'policy_path', 'cmd_init')
    assert all(all(c[k] == configs[0][k] for c in configs) for k in common)
    assert all(c['mpc_preview_rotation_correction_gain'] == 0. and
               c['mpc_preview_interval_lead_s'] == 0. for c in configs)
    (DEST / 'replays').mkdir(parents=True, exist_ok=True)
    manifest = dict(
        evaluation_window='0 <= t < 8 s', simulation_only=True,
        physics_dt_s=.002, control_dt_s=.006, horizon=9,
        payload=('D01-D03: hardware-matched OD 65 mm / ID 25 mm / H 75 mm / '
                 '250 g ring; optional historical PID replays retain the 200 mm model'),
        mpc_pair_configuration_difference=difference,
        shared_fields=list(common), cases={}, model_sha256={},
    )
    for path in (ROOT / 'resources/g1_description').glob('*.xml'):
        if path.name in ('scene.xml', 'g1_23dof_rev_1_0.xml',
                         'scene_payload75.xml', 'g1_23dof_rev_1_0_payload75.xml'):
            manifest['model_sha256'][str(path.relative_to(ROOT))] = sha(path)
    rows = []
    alternatives = json.loads((HERE / 'alternatives.json').read_text())
    variants = json.loads((HERE / 'roll50_variants.json').read_text())
    extras = alternatives + variants
    extra_configs = [yaml.safe_load((ROOT / c['base_config']).read_text())
                     for c in extras]
    for index, (case, config) in enumerate(zip(
            cases + extras, configs + extra_configs), 1):
        paths = sorted(STUDY.glob('20*_'+case['name'].lower()))
        if not paths:
            raise FileNotFoundError(f"Run {case['name']} first")
        path = paths[-1]
        result_path = STUDY / 'results' / (case['name']+'.json')
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if (path / 'metrics.npz').exists():
            result.update(summarize(path, xml_path=config['xml_path']))
        assert result.get('complete') and result['gates']['status'] == 'PASS', result
        assert not result['gates']['fallen'] and result['gates']['nan_inf_count'] == 0
        compact = path / 'compact_metrics.npz'
        with np.load(compact, allow_pickle=False) as data:
            mask = data['time'] <= 8.+1e-9
            arrays = {k: data[k][mask] for k in ('time', 'qpos', 'qvel')}
        ident = case.get('replay_id', f'D{index:02}')
        out = DEST / 'replays' / (ident+'.npz')
        np.savez_compressed(out, **arrays)
        result_path.write_text(json.dumps(result, indent=2)+'\n')
        exported = dict(result, showcase_config=case['base_config'], replay_id=ident)
        (DEST / (ident+'_metrics.json')).write_text(json.dumps(exported, indent=2)+'\n')
        controller_label = case.get('controller_label', 'PID')
        manifest['cases'][ident] = dict(
            title=case['title'], controller_label=controller_label,
            config=case['base_config'], config_sha256=sha(ROOT / case['base_config']),
            config_overrides=case.get('overrides', {}),
            source_run=str(path.relative_to(ROOT)),
            xml_path=config['xml_path'], replay=str(out.relative_to(DEST)),
            replay_sha256=sha(out), metrics=ident+'_metrics.json',
        )
        for side, metrics in result['windows']['full'].items():
            rows.append(dict(case=ident, side=side, **metrics))
        print(ident, result['gates'], result['windows']['full']['right'])
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    with (DEST / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
