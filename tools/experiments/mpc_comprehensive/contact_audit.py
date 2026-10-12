"""Audit arm contacts and physical joint limits from compact rollout states."""
from collections import Counter
from pathlib import Path
import json
import sys
import mujoco
import numpy as np
import yaml
from study import ROOT, REPO


def audit(result):
    config = yaml.safe_load(Path(result['config']).read_text())
    model = mujoco.MjModel.from_xml_path(str(REPO / config['xml_path']))
    data = mujoco.MjData(model)
    trajectory = np.load(Path(result['path']) / 'compact_metrics.npz')
    time = trajectory['time']
    qpos = trajectory['qpos']
    qvel = trajectory['qvel']
    trajectory.close()
    arm = {model.body(side + '_shoulder_pitch_link').id for side in ('left','right')}
    for body in range(1, model.nbody):
        if int(model.body_parentid[body]) in arm:
            arm.add(body)
    indices = np.flatnonzero((time >= 0.) & (time < 8. - 1e-10))
    contact_frames = 0
    pairs = Counter()
    joint_violation = 0.
    for i in indices:
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_fwdPosition(model, data)
        frame = set()
        for contact in data.contact:
            a, b = [int(model.geom_bodyid[int(g)]) for g in (contact.geom1, contact.geom2)]
            if contact.dist <= 0. and (a in arm or b in arm):
                frame.add(' / '.join(sorted((model.body(a).name, model.body(b).name))))
        contact_frames += bool(frame)
        pairs.update(frame)
        for joint in range(model.njnt):
            if model.jnt_bodyid[joint] in arm and model.jnt_limited[joint]:
                q = data.qpos[model.jnt_qposadr[joint]]
                joint_violation = max(joint_violation, model.jnt_range[joint,0]-q,
                                      q-model.jnt_range[joint,1])
    return dict(sample_count=len(indices), sample_period_s=.002,
                arm_contact_frames=contact_frames, pairs=dict(pairs),
                max_physical_joint_limit_violation_rad=float(joint_violation))


if __name__ == '__main__':
    for name in sys.argv[1:]:
        path = ROOT/'results'/(name+'.json')
        result = json.loads(path.read_text())
        result['contact_audit'] = audit(result)
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(name,result['contact_audit'])
