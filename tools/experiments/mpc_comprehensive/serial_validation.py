"""Repeat finalists through the unmodified run.sh entry on CPU 7.

Run only after the parallel research batches finish. Timings describe this
host's actual scheduler/governor; they are not robot deadline certification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from metrics import summarize
from study import REPO, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cases', type=Path)
    args = parser.parse_args()
    for case in json.loads(args.cases.read_text()):
        name = case['name']
        target = ROOT / 'results' / (name + '.json')
        if target.exists():
            continue
        config = REPO / case['config']
        env = dict(os.environ, MPC_CONFIG_FILE=str(config), MPC_CONTROL_CPU='7',
                   MPC_CONTROL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        start = time.monotonic()
        with (ROOT / 'logs' / (name + '.log')).open('w') as log:
            process = subprocess.run(['./run.sh', '--full-task-smoke', '--headless',
                '--no-video', '--evaluation-group', ROOT.name, '--run-label', name],
                cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        paths = sorted(ROOT.glob('20*_' + name))
        result = summarize(paths[-1]) if paths else dict(complete=False)
        result.update(case, returncode=process.returncode, cpu=7,
            wall_s=time.monotonic()-start, formal_entry=True,
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
        target.write_text(json.dumps(result, indent=2)+'\n')
        print(name, process.returncode, result.get('gates'), flush=True)
        if process.returncode:
            raise RuntimeError(f'{name} failed; inspect its preserved log')


if __name__ == '__main__':
    main()
