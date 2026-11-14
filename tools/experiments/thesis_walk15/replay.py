"""Play the recorded 12 s walk + 3 s stop using the existing MuJoCo viewer."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from protocol import EVIDENCE
import json
import argparse
import re

selection=argparse.ArgumentParser(add_help=False)
selection.add_argument('--variant',default=None)
chosen,rest=selection.parse_known_args(sys.argv[1:])
sys.argv=[sys.argv[0]]+rest
variant=chosen.variant
if variant is None and (EVIDENCE/'selected_variant.json').exists():
    variant=json.loads((EVIDENCE/'selected_variant.json').read_text())['variant']
if variant and variant!='original':
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',variant):
        raise SystemExit('invalid replay variant')
    EVIDENCE=EVIDENCE/'variants'/variant
print('Replay dataset:',variant or 'original',flush=True)
sys.path.insert(0,str(HERE.parent/'simulation_showcase'))
import replay as viewer

viewer.EVIDENCE=EVIDENCE
manifest=json.loads((EVIDENCE/'manifest.json').read_text())
if len(sys.argv)>1 and sys.argv[1] in manifest['cases']:
    case=manifest['cases'][sys.argv[1]]
    if case.get('complete') is False:
        print('INCOMPLETE FAILURE REPLAY:',case['failure'],flush=True)
        print('Actual recorded duration:',case['duration_s'],'s',flush=True)
    elif 'metrics' in case:
        gates=json.loads((EVIDENCE/case['metrics']).read_text())['gates']
        print('Recording quality:',gates,flush=True)
# Call the shared viewer directly. Its clock and checks use case.duration_s;
# source-text replacements can change labels while leaving an old loop limit.
if __name__ == '__main__':
    viewer.main()
