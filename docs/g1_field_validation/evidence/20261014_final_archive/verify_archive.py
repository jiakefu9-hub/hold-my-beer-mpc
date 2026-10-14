"""Read-only archive hash verification; --deep also decompresses all raw logs.

Standard library only. No robot SDK, network, writes or local evaluation needed.
"""
import argparse
import hashlib
import json
import lzma
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
SECRET=re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|"(?:password|api_key|access_token|refresh_token)"\s*:\s*"[^"\s]{8,}"')


def hash_stream(stream, scan=False):
    h=hashlib.sha256();n=0;tail=b''
    for data in iter(lambda:stream.read(1024*1024),b''):
        h.update(data);n+=len(data)
        if scan:
            assert not SECRET.search(tail+data), 'Potential credential pattern; stop publication and inspect locally'
            tail=data[-256:]
    return h.hexdigest(),n


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deep',action='store_true');args=p.parse_args()
    manifest=json.loads((HERE/'manifest.json').read_text());checked=raw=total=0
    for row in manifest['files']:
        f=(HERE/row['path']).resolve();assert f.is_relative_to(HERE)
        with f.open('rb') as stream:
            sha,size=hash_stream(stream,scan=f.suffix in ('.md','.json','.csv','.conf','.yaml','.txt'))
        assert sha==row['sha256'] and size==row['bytes'],row['path']
        if row['kind']=='byte_copy':assert sha==row['source_sha256']
        if f.suffix=='.json':json.loads(f.read_text())
        if args.deep and row['kind']=='lossless_xz':
            with lzma.open(f,'rb') as stream:sha,size=hash_stream(stream,scan=True)
            assert sha==row['source_sha256'] and size==row['source_bytes'],row['path']
            raw+=1
        checked+=1;total+=row['bytes']
    print(json.dumps(dict(status='PASS',files_verified=checked,full_raw_decompressed=raw,
                         archived_bytes=total,credential_pattern_scan='no matching patterns; not a general privacy audit'),indent=2))


if __name__=='__main__':main()
