"""Verify raw/profile consistency and demonstrate detection of altered derived data."""
import copy
import json
from pathlib import Path
import tempfile
from gml.assistant import load_assistant
from gml.evidence import Ledger

agent=load_assistant()
checks=['manifest-hash','raw-file-hashes','profile-identity','timing-from-raw','memory-from-raw']
import hashlib
manifest=json.loads(Path('results/benchmark/manifest.json').read_text())
for name,digest in manifest['source_hashes'].items():
    snapshot=Path('results/benchmark/source_snapshot')/name
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==digest
checks.append('measured-source-snapshot')
with tempfile.TemporaryDirectory(dir=Path('results').resolve()) as folder:
    altered=copy.deepcopy(agent.registry)
    altered['profiles'][0]['step_ms']=0.001
    path=Path(folder)/'altered.json';path.write_text(json.dumps(altered))
    try:load_assistant(path)
    except ValueError as e:
        assert 'timing' in str(e);checks.append('forged-derived-speed-rejected')
    else:raise AssertionError('Forged timing accepted')
rows=Ledger.verify('results/runtime.ledger.jsonl')
runtime=json.loads(Path('results/runtime.json').read_text())
assert rows[-1]['hash']==runtime['ledger_root'];checks.append('runtime-ledger-external-root')
result={'status':'PASS','checks':checks,'limitations':'Consistency checks are not signed authenticity.'}
Path('results/integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
