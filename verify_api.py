import json
from pathlib import Path
import urllib.request
import urllib.error

BASE='http://127.0.0.1:8765'
checks=[]

def request(path,payload=None,origin=None):
    headers={'Content-Type':'application/json'}
    if origin:headers['Origin']=origin
    req=urllib.request.Request(BASE+path,data=None if payload is None else json.dumps(payload).encode(),headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=20) as response:return response.status,json.load(response)
    except urllib.error.HTTPError as error:return error.code,json.load(error)

status,value=request('/api/status');assert status==200 and value['gpu_execution_endpoint'] is True
checks.append('read-only-status')
status,value=request('/api/assistant',{'text':'compare uniform_train budget 100'})
assert status==200 and value['chosen'];checks.append('measured-budget-selection')
status,value=request('/api/assistant',{'action':'compare','workload':'uniform_train','budget_mib':0})
assert status==200 and value['chosen'] is None;checks.append('no-feasible-budget')
status,value=request('/api/assistant',{'text':'propose ring_holdout budget 200'})
assert status==200 and value['plan']['status']=='PROPOSED_NOT_EXECUTED';checks.append('proposal-not-execution')
for name,payload in [
    ('stale-graph',{'action':'compare','workload':'uniform_train','expected_graph_hash':'wrong'}),
    ('precision-mismatch',{'action':'compare','workload':'uniform_train','precision':'float16'}),
    ('raw-command',{'action':'run','command':'arbitrary shell'}),
    ('validation-edit',{'action':'compare','workload':'uniform_train','numerical_valid':True})]:
    status,value=request('/api/assistant',payload);assert status==400 and 'error' in value;checks.append(name)
status,value=request('/api/assistant',{'text':'verify'},origin='https://unrelated.example')
assert status==403;checks.append('cross-origin-refusal')
status,value=request('/api/evidence');assert status==200 and len(value['profiles'])==60
checks.append('full-verified-registry')
report={'status':'PASS','checks':checks,'count':len(checks),'endpoint':BASE,
        'scope':'Loopback HTTP API tests; no public hosting or GPU job triggered'}
Path('results/api_tests.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
