import urllib.request,urllib.error,json,time
from pathlib import Path
base='http://127.0.0.1:8765'
def post(p,origin=base):
 r=urllib.request.Request(base+'/api/control',data=json.dumps(p).encode(),headers={'Content-Type':'application/json','Origin':origin})
 try:
  with urllib.request.urlopen(r,timeout=8) as s:return s.status,json.load(s)
 except urllib.error.HTTPError as e:return e.code,json.load(e)
checks=[]
assert post({'command':'job status'})[1]['status']=='IDLE';checks.append('idle')
assert post({'command':'run arbitrary shell'})[0]==400;checks.append('raw-command-refused')
assert post({'command':'job status','extra':1})[0]==400;checks.append('unknown-fields-refused')
assert post({'command':'run replay'},'https://unrelated.example')[0]==403;checks.append('cross-origin-refused')
code,job=post({'command':'run replay'});assert code==200 and job['status']=='RUNNING';checks.append('real-job-launched')
assert post({'command':'run replay'})[0]==400;checks.append('concurrent-job-refused')
print('Started real bounded job',job['id'],flush=True)
for i in range(42):
 time.sleep(5);status=post({'command':'job status'})[1]
 if status['status'] not in ('RUNNING','CANCELLING'):break
assert status['status']=='COMPLETED',status
assert all(t['budget_violations']==0 for t in status['trials']);checks.append('actual-replay-completed')
assert Path(job['output']).is_file();checks.append('fresh-result-preserved')
assert post({'command':'run replay'})[0]==200
assert post({'command':'cancel replay'})[1]['status']=='CANCELLING'
for i in range(20):
 time.sleep(.5);cancel=post({'command':'job status'})[1]
 if cancel['status']=='CANCELLED':break
assert cancel['status']=='CANCELLED',cancel;checks.append('actual-cancel-completed')
Path('results/control_api_tests.json').write_text(json.dumps({'status':'PASS','checks':checks,'completed_job':job['output'],'completed_trials':status['trials'],'cancelled_job':cancel['id']},indent=2));print('PASS',checks)
