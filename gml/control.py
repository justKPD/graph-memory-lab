"""Single-job local orchestration; fixed argv, timeout, fresh evidence paths."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from .assistant import load_assistant

class LocalControl:
    def __init__(self, root=None):
        self.root=Path(root or Path(__file__).resolve().parent.parent)
        self.lock=threading.Lock();self.job=None;self.process=None
    def command(self, request):
        if not isinstance(request,dict) or set(request)!={'command'}:
            raise ValueError('Only a typed command is accepted')
        action=request['command']
        if action not in ('run replay','job status','cancel replay'):
            raise ValueError('Only run replay, job status, cancel replay are supported')
        with self.lock:
            if action=='job status':return dict(self.job or {'status':'IDLE'})
            if action=='cancel replay':
                if self.job and self.job['status']=='RUNNING':
                    self.job['status']='CANCELLING'
                    if self.process:self.process.terminate()
                return dict(self.job or {'status':'IDLE'})
            if self.job and self.job['status'] in ('RUNNING','CANCELLING'):
                raise ValueError('A replay is already active; inspect or cancel it first')
            load_assistant(self.root/'results/registry.json')
            ident=uuid.uuid4().hex
            self.job={'id':ident,'status':'RUNNING','started':time.time(),
                      'output':f'results/chat_jobs/{ident}/runtime.json',
                      'timeout_seconds':180,'scope':'Fixed uniform_train replay; no arbitrary commands'}
            threading.Thread(target=self._run,args=(ident,),daemon=True).start()
            return dict(self.job)
    def _run(self,ident):
        folder=self.root/'results/chat_jobs'/ident;folder.mkdir(parents=True)
        output=folder/'runtime.json'
        env=os.environ.copy()
        env['PYTHONPATH']=str(Path(sys.prefix)/'Lib/site-packages')+os.pathsep+str(self.root)
        with (folder/'worker.log').open('w') as log:
            try:
                process=subprocess.Popen([getattr(sys,'_base_executable',sys.executable),'-m','gml.runtime','--output',str(output)],
                    cwd=self.root,env=env,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                with self.lock:
                    self.process=process
                    if self.job['status']=='CANCELLING':process.terminate()
                try:code=process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait();self._finish('TIMEOUT');return
                with self.lock:cancelled=self.job['status']=='CANCELLING'
                if cancelled:self._finish('CANCELLED');return
                if code!=0:self._finish('FAILED');return
                report=json.loads(output.read_text())
                summary=[{k:t[k] for k in ('strategy','trained_steps','pauses','budget_violations')} for t in report['trials']]
                self._finish('COMPLETED',summary)
            except Exception as e:
                log.write(str(e));self._finish('FAILED')
    def _finish(self,status,summary=None):
        with self.lock:
            self.job.update(status=status,finished=time.time())
            if summary is not None:self.job['trials']=summary
            self.process=None
