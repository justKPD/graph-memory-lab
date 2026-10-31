"""Loopback-only evidence server with a separate bounded local replay controller."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from gml.assistant import load_assistant
from gml.control import LocalControl
control=LocalControl()


class Handler(SimpleHTTPRequestHandler):
    def reply(self,status,value):
        payload=json.dumps(value,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(payload)
    def do_GET(self):
        if self.path=='/api/status':
            return self.reply(200,{'status':'ok','mode':'local bounded orchestration','gpu_execution_endpoint':True})
        if self.path=='/api/evidence':return self.reply(200,load_assistant().registry)
        return super().do_GET()
    def do_POST(self):
        if self.path not in ('/api/assistant','/api/control'):return self.reply(404,{'error':'Unknown endpoint'})
        origin=self.headers.get('Origin')
        if self.path=='/api/control' and origin!=f'http://127.0.0.1:{self.server.server_port}':
            return self.reply(403,{'error':'Local orchestration requires the exact loopback Origin'})
        if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):
            return self.reply(403,{'error':'Origin rejected'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=4096:raise ValueError('Request must be 1–4096 bytes')
            payload=json.loads(self.rfile.read(size))
            if self.path=='/api/control':return self.reply(200,control.command(payload))
            agent=load_assistant()
            request=agent.parse(payload['text']) if set(payload)=={'text'} else payload
            return self.reply(200,agent.execute(request))
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as e:
            return self.reply(400,{'error':str(e)})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    root=Path(__file__).resolve().parent
    import os
    os.chdir(root)
    handler=partial(Handler,directory=str(root/'web'))
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler)
    print(f'Graph Memory Lab: http://127.0.0.1:{args.port} (loopback only)',flush=True)
    server.serve_forever()
