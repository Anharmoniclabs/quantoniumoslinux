"""Loopback-only recovery workbench. Same backend contract as Tauri IPC."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import hmac, json, os, secrets, subprocess, sys, threading, urllib.parse, urllib.request, webbrowser
import bridge

ROOT=bridge.BASE/'runtime/desktop-shell/public'
PORT=18420
STATE=bridge.DATA/'workbench-server.json'

def open_window(token):
    url=f'http://127.0.0.1:{PORT}/?token={token}'
    for path in [Path(os.environ.get('PROGRAMFILES(X86)','C:/Program Files (x86)'))/'Microsoft/Edge/Application/msedge.exe',
                 Path(os.environ.get('PROGRAMFILES','C:/Program Files'))/'Google/Chrome/Application/chrome.exe']:
        if path.is_file():
            subprocess.Popen([str(path),'--app='+url,'--window-size=1280,900'],
                             creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));return
    webbrowser.open(url)

def existing():
    if not STATE.is_file():return None
    try:
        state=json.loads(STATE.read_text())
        req=urllib.request.Request(f'http://127.0.0.1:{PORT}/api',
              data=b'{"action":"health"}',headers={'Content-Type':'application/json','X-Quantonium-Token':state['token']})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=8) as r:
            if json.load(r).get('ok'):return state['token']
    except Exception:pass
    return None

def main():
    old=existing()
    if old:
        if '--open' in sys.argv:open_window(old)
        return
    token=secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass  # Do not put access tokens in logs.
        def send(self,status,body,kind='application/json'):
            self.send_response(status);self.send_header('Content-Type',kind+'; charset=utf-8')
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','no-referrer')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def host_ok(self):return self.headers.get('Host')==f'127.0.0.1:{PORT}'
        def do_GET(self):
            if not self.host_ok():self.send(403,b'{}');return
            name=urllib.parse.urlparse(self.path).path
            allowed={'/':('workbench.html','text/html'),'/workbench.html':('workbench.html','text/html'),
                     '/workbench.css':('workbench.css','text/css'),'/workbench.js':('workbench.js','application/javascript')}
            if name not in allowed:self.send(404,b'{}');return
            file,kind=allowed[name];self.send(200,(ROOT/file).read_bytes(),kind)
        def do_POST(self):
            origin=self.headers.get('Origin')
            if not self.host_ok() or self.path!='/api' or origin not in (None,f'http://127.0.0.1:{PORT}') or not hmac.compare_digest(self.headers.get('X-Quantonium-Token',''),token):
                self.send(403,b'{"ok":false,"error":"Unauthorized local request"}');return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if n<1 or n>100000:raise ValueError('Invalid request size')
                req=json.loads(self.rfile.read(n));result=bridge.dispatch(req['action'],req.get('args',{}))
                self.send(200,json.dumps({'ok':True,'data':result}).encode())
            except Exception as e:self.send(400,json.dumps({'ok':False,'error':str(e)}).encode())
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    bridge.atomic(STATE,{'pid':os.getpid(),'token':token,'port':PORT})
    if '--open' in sys.argv:threading.Timer(.3,lambda:open_window(token)).start()
    server.serve_forever()

if __name__=='__main__':main()
