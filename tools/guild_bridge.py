"""Project NESSA evidence onto the recovered Eldrae guild event protocol.

Avatars represent roles in one harness, not independently executing agents.
No idle tasks or synthetic success events are generated.
"""
import json, queue, sys, threading
import bridge
AGENTS=['FORGE','VIAL','WARD','LORE','SPARK','TIDE','ORACLE','SAGE']

def project_event(event, run_id):
    kind=event.get('event');name=event.get('name','')
    if kind=='start':return {'type':'quest_start','name':'FORGE','quest':event.get('task',run_id)}
    if kind=='check':return {'type':'log','name':'VIAL','msg':f"{event.get('phase')}: {name} = {event.get('status')}",'category':'check'}
    if kind in {'protocol_error','controller'}:return {'type':'log','name':'WARD','msg':json.dumps(event,ensure_ascii=True)[:1200],'category':'controller'}
    if kind=='plan':return {'type':'log','name':'ORACLE','msg':json.dumps(event.get('plan'),ensure_ascii=True)[:1200],'category':'plan'}
    if kind=='checkpoint':return {'type':'log','name':'LORE','msg':'Checkpoint: '+str(event.get('path')),'category':'evidence'}
    if kind=='end':
        if event.get('status')=='verified':return {'type':'quest_done','name':'FORGE','quest':run_id,'xp':10,'level':1}
        return {'type':'quest_fail','name':'FORGE','quest':run_id,'error':str(event.get('status'))}
    return None

def main():
    commands=queue.Queue();offsets={};following=True
    def emit(value):print(json.dumps(value,ensure_ascii=True),flush=True)
    def reader():
        for line in sys.stdin:
            try:commands.put(json.loads(line))
            except ValueError:commands.put({'cmd':'invalid'})
        commands.put({'cmd':'quit'})
    threading.Thread(target=reader,daemon=True).start()
    emit({'type':'ready','agents':AGENTS})
    for name in AGENTS:emit({'type':'stats','name':name,'level':1,'xp':0})
    emit({'type':'system','msg':'Guild roles now display NESSA evidence. Start coding tasks in NESSA Workbench.'})
    while True:
        try:cmd=commands.get(timeout=1)
        except queue.Empty:cmd={}
        action=cmd.get('cmd')
        if action=='quit':return
        if action=='auto':following=bool(cmd.get('on'))
        if action in {'report','patrol'}:
            for state in bridge.dispatch('tasks',{}):emit({'type':'log','name':'ORACLE','msg':state['status']+': '+state['task'][:400],'category':'run'})
            emit({'type':'system','msg':'Report refreshed. New work requires a project and task in NESSA Workbench.'})
        if action=='chat':
            emit({'type':'chat_stream','from':'ORACLE','event':'start'})
            try:emit({'type':'chat_chunk','from':'ORACLE','delta':bridge.chat({'message':str(cmd.get('text',''))})['text']})
            except Exception as e:emit({'type':'chat_chunk','from':'ORACLE','delta':'Model error: '+str(e)})
            emit({'type':'chat_stream','from':'ORACLE','event':'end'})
        if following:
            for file in bridge.RUNS.glob('*/work/evidence/events.jsonl'):
                run_id=file.parents[2].name
                with file.open('rb') as stream:
                    stream.seek(offsets.get(str(file),0))
                    for _ in range(200):
                        pos=stream.tell();line=stream.readline()
                        if not line or not line.endswith(b'\n'):stream.seek(pos);break
                        try:ev=project_event(json.loads(line),run_id)
                        except ValueError:continue
                        if ev:emit(ev)
                    offsets[str(file)]=stream.tell()

if __name__=='__main__':main()
