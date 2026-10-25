"""Index project-related user/assistant conversations from the July export."""
from pathlib import Path
import datetime, hashlib, json, re, sqlite3
from bridge import BASE,DATA,atomic
MATCH=re.compile(r'quantonium|eldrae|\bnessa\b|resonan[ct]\s+fourier|\brftmw\b|symbolic resonance',re.I)
def date(value):
    try:return datetime.datetime.fromtimestamp(float(value),datetime.timezone.utc).isoformat()
    except (TypeError,ValueError,OverflowError):return ''
def main():
    db=sqlite3.connect(DATA/'research.sqlite',timeout=30)
    items=[];scanned=0
    for folder in Path('H:/Lansky_Offload/Downloads').glob('*2026-07*'):
        if not folder.is_dir():continue
        for file in sorted(folder.glob('conversations-*.json')):
            conversations=json.loads(file.read_text(encoding='utf-8'))
            if isinstance(conversations,dict):conversations=conversations.get('conversations',[])
            file_items=[]
            for c in conversations:
                scanned+=1;messages=[]
                for node in c.get('mapping',{}).values():
                    m=node.get('message') or {};role=m.get('author',{}).get('role')
                    if role not in ('user','assistant'):continue
                    text='\n'.join(p for p in m.get('content',{}).get('parts',[]) if isinstance(p,str))
                    if text:messages.append((m.get('create_time') or 0,role,text,m.get('id','')))
                messages.sort(key=lambda x:x[0]);title=c.get('title') or 'Untitled conversation'
                body='\n\n'.join(f'{date(t)} | {role} | {mid}\n{text}' for t,role,text,mid in messages)
                if not MATCH.search(title+'\n'+body):continue
                ident=c.get('id') or c.get('conversation_id') or hashlib.sha256(body.encode()).hexdigest()
                member='conversation/'+ident;created=date(c.get('create_time'));updated=date(c.get('update_time'))
                transcript=f'{title}\nCreated: {created}\nUpdated: {updated}\nExported user/assistant branches sorted by message time. Media and tool-only content not transcribed.\n\n'+body
                digest=hashlib.sha256(transcript.encode()).hexdigest()
                exists=db.execute('SELECT id FROM documents WHERE sha256=?',(digest,)).fetchone()
                doc_id=exists[0] if exists else db.execute('INSERT INTO documents(sha256,title,kind,body) VALUES(?,?,?,?)',(digest,title,'.conversation',transcript)).lastrowid
                if not db.execute('SELECT 1 FROM occurrences WHERE source=? AND member=?',(str(file),member)).fetchone():
                    db.execute('INSERT INTO occurrences(document_id,source,member,recorded,bytes,status) VALUES(?,?,?,?,?,?)',
                               (doc_id,str(file),member,created,len(transcript.encode()),'indexed'))
                item={'id':doc_id,'title':title,'created':created,'updated':updated,'messages':len(messages),'source':str(file)}
                items.append(item);file_items.append(item)
            if file_items:db.execute('INSERT OR REPLACE INTO archives VALUES(?,?,?,?)',(str(file),len(file_items),min(x['created'] for x in file_items),max(x['updated'] for x in file_items)))
            db.commit();print(file.name,len(file_items),'project conversations',flush=True)
    db.execute("INSERT INTO search(search) VALUES('rebuild')");db.commit()
    report=json.loads((DATA/'index-report.json').read_text())
    report.update(documents=db.execute('SELECT count(*) FROM documents').fetchone()[0],
                  occurrences=db.execute('SELECT count(*) FROM occurrences').fetchone()[0],
                  archives=db.execute('SELECT count(*) FROM archives').fetchone()[0],
                  statuses=dict(db.execute('SELECT status,count(*) FROM occurrences GROUP BY status')),
                  project_conversations=len(items))
    note='Conversation import includes matching user/assistant text across exported branches; media and tool-only content are not transcribed.'
    if note not in report['limits']:report['limits'].append(note)
    db.close();atomic(DATA/'index-report.json',report)
    atomic(DATA/'conversation-history.json',{'scanned':scanned,'matching':len(items),'conversations':sorted(items,key=lambda x:x['created'])})
    print('Imported',len(items),'of',scanned,'conversations')
if __name__=='__main__':main()
