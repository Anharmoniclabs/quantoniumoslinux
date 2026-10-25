"""Fill native-code and launcher text omitted by the initial text-extension pass."""
from pathlib import Path, PurePosixPath
import hashlib, io, json, sqlite3, zipfile
from bridge import DATA,atomic
EXTRA={'.c','.cc','.cpp','.cxx','.h','.hpp','.asm','.s','.sh','.ps1','.bat','.cmd','.jl','.r','.gd','.rst','.qss','.tlv'}
def load(source,member):
    if not source.lower().endswith('.zip'):return Path(source,member).read_bytes()
    parts=source.split('!'); z=zipfile.ZipFile(parts[0]); opened=[z]
    try:
        for inner in parts[1:]:
            z=zipfile.ZipFile(io.BytesIO(z.read(inner)));opened.append(z)
        return z.read(member)
    finally:
        for archive in opened:archive.close()
def main():
    db=sqlite3.connect(DATA/'research.sqlite',timeout=30);count=0;errors=[]
    rows=db.execute("SELECT id,source,member,bytes FROM occurrences WHERE status='metadata-only'").fetchall()
    for oid,source,member,size in rows:
        path=PurePosixPath(member)
        if path.suffix.lower() not in EXTRA or size>3_000_000:continue
        if any(p.lower() in {'third_party','vendor','vendors','external','extern','include','lib','libs'} for p in path.parts):continue
        try:
            raw=load(source,member)
            if b'\0' in raw[:8192]:continue
            digest=hashlib.sha256(raw).hexdigest();old=db.execute('SELECT id FROM documents WHERE sha256=?',(digest,)).fetchone()
            doc_id=old[0] if old else db.execute('INSERT INTO documents(sha256,title,kind,body) VALUES(?,?,?,?)',(digest,path.name,path.suffix.lower(),raw.decode('utf-8','replace'))).lastrowid
            db.execute("UPDATE occurrences SET document_id=?,status='indexed' WHERE id=?",(doc_id,oid));count+=1
        except Exception as e:errors.append({'source':source,'member':member,'error':str(e)})
    db.execute("INSERT INTO search(search) VALUES('rebuild')");db.commit()
    report=json.loads((DATA/'index-report.json').read_text())
    report.update(documents=db.execute('SELECT count(*) FROM documents').fetchone()[0],statuses=dict(db.execute('SELECT status,count(*) FROM occurrences GROUP BY status')))
    report['errors'].extend(errors);db.close();atomic(DATA/'index-report.json',report)
    print('Native-code and launcher occurrences indexed:',count,'errors:',len(errors))
if __name__=='__main__':main()
