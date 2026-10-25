"""Read-only, content-addressed history index. Never executes archived source."""
from pathlib import Path, PurePosixPath
import hashlib, io, json, os, sqlite3, sys, time, zipfile

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
SKIP = {'.git', 'node_modules', 'target', '__pycache__', '.venv', 'venv', 'env',
        'site-packages', 'dist', 'build', 'release', '.cache', '$RECYCLE.BIN',
        'System Volume Information', 'models', 'hf_models', 'blobs'}
TEXT = {'.md', '.txt', '.tex', '.py', '.rs', '.ts', '.tsx', '.js', '.jsx', '.json',
        '.yaml', '.yml', '.toml', '.csv', '.sv', '.v', '.html', '.css', '.ipynb', '.cff',
        '.c', '.cc', '.cpp', '.cxx', '.h', '.hpp', '.asm', '.s', '.sh', '.ps1', '.bat', '.cmd', '.jl', '.gd', '.rst', '.qss', '.tlv'}
KEYWORDS = ('quant', 'nessa', 'eldrae', 'research', 'early', 'full', 'zenodo', 'rft', 'personalai', 'nonprovis')

def eligible(name):
    p = PurePosixPath(name.replace('\\', '/'))
    return not any(x.lower() in SKIP or x.lower().endswith('.egg-info') for x in p.parts) and not p.name.startswith('.env') and p.name.lower() not in {'seeds.txt', 'wallet.dat'}

def extract(raw, suffix):
    if suffix == '.pdf':
        from pypdf import PdfReader
        return '\n\n'.join(p.extract_text() or '' for p in PdfReader(io.BytesIO(raw)).pages)
    if suffix == '.docx':
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            return ' '.join(ET.fromstring(z.read('word/document.xml')).itertext())
    if b'\0' in raw[:8192]:
        if raw.startswith((b'\xff\xfe', b'\xfe\xff')): return raw.decode('utf-16')
        return ''
    return raw.decode('utf-8-sig', 'replace')

def build():
    DATA.mkdir(exist_ok=True)
    target = DATA / 'research.sqlite'
    staging = DATA / 'research-building.sqlite'
    if staging.exists(): staging.unlink()
    db = sqlite3.connect(staging)
    db.executescript('''
      CREATE TABLE documents(id INTEGER PRIMARY KEY, sha256 TEXT UNIQUE, title TEXT, kind TEXT, body TEXT);
      CREATE VIRTUAL TABLE search USING fts5(title, body, content='documents', content_rowid='id');
      CREATE TABLE occurrences(id INTEGER PRIMARY KEY, document_id INTEGER, source TEXT, member TEXT, recorded TEXT, bytes INTEGER, status TEXT);
      CREATE INDEX by_document ON occurrences(document_id);
      CREATE TABLE archives(source TEXT PRIMARY KEY, entries INTEGER, first_project_date TEXT, last_project_date TEXT);
    ''')
    cache = {}; errors = []; processed = 0
    def add(source, name, stamp, size, loader):
        nonlocal processed
        processed += 1
        suffix = PurePosixPath(name).suffix.lower()
        doc = None; state = 'metadata-only'
        if suffix in TEXT | {'.pdf', '.docx'} and size <= (20_000_000 if suffix in {'.pdf','.docx'} else 3_000_000):
            try:
                raw = loader(); digest = hashlib.sha256(raw).hexdigest()
                if digest in cache: doc = cache[digest]; state = 'indexed'
                else:
                    body = extract(raw, suffix)
                    if body.strip():
                        cur = db.execute('INSERT INTO documents(sha256,title,kind,body) VALUES(?,?,?,?)',
                                         (digest, PurePosixPath(name).name, suffix, body))
                        doc = cur.lastrowid; cache[digest] = doc; state = 'indexed'
                    else: state = 'no-extractable-text'
            except Exception as e:
                state = 'extraction-error'; errors.append({'source':source, 'member':name, 'error':str(e)[:500]})
        db.execute('INSERT INTO occurrences(document_id,source,member,recorded,bytes,status) VALUES(?,?,?,?,?,?)',
                   (doc, source, name, stamp, size, state))
        if processed % 3000 == 0:
            db.commit(); print(f'Indexed metadata for {processed} files; {len(cache)} distinct texts', flush=True)

    def archive(path, raw=None, depth=0):
        source = str(path)
        try:
            with zipfile.ZipFile(io.BytesIO(raw) if raw is not None else path) as z:
                infos = [i for i in z.infolist() if not i.is_dir() and eligible(i.filename)]
                dates = ['%04d-%02d-%02dT%02d:%02d:%02d' % i.date_time for i in infos]
                db.execute('INSERT OR IGNORE INTO archives VALUES(?,?,?,?)', (source,len(infos),min(dates,default=''),max(dates,default='')))
                for i in infos:
                    date = '%04d-%02d-%02dT%02d:%02d:%02d' % i.date_time
                    add(source,i.filename,date,i.file_size,lambda i=i:z.read(i))
                    if i.filename.lower().endswith('.zip') and depth < 3 and i.file_size < 64_000_000:
                        archive(source+'!'+i.filename,z.read(i),depth+1)
        except Exception as e: errors.append({'source':source,'error':str(e)[:500]})

    roots = [Path('H:/DESKTOP-ZIPPED PROJECTS'),Path('H:/files (pdf,exel,word)'),
             Path('H:/misc'),Path('H:/Lansky_Offload/Downloads'),Path('H:/quantonium_os_src'),
             Path('H:/Code (misc)'),Path('H:/NESSA-Colab-Validation-Kit'),Path('H:/personalAi (6)'),
             Path('H:/quantonium-mobile'),BASE/'runtime',BASE/'nessa',BASE/'reference-main']
    for root in roots:
        print('Scanning '+str(root),flush=True)
        selective = root in [Path('H:/misc'),Path('H:/Lansky_Offload/Downloads')]
        for parent, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP and not Path(parent,d).is_symlink())
            if selective and Path(parent)==root:
                dirs[:] = [d for d in dirs if any(k in d.lower() for k in KEYWORDS)]
            for name in sorted(files):
                p = Path(parent,name); rel = p.relative_to(root).as_posix()
                if not eligible(rel):continue
                if selective and Path(parent)==root and not any(k in name.lower() for k in KEYWORDS):continue
                try:
                    stat=p.stat()
                    if p.suffix.lower()=='.zip': archive(p)
                    else: add(str(root),rel,time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(stat.st_mtime)),stat.st_size,p.read_bytes)
                except OSError as e:errors.append({'source':str(p),'error':str(e)})
    db.execute("INSERT INTO search(search) VALUES('rebuild')")
    db.commit()
    stats={'documents':db.execute('SELECT count(*) FROM documents').fetchone()[0],
           'occurrences':db.execute('SELECT count(*) FROM occurrences').fetchone()[0],
           'archives':db.execute('SELECT count(*) FROM archives').fetchone()[0],
           'statuses':dict(db.execute('SELECT status,count(*) FROM occurrences GROUP BY status')),
           'created':time.strftime('%Y-%m-%dT%H:%M:%S'), 'errors':errors,
           'roots':[str(r) for r in roots],
           'limits':['Archive timestamps are provenance hints, not proven authorship dates.',
                     'Build products, dependencies, model weights and credential files excluded.',
                     'Nested ZIPs limited to three levels and 64 MB; larger members remain metadata-only.',
                     'Text/code over 3 MB and PDFs/DOCX over 20 MB remain metadata-only.',
                     'Images, audio, video and scanned documents without text need separate review.',
                     'Indexing preserves source evidence; it does not validate every research claim.']}
    db.close();os.replace(staging,target)
    (DATA/'index-report.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in stats.items() if k not in ['errors','roots']},indent=2),flush=True)

if __name__=='__main__':build()
