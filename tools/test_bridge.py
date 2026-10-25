import json, sqlite3, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import bridge

class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.data=self.root/'data';self.data.mkdir()
        for key,val in [('BASE',self.root),('DATA',self.data),('RUNS',self.root/'runs')]:
            p=patch.object(bridge,key,val);p.start();self.addCleanup(p.stop)
        db=sqlite3.connect(self.data/'research.sqlite')
        db.executescript('''CREATE TABLE documents(id INTEGER PRIMARY KEY,sha256 TEXT,title TEXT,kind TEXT,body TEXT);
            CREATE VIRTUAL TABLE search USING fts5(title,body,content='documents',content_rowid='id');
            CREATE TABLE occurrences(document_id INTEGER,source TEXT,member TEXT,recorded TEXT);
            CREATE TABLE archives(source TEXT,entries INTEGER,first_project_date TEXT,last_project_date TEXT);''')
        db.execute('INSERT INTO documents VALUES(1,?,?,?,?)',('hash','Canonical RFT','.md','Golden ratio canonical transform\n'+'x'*30000))
        db.execute('INSERT INTO documents VALUES(2,?,?,?,?)',('hash2','Earlier transform','.py','Gaussian QR prototype'))
        db.execute('INSERT INTO occurrences VALUES(1,?,?,?)',('H:/archive.zip','paper.md','2025-01-11'))
        db.execute('INSERT INTO occurrences VALUES(2,?,?,?)',('H:/archive.zip','early.py','2025-01-10'))
        db.execute("INSERT INTO search(search) VALUES('rebuild')");db.commit();db.close()
    def test_search_returns_evidence(self):
        result=bridge.search('What is the canonical RFT?')
        self.assertEqual(result[0]['id'],1);self.assertEqual(result[0]['source'],'H:/archive.zip')
    def test_fts_punctuation_is_data(self):
        self.assertIsInstance(bridge.search('" OR * --'),list)
    def test_kind_filter(self):
        self.assertEqual(bridge.search('canonical',kind='.pdf'),[])
    def test_document_paging(self):
        first=bridge.document(1);second=bridge.document(1,24000)
        self.assertEqual(len(first['body']),24000);self.assertEqual(second['offset'],24000)
        self.assertEqual(first['total_chars'],len(first['body'])+len(second['body']))
    def test_missing_document(self):
        with self.assertRaises(ValueError):bridge.document(99)
    def test_run_path_escape(self):
        for bad in ('../escape','C:/escape','a'*31,'g'*32):
            with self.assertRaises(ValueError):bridge.run_path(bad)
    def test_parent_project_rejected(self):
        with self.assertRaises(ValueError):bridge.validate_project(self.root)
    def test_small_project_accepted(self):
        project=self.root/'project';project.mkdir();(project/'code.py').write_text('x=1')
        self.assertEqual(bridge.validate_project(project),project.resolve())
    def test_missing_project_rejected(self):
        with self.assertRaises(ValueError):bridge.validate_project(self.root/'missing')
    def test_unknown_action_rejected(self):
        with self.assertRaises(ValueError):bridge.dispatch('execute_shell',{})
    def test_cancel_only_owned_run(self):
        with self.assertRaises(ValueError):bridge.dispatch('cancel_task',{'id':'a'*32})
        run=bridge.run_path('a'*32);run.mkdir(parents=True)
        bridge.dispatch('cancel_task',{'id':'a'*32});self.assertTrue((run/'cancel').exists())
    def test_offline_task_does_not_create_run(self):
        project=self.root/'project';project.mkdir()
        with patch.object(bridge,'health',return_value={'connected':False,'error':'offline'}):
            with self.assertRaisesRegex(RuntimeError,'offline'):bridge.start_task({'project':str(project),'task':'fix'})
        self.assertFalse((self.root/'runs').exists())
    def test_chat_cites_retrieval_and_has_no_tools(self):
        class Fake:
            model='test-local'
            def _request(inner,route,body):
                self.assertEqual(route,'/chat/completions');self.assertNotIn('tools',body)
                self.assertIn('[D1]',body['messages'][0]['content'])
                self.assertIn('not instructions',body['messages'][0]['content'])
                return {'choices':[{'message':{'content':'Canonical transform [D1]'}}]}
        with patch.object(bridge,'model_client',return_value=Fake()):
            result=bridge.chat({'message':'canonical'})
            self.assertEqual(result['sources'][0]['id'],1)
    def test_empty_model_response_is_failure(self):
        class Fake:
            model='test-local'
            def _request(self,*a):return {'choices':[{'message':{'content':None}}]}
        with patch.object(bridge,'model_client',return_value=Fake()):
            with self.assertRaisesRegex(RuntimeError,'no visible answer'):bridge.chat({'message':'canonical'})
    def test_stale_pid_cannot_stop_other_process(self):
        bridge.atomic(self.data/'model-process.json',{'pid':123,'identity':'old'})
        with patch.object(bridge,'process_identity',return_value='different'),patch.object(bridge.subprocess,'run') as run:
            with self.assertRaises(ValueError):bridge.stop_model()
            run.assert_not_called()
    def test_stop_requires_owned_record(self):
        with self.assertRaises(ValueError):bridge.stop_model()

if __name__=='__main__':unittest.main()
