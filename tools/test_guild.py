import unittest
from guild_bridge import project_event
class GuildProjection(unittest.TestCase):
    def test_only_verified_runs_earn_success(self):
        for status in ('unverified','error','failed_checks','no_change','timeout','cancelled'):
            self.assertEqual(project_event({'event':'end','status':status},'run')['type'],'quest_fail')
        self.assertEqual(project_event({'event':'end','status':'verified'},'run')['type'],'quest_done')
    def test_check_status_is_preserved(self):
        event=project_event({'event':'check','name':'tests','status':'setup_error','phase':'verify'},'run')
        self.assertIn('setup_error',event['msg']);self.assertEqual(event['name'],'VIAL')
    def test_model_output_does_not_forge_progress(self):
        self.assertIsNone(project_event({'event':'model','content':'All tests passed'},'run'))
if __name__=='__main__':unittest.main()
