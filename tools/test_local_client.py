import json,unittest
from unittest.mock import patch
import bridge
from local_client import ConstrainedLocalClient,tool_response_schema
from agentharness.llm import ChatClient

class LocalClientTests(unittest.TestCase):
    def test_schema_cannot_expand_allowed_tools(self):
        s=tool_response_schema({'read_file','propose_plan'})
        names={v['properties']['name']['const'] for v in s['oneOf']}
        self.assertEqual(names,{'read_file','propose_plan'})
        for v in s['oneOf']:self.assertFalse(v['properties']['arguments']['additionalProperties'])
    def test_shell_tool_not_available(self):
        with self.assertRaises(ValueError):tool_response_schema({'run_command'})
    def test_request_uses_schema_and_normal_parser(self):
        c=ConstrainedLocalClient('http://127.0.0.1:18081/v1','test')
        def request(client,route,body,timeout):
            self.assertEqual(body['response_format']['type'],'json_object')
            self.assertEqual(body['response_format']['schema']['oneOf'][0]['properties']['name']['const'],'read_file')
            return {'choices':[{'message':{'content':json.dumps({'name':'read_file','arguments':{'path':'calc.py'}})}}]}
        with patch.object(ChatClient,'_request',request):
            reply=c.chat([{'role':'user','content':'read'}],tool_names={'read_file'})
            self.assertEqual(reply.tool_calls[0].arguments,{'path':'calc.py'})
if __name__=='__main__':unittest.main()
