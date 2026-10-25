"""Deterministic protocol/approval boundaries; no model or optional packages required."""
import json
import unittest

from agentharness.llm import ChatClient, ContextOverflow, ModelError, parse_text_tool_calls
from agentharness.tools import build_tools
from agentharness.workspace import ToolError
from agentharness.tests.test_agent import Base
from agentharness.tests.test_http_cli import FakeServer


class ProtocolFlow(Base):
    def test_malformed_text_feedback_then_repair(self):
        agent = self.agent([
            '<tool_call>{"name":"read_file","arguments":</tool_call>',
            [("finish", {"summary": "blocked"})],
        ], plan_first=False, baseline_checks=False)
        result = agent.run("inspect")
        self.assertEqual(result.status, "no_change")
        feedback = json.dumps(self.client.seen[-1][0])
        self.assertIn("Invalid tool-call JSON", feedback)
        self.assertIn("Repair attempts remaining: 2", feedback)

    def test_schema_errors_stop_at_repair_budget(self):
        agent = self.agent([[("read_file", {"file": "calc.py"})]],
                           plan_first=False, baseline_checks=False, tool_retries=2)
        result = agent.run("inspect")
        self.assertEqual(result.status, "error")
        self.assertEqual(result.steps, 3)
        self.assertIn("repair budget exhausted", result.summary)
        self.assertEqual(agent.ws.journal, [])
        events = [json.loads(line) for line in (agent.evidence_dir / "events.jsonl").read_text().splitlines()]
        self.assertEqual(sum(e['event'] == 'protocol_error' for e in events), 3)

    def test_invalid_finish_cannot_end_run(self):
        result = self.agent([[("finish", {})]], plan_first=False,
                            baseline_checks=False, tool_retries=0).run("inspect")
        self.assertEqual(result.status, "error")
        self.assertEqual(result.steps, 1)

    def test_multiple_finish_calls_cannot_leave_unanswered_native_calls(self):
        agent = self.agent([[('finish', {'summary': 'one'}), ('finish', {'summary': 'two'})]],
                           plan_first=False, baseline_checks=False, tool_retries=0)
        self.assertEqual(agent.run('inspect').status, 'error')
        responses = [m for m in agent.messages if m['role'] == 'tool']
        self.assertEqual([m['tool_call_id'] for m in responses], ['c0', 'c1'])

    def test_mixed_malformed_text_never_executes_valid_nested_edits(self):
        write = '{"name":"write_file","arguments":{"path":"leak.txt","content":"leak"}}'
        for text in ('<tool_call>{"arguments":{}} ' + write + '</tool_call>',
                     '<tool_call>' + write + '</tool_call><tool_call>{"name":"finish"}',
                     '<tool_call>[' + write + ',</tool_call>',
                     '{"arguments":{}} ' + write,
                     '<tool_call>' + write + '</tool_call> {"name":"finish","arguments":',
                     '```json\n' + write + '\n``` {"name":"finish","arguments":'):
            with self.subTest(text=text):
                agent = self.agent([text], plan_first=False, baseline_checks=False, tool_retries=0)
                self.assertEqual(agent.run('inspect').status, 'error')
                self.assertFalse((self.ws.repo / 'leak.txt').exists())

    def test_invalid_batch_has_no_effects(self):
        agent = self.agent([[("write_file", {"path": "new.py", "content": "x=1\n"}),
                             ("finish", {"summary": 42})]],
                           plan_first=False, baseline_checks=False, tool_retries=0)
        result = agent.run("write")
        self.assertEqual(result.status, "error")
        self.assertEqual(result.patch, "")
        self.assertFalse((self.ws.repo / "new.py").exists())
        self.assertEqual(agent.edit_count, 0)

    def test_valid_turn_resets_consecutive_repair_budget(self):
        agent = self.agent([
            [("read_file", {"file": "calc.py"})],
            [("read_file", {"path": "calc.py"})],
            [("read_file", {"file": "calc.py"})],
            [("finish", {"summary": "blocked"})],
        ], plan_first=False, baseline_checks=False, tool_retries=1)
        self.assertEqual(agent.run("inspect").status, "no_change")

    def test_invalid_plan_is_bounded(self):
        agent = self.agent([[("propose_plan", {"goal": "g", "steps": [False]})]],
                           baseline_checks=False, tool_retries=1)
        result = agent.run("inspect")
        self.assertEqual(result.status, "error")
        self.assertEqual(len(self.client.seen), 2)
        self.assertIsNone(result.plan)

    def test_unknown_text_tool_reports_name(self):
        result = self.agent(['{"name":"invented","arguments":{}}'],
                            plan_first=False, baseline_checks=False, tool_retries=0).run("inspect")
        self.assertEqual(result.status, "error")
        messages = json.loads((self.ws.work_dir / 'evidence/messages.json').read_text())
        self.assertIn("invented", json.dumps(messages))
        self.assertIn("not available", json.dumps(messages))

    def test_planning_extraction_cannot_write(self):
        for name in ("extract_table", "export_table"):
            with self.subTest(name=name):
                agent = self.agent([[(name, {"paths": ["calc.py"], "out": "leak.csv"})]],
                                   baseline_checks=False, tool_retries=0)
                result = agent.run("inspect")
                self.assertEqual(result.status, "error")
                self.assertEqual(result.patch, "")
                self.assertNotIn("export_table", self.client.seen[0][1])
                self.assertFalse((self.ws.repo / "leak.csv").exists())

    def test_export_has_journal_and_checkpoint(self):
        agent = self.agent([
            [("export_table", {"paths": ["calc.py"], "out": "rows.csv"})],
            [("finish", {"summary": "exported"})],
        ], plan_first=False, baseline_checks=False, verify=(), continuous_verify=())
        result = agent.run("export")
        self.assertEqual(result.status, "unverified")
        self.assertIn("rows.csv", result.changed_files)
        self.assertEqual(len(self.ws.journal), 1)
        self.assertEqual(agent.edit_count, 1)
        self.assertEqual(len(list((agent.evidence_dir / "checkpoints").glob("*.diff"))), 1)

    def test_export_cannot_overwrite_unread_file(self):
        agent = self.agent([
            [("export_table", {"paths": ["calc.py"], "out": "calc.py"})],
            [("finish", {"summary": "blocked"})],
        ], plan_first=False, baseline_checks=False)
        self.assertEqual(agent.run("export").status, "no_change")
        self.assertEqual(self.ws.journal, [])


class ProtocolUnits(unittest.TestCase):
    def test_malformed_outer_never_promotes_nested_tool(self):
        calls = parse_text_tool_calls('{"name":"bad", broken: {"name":"finish","arguments":{"summary":"done"}}}')
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].error)
        self.assertIsNone(calls[0].arguments)

    def test_empty_and_nonobject_tagged_calls_are_errors(self):
        for text in ('<tool_call></tool_call>', '<tool_call>[]</tool_call>',
                     '<tool_call>{"arguments":{}}</tool_call>'):
            with self.subTest(text=text):
                self.assertTrue(parse_text_tool_calls(text)[0].error)

    def test_array_items_and_boolean_integer_rejected(self):
        tools = build_tools()
        for name, args in [("dev_start", {"name":"x", "argv":[123]}),
                           ("read_file", {"path":"x", "start":True}),
                           ("propose_plan", {"goal":"g", "steps":[{}]})]:
            with self.subTest(name=name), self.assertRaises(ToolError):
                tools[name].validate(args)
        self.assertEqual(tools['read_file'].validate({'path':'x','start':'2'})['start'], 2)

    def test_malformed_native_metadata_produces_repairable_calls(self):
        client = ChatClient("http://localhost/v1", "m")
        for calls in ([None], [{"function": {"name": "read_file", "arguments": 123}}],
                      [{"function": {"name": [], "arguments": "{}"}}], {"function": None}):
            with self.subTest(calls=calls):
                client._request = lambda *args: {"choices": [{"message": {"tool_calls": calls}}]}
                reply = client.chat([])
                self.assertTrue(reply.native)
                self.assertTrue(reply.tool_calls[0].error)
                self.assertIsInstance(reply.raw_tool_calls[0]['function']['name'], str)
                self.assertIsInstance(reply.raw_tool_calls[0]['function']['arguments'], str)

    def test_oversized_integer_is_a_validation_error(self):
        with self.assertRaises(ToolError):
            build_tools()['read_file'].validate({'path': 'x', 'start': '9' * 5000})

    def test_completion_usage_remains_unavailable_when_omitted(self):
        client = ChatClient("http://localhost/v1", "m")
        response = {"choices": [{"message": {"content": "hello"}}]}
        client._request = lambda *args: response
        self.assertIsNone(client.chat([]).completion_tokens)
        self.assertIsNone(client.chat([]).prompt_tokens)
        response["usage"] = {"completion_tokens": 7}
        self.assertEqual(client.chat([]).completion_tokens, 7)

    def test_bad_request_is_not_context_overflow_or_retried(self):
        server = FakeServer("bad_request")
        self.addCleanup(server.close)
        try:
            ChatClient(server.url, "m").chat([])
        except ModelError as exc:
            self.assertNotIsInstance(exc, ContextOverflow)
            self.assertIn("unsupported", str(exc))
        else:
            self.fail("Expected ModelError")
        self.assertEqual(len(server.requests), 1)
