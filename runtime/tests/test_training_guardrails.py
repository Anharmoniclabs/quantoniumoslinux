# SPDX-License-Identifier: AGPL-3.0-or-later

from src.apps.training_guardrails import GuardrailPolicy, GuardrailStats, sanitize_text


def test_drops_obvious_secrets():
    policy = GuardrailPolicy(drop_secrets=True, redact_pii=False, drop_tool_instructions=False)
    stats = GuardrailStats()
    assert sanitize_text("AKIA1234567890ABCDEF some text", policy, stats) is None
    assert stats.dropped_secrets == 1


def test_redacts_email():
    policy = GuardrailPolicy(drop_secrets=False, redact_pii=True, drop_tool_instructions=False)
    stats = GuardrailStats()
    out = sanitize_text("Contact me at alice@example.com please.", policy, stats)
    assert out is not None
    assert "[REDACTED_EMAIL]" in out
    assert stats.redacted_emails == 1


def test_drops_tool_instructions():
    policy = GuardrailPolicy(drop_secrets=False, redact_pii=False, drop_tool_instructions=True)
    stats = GuardrailStats()
    assert sanitize_text("Run this command: pip install torch", policy, stats) is None
    assert stats.dropped_tool_instructions == 1

