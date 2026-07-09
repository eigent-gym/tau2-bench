import json
import subprocess
from pathlib import Path

from tau2.data_model.message import AssistantMessage
from tau2.user.codex_user_simulator import (
    CodexUserSimulator,
    _extract_json_object,
)
from tau2.user.user_simulator_base import STOP


def _mock_codex_run(monkeypatch, payload: dict):
    calls = {}

    def fake_run(cmd, input, text, capture_output, timeout, cwd, env, check):
        calls["cmd"] = cmd
        calls["input"] = input
        calls["timeout"] = timeout
        calls["cwd"] = cwd
        calls["env"] = env
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("tau2.user.codex_user_simulator.subprocess.run", fake_run)
    return calls


def test_codex_user_simulator_returns_text(monkeypatch):
    calls = _mock_codex_run(
        monkeypatch,
        {"content": "I need help with my card.", "tool_calls": [], "stop": False},
    )
    user = CodexUserSimulator(
        llm="gpt-5.5",
        instructions="You are Sarah. Ask about your credit card.",
        llm_args={
            "reasoning_effort": "xhigh",
            "timeout_seconds": 12,
            "codex_extra_args": ["--ephemeral", "--skip-git-repo-check"],
        },
    )

    state = user.get_init_state()
    message, state = user.generate_next_message(
        AssistantMessage(role="assistant", content="Hello, how can I help?"),
        state,
    )

    assert message.content == "I need help with my card."
    assert message.tool_calls is None
    assert state.messages[-1] == message
    assert calls["cmd"][:4] == ["codex", "exec", "--model", "gpt-5.5"]
    assert '-c' in calls["cmd"]
    assert 'model_reasoning_effort="xhigh"' in calls["cmd"]
    assert calls["cmd"][-1] == "-"
    assert calls["timeout"] == 12
    assert "ASSISTANT_UNDER_TEST: Hello, how can I help?" in calls["input"]


def test_codex_user_simulator_returns_user_tool_call(monkeypatch):
    _mock_codex_run(
        monkeypatch,
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "apply_for_credit_card",
                    "arguments": {"card_type": "Gold Rewards Card"},
                }
            ],
            "stop": False,
        },
    )
    user = CodexUserSimulator(
        llm="gpt-5.5",
        instructions="Apply for the right card once the assistant recommends it.",
    )

    state = user.get_init_state()
    message, _ = user.generate_next_message(
        AssistantMessage(role="assistant", content="The Gold Rewards Card fits best."),
        state,
    )

    assert message.content is None
    assert message.tool_calls is not None
    assert message.tool_calls[0].id == "call_1"
    assert message.tool_calls[0].name == "apply_for_credit_card"
    assert message.tool_calls[0].arguments == {"card_type": "Gold Rewards Card"}
    assert message.tool_calls[0].requestor == "user"


def test_codex_user_simulator_stop_without_content():
    user = CodexUserSimulator(llm="gpt-5.5", instructions="Stop immediately.")

    message = user._response_to_user_message(
        {"content": None, "tool_calls": [], "stop": True}
    )

    assert message.content == STOP
    assert user.is_stop(message)


def test_extract_json_object_from_wrapped_text():
    assert _extract_json_object('status\n{"content": "ok", "tool_calls": [], "stop": false}') == {
        "content": "ok",
        "tool_calls": [],
        "stop": False,
    }
