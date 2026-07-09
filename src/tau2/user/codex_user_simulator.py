"""
Codex-backed user simulator.

This module keeps the tau2 user protocol unchanged while delegating the next
user turn to the Codex CLI instead of LiteLLM.
"""

import json
import os
import subprocess
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.persona import PersonaConfig
from tau2.environment.tool import Tool
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import (
    OUT_OF_SCOPE,
    STOP,
    TRANSFER,
    UserState,
    ValidUserInputMessage,
)

CODEX_USER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "content": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ],
            "description": "Text to send to the assistant under test.",
        },
        "tool_calls": {
            "type": "array",
            "description": "User-side tool calls to execute, if any.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "arguments": {
                        "description": "JSON object of tool arguments.",
                        "anyOf": [
                            {"type": "object"},
                            {"type": "string"},
                        ],
                    },
                },
                "required": ["name", "arguments"],
            },
        },
        "stop": {
            "type": "boolean",
            "description": "Set true only when the simulated user is done.",
        },
    },
    "required": ["content", "tool_calls", "stop"],
}


class CodexUserSimulator(UserSimulator):
    """A half-duplex user simulator that calls ``codex exec`` for each turn.

    Supported ``llm_args`` keys:
    - ``codex_cmd``: executable path, defaults to ``codex``.
    - ``codex_extra_args``: extra CLI args inserted before the prompt.
    - ``reasoning_effort``: mapped to ``-c model_reasoning_effort="<value>"``.
    - ``timeout_seconds``: subprocess timeout, defaults to 300.
    - ``codex_cwd``: working directory for Codex, defaults to current process cwd.
    - ``codex_env``: environment overrides for the subprocess.
    """

    def __init__(
        self,
        llm: str,
        instructions: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
        llm_args: Optional[dict] = None,
        persona_config: Optional[PersonaConfig] = None,
    ):
        super().__init__(
            llm=llm,
            instructions=instructions,
            tools=tools,
            llm_args=llm_args,
            persona_config=persona_config,
        )
        codex_args = deepcopy(llm_args) if llm_args is not None else {}
        self.codex_cmd = str(codex_args.get("codex_cmd", "codex"))
        self.codex_extra_args = list(
            codex_args.get(
                "codex_extra_args",
                [
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "--sandbox",
                    "read-only",
                    "--color",
                    "never",
                    "--ignore-user-config",
                    "--ignore-rules",
                ],
            )
        )
        self.reasoning_effort = codex_args.get("reasoning_effort")
        self.timeout_seconds = int(codex_args.get("timeout_seconds", 300))
        self.codex_cwd = codex_args.get("codex_cwd")
        self.codex_env = dict(codex_args.get("codex_env", {}))

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> UserMessage:
        if isinstance(message, AssistantMessage) and message.is_audio:
            raise ValueError(
                "Assistant message cannot be audio. Use VoiceUserSimulator instead."
            )
        logger.debug(f"Codex user responds to message: {message}")

        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif isinstance(message, ToolMessage):
            state.messages.append(message)
        elif message.has_content() or message.is_tool_call():
            state.messages.append(message)

        prompt = self._build_codex_prompt(state)
        response = self._run_codex(prompt)
        user_message = self._response_to_user_message(response)
        logger.debug(f"Codex user response: {user_message}")
        return user_message

    def _build_codex_prompt(self, state: UserState) -> str:
        tool_schemas = []
        if self.tools is not None:
            tool_schemas = [tool.openai_schema for tool in self.tools]

        return "\n\n".join(
            [
                "You are the tau2 simulated user. Produce exactly one user turn.",
                "Follow the system prompt and scenario below. Do not solve the assistant's side of the task.",
                "Return only JSON matching this schema: "
                '{"content": string|null, "tool_calls": array, "stop": boolean}.',
                f"When the user is finished, set stop=true and include one of: {STOP}, {TRANSFER}, {OUT_OF_SCOPE}.",
                "If you need to use a user-side tool, add it to tool_calls. The runtime will execute it after your turn.",
                "Do not wrap the JSON in Markdown.",
                "<system_prompt>",
                state.system_messages[0].content or "",
                "</system_prompt>",
                "<available_user_tools>",
                json.dumps(tool_schemas, indent=2, sort_keys=True),
                "</available_user_tools>",
                "<conversation>",
                self._render_conversation(state),
                "</conversation>",
            ]
        )

    def _render_conversation(self, state: UserState) -> str:
        rendered: list[str] = []
        for message in state.messages:
            if isinstance(message, AssistantMessage):
                rendered.append(f"ASSISTANT_UNDER_TEST: {message.content or ''}")
            elif isinstance(message, UserMessage):
                rendered.append(f"SIMULATED_USER: {message.content or ''}")
                if message.tool_calls:
                    rendered.append(
                        "SIMULATED_USER_TOOL_CALLS: "
                        + json.dumps(
                            [tool_call.model_dump() for tool_call in message.tool_calls],
                            sort_keys=True,
                        )
                    )
            elif isinstance(message, ToolMessage):
                rendered.append(
                    "USER_TOOL_RESULT: "
                    + json.dumps(
                        {
                            "id": message.id,
                            "content": message.content,
                            "error": message.error,
                            "requestor": message.requestor,
                        },
                        sort_keys=True,
                    )
                )
        return "\n".join(rendered)

    def _run_codex(self, prompt: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="tau2-codex-user-") as tmpdir:
            tmp_path = Path(tmpdir)
            output_path = tmp_path / "last_message.json"
            schema_path = tmp_path / "response_schema.json"
            schema_path.write_text(json.dumps(CODEX_USER_RESPONSE_SCHEMA), encoding="utf-8")

            cmd = [
                self.codex_cmd,
                "exec",
                "--model",
                self.llm,
                "--output-last-message",
                str(output_path),
                "--output-schema",
                str(schema_path),
            ]
            if self.reasoning_effort:
                cmd.extend(
                    [
                        "-c",
                        f'model_reasoning_effort="{self.reasoning_effort}"',
                    ]
                )
            cmd.extend(self.codex_extra_args)
            cmd.append("-")

            env = os.environ.copy()
            env.update(self.codex_env)
            completed = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                cwd=self.codex_cwd,
                env=env,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Codex user simulator failed with exit code "
                    f"{completed.returncode}: {completed.stderr[-2000:]}"
                )

            raw_response = ""
            if output_path.exists():
                raw_response = output_path.read_text(encoding="utf-8").strip()
            if not raw_response:
                raw_response = completed.stdout.strip()
            return _extract_json_object(raw_response)

    def _response_to_user_message(self, response: dict[str, Any]) -> UserMessage:
        content = response.get("content")
        tool_calls_data = response.get("tool_calls") or []
        stop = bool(response.get("stop", False))

        if stop and not content and not tool_calls_data:
            content = STOP
        if content is not None and not isinstance(content, str):
            raise ValueError(f"Codex user content must be a string or null: {content!r}")
        if not isinstance(tool_calls_data, list):
            raise ValueError("Codex user tool_calls must be a list")

        tool_calls = [
            _build_user_tool_call(tool_call_data) for tool_call_data in tool_calls_data
        ]
        if not content and not tool_calls:
            raise ValueError("Codex user response must include content or tool_calls")

        return UserMessage(
            role="user",
            content=content,
            tool_calls=tool_calls or None,
            raw_data={"codex_user_response": response},
        )


def _build_user_tool_call(tool_call_data: dict[str, Any]) -> ToolCall:
    if not isinstance(tool_call_data, dict):
        raise ValueError(f"Tool call must be an object: {tool_call_data!r}")
    name = tool_call_data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Tool call name must be a non-empty string: {tool_call_data!r}")

    arguments = tool_call_data.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError(f"Tool call arguments must be an object: {tool_call_data!r}")

    tool_call_id = tool_call_data.get("id") or f"codex_user_tool_{uuid.uuid4().hex[:8]}"
    return ToolCall(
        id=str(tool_call_id),
        name=name,
        arguments=arguments,
        requestor="user",
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Codex user simulator returned an empty response")

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Codex user simulator did not return a JSON object: {text!r}")
