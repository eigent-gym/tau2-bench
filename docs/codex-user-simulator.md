# Codex User Simulator

tau2 defaults to the LiteLLM-backed `user_simulator`. This fork also provides a
text-mode `codex_user_simulator` for running the simulated user through the
Codex CLI while keeping the existing half-duplex tau2 protocol unchanged.

The runtime environment must have the Codex CLI installed and authenticated
through the normal Codex auth mechanism, such as `CODEX_HOME` or the user's
default Codex login.

## CLI

```bash
tau2 run \
  --domain banking_knowledge \
  --agent llm_agent \
  --user codex_user_simulator \
  --user-llm gpt-5.5 \
  --user-llm-args '{"reasoning_effort":"xhigh"}'
```

## Programmatic Backend Switch

Callers that already use `user_simulator` can switch through `llm_args_user`:

```python
config = TextRunConfig(
    domain="banking_knowledge",
    agent="llm_agent",
    user="user_simulator",
    llm_user="gpt-5.5",
    llm_args_user={
        "backend": "codex",
        "reasoning_effort": "xhigh",
    },
)
```

`backend` is consumed by the tau2 builder and is not forwarded to LiteLLM.

## Codex Options

`CodexUserSimulator` accepts these `llm_args` keys:

- `codex_cmd`: Codex executable path. Defaults to `codex`.
- `codex_extra_args`: Extra CLI arguments. Defaults to an ephemeral, read-only
  execution with user config and project rules ignored.
- `reasoning_effort`: Mapped to `-c model_reasoning_effort="<value>"`.
- `timeout_seconds`: Per-turn subprocess timeout. Defaults to `300`.
- `codex_cwd`: Working directory for `codex exec`.
- `codex_env`: Environment variable overrides for the subprocess.

The simulator asks Codex to return a single JSON object:

```json
{
  "content": "text to send to the tested assistant, or null",
  "tool_calls": [],
  "stop": false
}
```

User-side tool calls are converted back into tau2 `ToolCall` objects with
`requestor="user"`, so the existing environment/tool routing remains unchanged.
