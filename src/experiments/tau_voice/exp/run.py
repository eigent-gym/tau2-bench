#!/usr/bin/env python3
"""
Run a single audio-native voice experiment.

Usage:
    python run.py <domain> [options]

Examples:
    python run.py airline
    python run.py retail --speech-complexity regular --provider gemini
    python run.py airline --save-to my_experiment --auto-resume
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_AUDIO_NATIVE_PROVIDER,
    DEFAULT_INTEGRATION_DURATION_SECONDS,
    DEFAULT_INTERRUPTION_CHECK_INTERVAL_SECONDS,
    DEFAULT_LLM_USER,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS_SECONDS,
    DEFAULT_SEED,
    DEFAULT_SILENCE_ANNOTATION_THRESHOLD_SECONDS,
    DEFAULT_SPEECH_COMPLEXITY,
    DEFAULT_TICK_DURATION_SECONDS,
    DEFAULT_WAIT_TO_RESPOND_THRESHOLD_OTHER_SECONDS,
    DEFAULT_WAIT_TO_RESPOND_THRESHOLD_SELF_SECONDS,
    DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTED_SECONDS,
    DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTING_SECONDS,
)


@dataclass
class AudioNativeSettings:
    """Audio-native mode configuration."""

    tick_duration: float = DEFAULT_TICK_DURATION_SECONDS
    wait_to_respond_other: float = DEFAULT_WAIT_TO_RESPOND_THRESHOLD_OTHER_SECONDS
    wait_to_respond_self: float = DEFAULT_WAIT_TO_RESPOND_THRESHOLD_SELF_SECONDS
    yield_when_interrupted: float = DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTED_SECONDS
    yield_when_interrupting: float = DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTING_SECONDS
    interruption_check_interval: float = DEFAULT_INTERRUPTION_CHECK_INTERVAL_SECONDS
    integration_duration: float = DEFAULT_INTEGRATION_DURATION_SECONDS
    silence_annotation_threshold: float = DEFAULT_SILENCE_ANNOTATION_THRESHOLD_SECONDS


@dataclass
class RunConfig:
    """Configuration for a single run."""

    domain: str
    speech_complexity: str = DEFAULT_SPEECH_COMPLEXITY
    provider: str = DEFAULT_AUDIO_NATIVE_PROVIDER
    model: Optional[str] = None
    num_tasks: Optional[int] = None
    task_ids: Optional[list[str]] = None
    seed: int = DEFAULT_SEED
    max_steps_seconds: int = DEFAULT_MAX_STEPS_SECONDS
    max_errors: int = DEFAULT_MAX_ERRORS
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    save_to: Optional[str] = None
    auto_resume: bool = False
    user_llm: str = DEFAULT_LLM_USER
    cascaded_config: Optional[str] = None
    use_xml_prompt: bool = False
    review_mode: str = "full"
    hallucination_retries: int = 3
    audio_native: AudioNativeSettings = field(default_factory=AudioNativeSettings)

    def __post_init__(self):
        # Default model based on provider from config
        if self.model is None:
            self.model = DEFAULT_AUDIO_NATIVE_MODELS.get(self.provider)

    @property
    def effective_model(self) -> Optional[str]:
        """Return model if it's not 'None' string, else None."""
        if self.model and self.model.lower() != "none":
            return self.model
        return None

    def get_default_save_to(self) -> str:
        """Generate default save path."""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if self.effective_model:
            return f"{timestamp}_{self.domain}_{self.num_tasks}_voice_{self.speech_complexity}_{self.provider}_{self.effective_model}_{self.max_steps_seconds}s"
        else:
            return f"{timestamp}_{self.domain}_{self.num_tasks}_voice_{self.speech_complexity}_{self.provider}_{self.max_steps_seconds}s"


def build_command(config: RunConfig) -> list[str]:
    """Build the tau2 CLI command."""
    cmd = [
        "uv",
        "run",
        "tau2",
        "run",
        "--domain",
        config.domain,
        "--audio-native",
        "--audio-native-provider",
        config.provider,
    ]

    # Add model if specified
    if config.effective_model:
        cmd.extend(["--audio-native-model", config.effective_model])

    # Add cascaded config if specified (for livekit provider)
    if config.cascaded_config:
        cmd.extend(["--cascaded-config", config.cascaded_config])

    # Task selection
    if config.task_ids:
        cmd.extend(["--task-ids"] + config.task_ids)
    elif config.num_tasks:
        cmd.extend(["--num-tasks", str(config.num_tasks)])

    # Basic settings
    cmd.extend(
        [
            "--seed",
            str(config.seed),
            "--max-steps-seconds",
            str(config.max_steps_seconds),
            "--max-errors",
            str(config.max_errors),
            "--speech-complexity",
            config.speech_complexity,
            "--max-concurrency",
            str(config.max_concurrency),
            "--user-llm",
            config.user_llm,
        ]
    )

    # Audio-native settings
    audio = config.audio_native
    cmd.extend(
        [
            "--tick-duration",
            str(audio.tick_duration),
            "--wait-to-respond-other",
            str(audio.wait_to_respond_other),
            "--wait-to-respond-self",
            str(audio.wait_to_respond_self),
            "--yield-when-interrupted",
            str(audio.yield_when_interrupted),
            "--yield-when-interrupting",
            str(audio.yield_when_interrupting),
            "--interruption-check-interval",
            str(audio.interruption_check_interval),
            "--integration-duration",
            str(audio.integration_duration),
            "--silence-annotation-threshold",
            str(audio.silence_annotation_threshold),
        ]
    )

    # Always enable verbose logs
    cmd.append("--verbose-logs")

    # Use latest LLM log mode to save space
    cmd.extend(["--llm-log-mode", "latest"])

    # Always enable auto-review
    cmd.append("--auto-review")
    cmd.extend(["--review-mode", config.review_mode])

    # Hallucination retries
    cmd.extend(["--hallucination-retries", str(config.hallucination_retries)])

    # Prompt format
    if config.use_xml_prompt:
        cmd.append("--xml-prompt")
    else:
        cmd.append("--no-xml-prompt")

    # Save path
    save_to = config.save_to or config.get_default_save_to()
    cmd.extend(["--save-to", save_to])

    # Auto-resume
    if config.auto_resume:
        cmd.append("--auto-resume")

    return cmd


def run(config: RunConfig) -> int:
    """Execute the tau2 run command."""
    cmd = build_command(config)

    print(f"Running command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run a single audio-native voice experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "domain", type=str, help="Domain to run (e.g., airline, retail)"
    )

    parser.add_argument(
        "--speech-complexity",
        type=str,
        default=DEFAULT_SPEECH_COMPLEXITY,
        choices=[
            "control",
            "regular",
            "control_audio",
            "control_accents",
            "control_behavior",
            "control_audio_accents",
            "control_audio_behavior",
            "control_accents_behavior",
        ],
        help=f"Speech complexity level. Default is '{DEFAULT_SPEECH_COMPLEXITY}'.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=DEFAULT_AUDIO_NATIVE_PROVIDER,
        choices=["openai", "gemini", "xai", "nova", "qwen", "deepgram", "livekit"],
        help=f"Audio native provider. Default is '{DEFAULT_AUDIO_NATIVE_PROVIDER}'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Audio native model (None uses provider default)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to run",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific task IDs to run (overrides --num-tasks)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default is {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--max-steps-seconds",
        type=int,
        default=DEFAULT_MAX_STEPS_SECONDS,
        help=f"Maximum conversation duration in seconds. Default is {DEFAULT_MAX_STEPS_SECONDS}.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"Maximum concurrent simulations. Default is {DEFAULT_MAX_CONCURRENCY}.",
    )
    parser.add_argument(
        "--save-to",
        type=str,
        default=None,
        help="Custom save path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically resume from existing save file",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help=f"Maximum errors allowed per simulation. Default is {DEFAULT_MAX_ERRORS}.",
    )
    parser.add_argument(
        "--user-llm",
        type=str,
        default=DEFAULT_LLM_USER,
        help=f"LLM to use for user simulator. Default is {DEFAULT_LLM_USER}.",
    )
    parser.add_argument(
        "--cascaded-config",
        type=str,
        default=None,
        help="Cascaded config preset name for livekit provider. "
        "Available presets: 'default', 'openai-thinking', 'openai-thinking-high'.",
    )
    parser.add_argument(
        "--review-mode",
        type=str,
        choices=["full", "user"],
        default="full",
        help="Review mode when auto-review is enabled: 'full' (agent+user errors, default) or 'user' (user simulator only).",
    )
    parser.add_argument(
        "--hallucination-retries",
        type=int,
        default=3,
        help="Max retries when a user simulator hallucination is detected (full-duplex only). Set to 0 to disable. Default is 3.",
    )

    # Prompt format
    prompt_format_group = parser.add_mutually_exclusive_group()
    prompt_format_group.add_argument(
        "--xml-prompt",
        action="store_true",
        default=False,
        help="Use XML tags in system prompt (overrides auto-detection).",
    )
    prompt_format_group.add_argument(
        "--no-xml-prompt",
        action="store_true",
        default=False,
        help="Use plain text system prompt without XML tags (overrides auto-detection).",
    )

    args = parser.parse_args()

    # Determine use_xml_prompt: defaults to False (plain text)
    use_xml_prompt = False
    if args.xml_prompt:
        use_xml_prompt = True

    config = RunConfig(
        domain=args.domain,
        speech_complexity=args.speech_complexity,
        provider=args.provider,
        model=args.model,
        num_tasks=args.num_tasks,
        task_ids=args.task_ids,
        seed=args.seed,
        max_steps_seconds=args.max_steps_seconds,
        max_errors=args.max_errors,
        max_concurrency=args.max_concurrency,
        save_to=args.save_to,
        auto_resume=args.auto_resume,
        user_llm=args.user_llm,
        cascaded_config=args.cascaded_config,
        use_xml_prompt=use_xml_prompt,
        review_mode=args.review_mode,
        hallucination_retries=args.hallucination_retries,
    )

    return run(config)


if __name__ == "__main__":
    sys.exit(main())
