#!/usr/bin/env python3
"""
Hyperparameter sweep for audio-native voice experiments.

Usage:
    # New run
    python hyperparam_run.py --domains airline,retail --complexities control,regular --providers openai,gemini,xai

    # Resume existing run (exact same config)
    python hyperparam_run.py --resume data/exp/2026_01_08_14_18_41

    # Amend existing run with more tasks or new providers/domains
    python hyperparam_run.py --amend data/exp/2026_01_08_14_18_41 --num-tasks 30
    python hyperparam_run.py --amend data/exp/2026_01_08_14_18_41 --add-providers gemini,xai
    python hyperparam_run.py --amend data/exp/2026_01_08_14_18_41 --add-domains retail --add-providers xai

Examples:
    python hyperparam_run.py --domains airline --complexities regular --providers openai
    python hyperparam_run.py --domains airline,retail --complexities control,regular --providers openai,gemini,xai --num-tasks 20
    python hyperparam_run.py --domains retail --complexities control --providers xai --task-ids 0 1 2 3
    python hyperparam_run.py --domains retail --complexities control --providers xai --save-to my_experiment
    python hyperparam_run.py --resume data/exp/2026_01_08_14_18_41
    python hyperparam_run.py --amend data/exp/2026_01_08_14_18_41 --num-tasks 30 --add-providers gemini
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

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
class HyperparamConfig:
    """Configuration for a hyperparameter sweep."""

    domains: list[str]
    speech_complexities: list[str]
    provider_models: list[str]  # Format: "provider" or "provider:model"
    num_tasks: Optional[int] = None  # Number of tasks (ignored if task_ids is set)
    task_ids: Optional[list[str]] = (
        None  # Specific task IDs to run (overrides num_tasks)
    )
    seed: int = DEFAULT_SEED
    max_steps_seconds: int = DEFAULT_MAX_STEPS_SECONDS
    max_errors: int = DEFAULT_MAX_ERRORS
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    use_xml_prompt: bool = False  # True = XML, False = plain text (default)
    user_llm: str = DEFAULT_LLM_USER  # LLM for user simulator
    cascaded_config: Optional[str] = None  # Cascaded config preset for livekit
    review_mode: str = "full"  # Review mode: "full" or "user"
    hallucination_retries: int = 3  # Max retries on user simulator hallucination
    audio_native: AudioNativeSettings = field(default_factory=AudioNativeSettings)

    # Set during initialization
    timestamp: str = ""
    exp_dir: Path = field(default_factory=Path)
    resume_mode: bool = False

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # Normalize provider_models to always include the model
        self.provider_models = [
            f"{p}:{DEFAULT_AUDIO_NATIVE_MODELS[p]}" if ":" not in p else p
            for p in self.provider_models
        ]

    @property
    def total_combinations(self) -> int:
        return (
            len(self.domains)
            * len(self.speech_complexities)
            * len(self.provider_models)
        )

    def to_yaml_dict(self) -> dict:
        """Convert to YAML-serializable dict."""
        settings = {
            "seed": self.seed,
            "max_steps_seconds": self.max_steps_seconds,
            "max_errors": self.max_errors,
            "max_concurrency": self.max_concurrency,
        }
        # Include task selection (either task_ids or num_tasks)
        if self.task_ids:
            settings["task_ids"] = self.task_ids
        else:
            settings["num_tasks"] = self.num_tasks
        # Include use_xml_prompt setting
        settings["use_xml_prompt"] = self.use_xml_prompt
        # Include user_llm setting
        settings["user_llm"] = self.user_llm
        # Include cascaded_config if set
        if self.cascaded_config is not None:
            settings["cascaded_config"] = self.cascaded_config
        # Include review settings
        settings["review_mode"] = self.review_mode
        settings["hallucination_retries"] = self.hallucination_retries

        return {
            "experiment": {
                "timestamp": self.timestamp,
                "total_combinations": self.total_combinations,
            },
            "hyperparameters": {
                "domains": self.domains,
                "speech_complexities": self.speech_complexities,
                "provider_models": self.provider_models,
            },
            "settings": settings,
            "audio_native_settings": {
                "tick_duration": self.audio_native.tick_duration,
                "wait_to_respond_other": self.audio_native.wait_to_respond_other,
                "wait_to_respond_self": self.audio_native.wait_to_respond_self,
                "yield_when_interrupted": self.audio_native.yield_when_interrupted,
                "yield_when_interrupting": self.audio_native.yield_when_interrupting,
                "interruption_check_interval": self.audio_native.interruption_check_interval,
                "integration_duration": self.audio_native.integration_duration,
                "silence_annotation_threshold": self.audio_native.silence_annotation_threshold,
            },
        }

    @classmethod
    def from_yaml_file(cls, yaml_path: Path) -> "HyperparamConfig":
        """Load config from a hyperparams.yaml file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        audio_native = AudioNativeSettings()
        if "audio_native_settings" in data:
            ans = data["audio_native_settings"]
            audio_native = AudioNativeSettings(
                tick_duration=ans.get("tick_duration", DEFAULT_TICK_DURATION_SECONDS),
                wait_to_respond_other=ans.get(
                    "wait_to_respond_other",
                    DEFAULT_WAIT_TO_RESPOND_THRESHOLD_OTHER_SECONDS,
                ),
                wait_to_respond_self=ans.get(
                    "wait_to_respond_self",
                    DEFAULT_WAIT_TO_RESPOND_THRESHOLD_SELF_SECONDS,
                ),
                yield_when_interrupted=ans.get(
                    "yield_when_interrupted",
                    DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTED_SECONDS,
                ),
                yield_when_interrupting=ans.get(
                    "yield_when_interrupting",
                    DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTING_SECONDS,
                ),
                interruption_check_interval=ans.get(
                    "interruption_check_interval",
                    DEFAULT_INTERRUPTION_CHECK_INTERVAL_SECONDS,
                ),
                integration_duration=ans.get(
                    "integration_duration", DEFAULT_INTEGRATION_DURATION_SECONDS
                ),
                silence_annotation_threshold=ans.get(
                    "silence_annotation_threshold",
                    DEFAULT_SILENCE_ANNOTATION_THRESHOLD_SECONDS,
                ),
            )

        settings = data.get("settings", {})
        hyperparams = data.get("hyperparameters", {})
        experiment = data.get("experiment", {})

        config = cls(
            domains=hyperparams.get("domains", []),
            speech_complexities=hyperparams.get("speech_complexities", []),
            provider_models=hyperparams.get("provider_models", []),
            num_tasks=settings.get("num_tasks"),
            task_ids=settings.get("task_ids"),
            seed=settings.get("seed", DEFAULT_SEED),
            max_steps_seconds=settings.get(
                "max_steps_seconds", DEFAULT_MAX_STEPS_SECONDS
            ),
            max_errors=settings.get("max_errors", DEFAULT_MAX_ERRORS),
            max_concurrency=settings.get("max_concurrency", DEFAULT_MAX_CONCURRENCY),
            use_xml_prompt=settings.get("use_xml_prompt"),
            user_llm=settings.get("user_llm", DEFAULT_LLM_USER),
            cascaded_config=settings.get("cascaded_config"),
            review_mode=settings.get("review_mode", "full"),
            hallucination_retries=settings.get("hallucination_retries", 3),
            audio_native=audio_native,
            timestamp=experiment.get("timestamp", ""),
            resume_mode=True,
        )
        config.exp_dir = yaml_path.parent
        return config

    def save_yaml(self):
        """Save config to hyperparams.yaml in exp_dir."""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = self.exp_dir / "hyperparams.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_yaml_dict(), f, default_flow_style=False, sort_keys=False)
        print(f"Saved hyperparams to: {yaml_path}")


def parse_provider_model(provider_model: str) -> tuple[str, Optional[str]]:
    """Parse 'provider:model' or 'provider' format."""
    if ":" in provider_model:
        provider, model = provider_model.split(":", 1)
        return provider, model
    return provider_model, None


def get_run_name(domain: str, complexity: str, provider: str, model: str) -> str:
    """Generate run name for a combination."""
    return f"{domain}_{complexity}_{provider}_{model}"


def build_command(
    config: HyperparamConfig,
    domain: str,
    complexity: str,
    provider: str,
    model: Optional[str],
    save_to: str,
) -> list[str]:
    """Build the tau2 CLI command."""
    cmd = [
        "uv",
        "run",
        "tau2",
        "run",
        "--domain",
        domain,
        "--audio-native",
        "--audio-native-provider",
        provider,
    ]

    # Add model if specified
    if model:
        cmd.extend(["--audio-native-model", model])

    # Add cascaded config if specified (for livekit provider)
    if config.cascaded_config:
        cmd.extend(["--cascaded-config", config.cascaded_config])

    # Task selection: either specific task IDs or num_tasks
    if config.task_ids:
        cmd.extend(["--task-ids"] + config.task_ids)
    elif config.num_tasks is not None:
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
            complexity,
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

    # Use latest LLM log mode to save space during hyperparameter sweeps
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
    cmd.extend(["--save-to", save_to])

    # Auto-resume if in resume mode
    if config.resume_mode:
        cmd.append("--auto-resume")

    return cmd


def run_sweep(config: HyperparamConfig) -> int:
    """Run all combinations in the sweep."""
    total = config.total_combinations
    current = 0

    print("=" * 40)
    print("Hyperparameter sweep")
    print(f"Experiment directory: {config.exp_dir}")
    print(f"Domains: {config.domains}")
    print(f"Speech complexities: {config.speech_complexities}")
    print(f"Provider:Model combinations: {config.provider_models}")
    print(f"User simulator LLM: {config.user_llm}")
    if config.task_ids:
        print(f"Task IDs: {config.task_ids}")
    else:
        print(f"Num tasks: {config.num_tasks}")
    print(f"Total combinations: {total}")
    if config.resume_mode:
        print("Mode: RESUME (auto-resume enabled)")
    print("=" * 40)
    print()

    for domain in config.domains:
        for complexity in config.speech_complexities:
            for provider_model in config.provider_models:
                current += 1
                provider, model = parse_provider_model(provider_model)
                if model is None:
                    model = DEFAULT_AUDIO_NATIVE_MODELS[provider]
                run_name = get_run_name(domain, complexity, provider, model)

                # Save path - use absolute path from exp_dir
                save_to = str(config.exp_dir / run_name)

                print("-" * 40)
                print(f"Running combination {current}/{total}")
                print(f"  Domain: {domain}")
                print(f"  Speech complexity: {complexity}")
                print(f"  Provider: {provider}")
                print(f"  Model: {model}")
                print(f"  Save to: {config.exp_dir / run_name}")
                if config.resume_mode:
                    print("  Auto-resume: enabled")
                print("-" * 40)

                cmd = build_command(
                    config, domain, complexity, provider, model, save_to
                )
                print(f"Command: {' '.join(cmd)}")
                print()

                result = subprocess.run(cmd)

                print()
                print(f"Completed combination {current}/{total}")
                if result.returncode != 0:
                    print(f"  WARNING: Exit code {result.returncode}")
                print()

    print("=" * 40)
    print(f"All {total} combinations completed!")
    print(f"Results saved to: {config.exp_dir}")
    print("=" * 40)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for audio-native voice experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # New run
  python hyperparam_run.py --domains airline,retail --complexities regular --providers openai

  # Resume existing run (exact same config)
  python hyperparam_run.py --resume data/exp/2026_01_08_14_18_41

  # Amend existing run with more tasks
  python hyperparam_run.py --amend data/exp/2026_01_08_14_18_41 --num-tasks 30

  # Amend existing run with additional providers
  python hyperparam_run.py --amend data/exp/2026_01_08_14_18_41 --add-providers gemini,xai

  # Amend existing run with additional domains and providers
  python hyperparam_run.py --amend data/exp/2026_01_08_14_18_41 --add-domains retail --add-providers xai
        """,
    )

    # Resume/amend mode (mutually exclusive)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        type=str,
        metavar="DIR",
        help="Resume from existing experiment directory (loads hyperparams.yaml)",
    )
    resume_group.add_argument(
        "--amend",
        type=str,
        metavar="DIR",
        help="Amend existing experiment with additional tasks/providers/domains/complexities",
    )

    # Amend-specific options (additive)
    parser.add_argument(
        "--add-providers",
        type=str,
        metavar="PROVIDERS",
        help="Additional provider:model pairs to add (comma-separated, used with --amend)",
    )
    parser.add_argument(
        "--add-domains",
        type=str,
        metavar="DOMAINS",
        help="Additional domains to add (comma-separated, used with --amend)",
    )
    parser.add_argument(
        "--add-complexities",
        type=str,
        metavar="COMPLEXITIES",
        help="Additional speech complexities to add (comma-separated, used with --amend)",
    )

    # New run options
    parser.add_argument(
        "--domains",
        type=str,
        help="Comma-separated list of domains (e.g., airline,retail)",
    )
    parser.add_argument(
        "--complexities",
        type=str,
        default=DEFAULT_SPEECH_COMPLEXITY,
        help=f"Comma-separated speech complexities (e.g., control,regular). Default is '{DEFAULT_SPEECH_COMPLEXITY}'.",
    )
    # Build default provider:model string from config
    default_provider = DEFAULT_AUDIO_NATIVE_PROVIDER
    default_model = DEFAULT_AUDIO_NATIVE_MODELS.get(default_provider, "")
    default_provider_model = (
        f"{default_provider}:{default_model}" if default_model else default_provider
    )

    parser.add_argument(
        "--providers",
        type=str,
        default=default_provider_model,
        help=f"Comma-separated provider:model pairs (e.g., {default_provider_model},gemini,xai)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to run (ignored if --task-ids is specified)",
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
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help=f"Maximum errors allowed per simulation. Default is {DEFAULT_MAX_ERRORS}.",
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
        help="Directory path for results (relative to cwd or absolute). Defaults to data/exp/<timestamp>/.",
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

    # Get project root (4 levels up from this script)
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent.parent.parent

    # Validate amend-specific options
    if not args.amend and (
        args.add_providers or args.add_domains or args.add_complexities
    ):
        parser.error(
            "--add-providers, --add-domains, and --add-complexities require --amend"
        )

    if args.resume:
        # Resume mode (exact same config)
        resume_dir = Path(args.resume)
        if not resume_dir.is_absolute():
            resume_dir = project_root / resume_dir

        hyperparams_file = resume_dir / "hyperparams.yaml"
        if not hyperparams_file.exists():
            print(f"Error: hyperparams.yaml not found in {resume_dir}")
            return 1

        print("=" * 40)
        print(f"Resuming experiment from: {resume_dir}")
        print("=" * 40)

        config = HyperparamConfig.from_yaml_file(hyperparams_file)

    elif args.amend:
        # Amend mode (add tasks/providers/domains/complexities to existing experiment)
        amend_dir = Path(args.amend)
        if not amend_dir.is_absolute():
            amend_dir = project_root / amend_dir

        hyperparams_file = amend_dir / "hyperparams.yaml"
        if not hyperparams_file.exists():
            print(f"Error: hyperparams.yaml not found in {amend_dir}")
            return 1

        config = HyperparamConfig.from_yaml_file(hyperparams_file)

        # Track what's being amended
        amendments = []

        # Add new providers (merge, no duplicates)
        if args.add_providers:
            new_providers = [p.strip() for p in args.add_providers.split(",")]
            existing = set(config.provider_models)
            to_add = [p for p in new_providers if p not in existing]
            if to_add:
                config.provider_models = config.provider_models + to_add
                amendments.append(f"providers: +{to_add}")

        # Add new domains (merge, no duplicates)
        if args.add_domains:
            new_domains = [d.strip() for d in args.add_domains.split(",")]
            existing = set(config.domains)
            to_add = [d for d in new_domains if d not in existing]
            if to_add:
                config.domains = config.domains + to_add
                amendments.append(f"domains: +{to_add}")

        # Add new complexities (merge, no duplicates)
        if args.add_complexities:
            new_complexities = [c.strip() for c in args.add_complexities.split(",")]
            existing = set(config.speech_complexities)
            to_add = [c for c in new_complexities if c not in existing]
            if to_add:
                config.speech_complexities = config.speech_complexities + to_add
                amendments.append(f"complexities: +{to_add}")

        # Update num_tasks if specified (override)
        if args.num_tasks is not None:
            old_num_tasks = config.num_tasks
            if args.num_tasks != old_num_tasks:
                config.num_tasks = args.num_tasks
                amendments.append(f"num_tasks: {old_num_tasks} -> {args.num_tasks}")

        if not amendments:
            print(
                "Warning: No amendments specified. Use --add-providers, --add-domains, --add-complexities, or --num-tasks"
            )
            print("Running with existing config (same as --resume)")
        else:
            print("=" * 40)
            print(f"Amending experiment: {amend_dir}")
            print("Amendments:")
            for amendment in amendments:
                print(f"  - {amendment}")
            print("=" * 40)

            # Save updated hyperparams
            config.save_yaml()

    else:
        # New run mode
        if not args.domains:
            parser.error("--domains is required for new runs (or use --resume/--amend)")

        domains = [d.strip() for d in args.domains.split(",")]
        complexities = [c.strip() for c in args.complexities.split(",")]
        providers = [p.strip() for p in args.providers.split(",")]

        # Determine use_xml_prompt: defaults to False (plain text)
        use_xml_prompt = False
        if args.xml_prompt:
            use_xml_prompt = True

        config = HyperparamConfig(
            domains=domains,
            speech_complexities=complexities,
            provider_models=providers,
            num_tasks=args.num_tasks if not args.task_ids else None,
            task_ids=args.task_ids,
            seed=args.seed,
            max_steps_seconds=args.max_steps_seconds,
            max_errors=args.max_errors,
            max_concurrency=args.max_concurrency,
            use_xml_prompt=use_xml_prompt,
            user_llm=args.user_llm,
            cascaded_config=args.cascaded_config,
            review_mode=args.review_mode,
            hallucination_retries=args.hallucination_retries,
        )
        # Use custom save_to (relative to cwd) or default to data/exp/timestamp
        if args.save_to:
            # Resolve relative to current working directory
            config.exp_dir = Path(args.save_to).resolve()
        else:
            config.exp_dir = project_root / "data" / "exp" / config.timestamp

        # Save hyperparams
        config.save_yaml()

    return run_sweep(config)


if __name__ == "__main__":
    sys.exit(main())
