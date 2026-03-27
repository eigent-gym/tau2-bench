#!/usr/bin/env python3
"""
Run Statistics Analysis for Tau Voice Experiments.

This script analyzes experiment runs and generates:
- Successful task counts per experiment
- Retry counts per task per experiment (full table + summary)
- Wall clock time per task per experiment (full table + summary)

Usage:
    python -m experiments.tau_voice.exp.run_stats --data-dir data/tmp/qa_results/victor/experiment_2025_01_21

The script expects experiment folders with the pattern:
    {domain}_{complexity}_{provider}/tasks/task_{N}/sim_{uuid}/
"""

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from tau2.config import DEFAULT_TICK_DURATION_SECONDS
from tau2.utils.utils import DATA_DIR

# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class TaskTiming:
    """Timing information for a successful task."""

    wall_clock_seconds: Optional[float] = None
    audio_time_seconds: Optional[float] = None
    num_ticks: Optional[int] = None

    @property
    def ratio(self) -> Optional[float]:
        """Wall clock / audio time ratio (how many times slower than real-time)."""
        if (
            self.wall_clock_seconds
            and self.audio_time_seconds
            and self.audio_time_seconds > 0
        ):
            return self.wall_clock_seconds / self.audio_time_seconds
        return None

    @property
    def time_per_tick_ms(self) -> Optional[float]:
        """Average wall clock time per tick in milliseconds."""
        if self.wall_clock_seconds and self.num_ticks and self.num_ticks > 0:
            return (self.wall_clock_seconds / self.num_ticks) * 1000
        return None


@dataclass
class UserSimulatorTiming:
    """Timing breakdown for user simulator operations."""

    # LLM generation
    llm_gen_count: int = 0
    llm_gen_total_seconds: float = 0.0

    # TTS synthesis
    tts_count: int = 0
    tts_total_seconds: float = 0.0

    # Interruption checks
    interrupt_count: int = 0
    interrupt_total_seconds: float = 0.0

    # Backchannel checks
    backchannel_count: int = 0
    backchannel_total_seconds: float = 0.0

    @property
    def avg_llm_gen_seconds(self) -> Optional[float]:
        """Average time per user LLM generation call."""
        if self.llm_gen_count > 0:
            return self.llm_gen_total_seconds / self.llm_gen_count
        return None

    @property
    def avg_tts_seconds(self) -> Optional[float]:
        """Average time per TTS synthesis call."""
        if self.tts_count > 0:
            return self.tts_total_seconds / self.tts_count
        return None

    @property
    def avg_interrupt_seconds(self) -> Optional[float]:
        """Average time per interruption check LLM call."""
        if self.interrupt_count > 0:
            return self.interrupt_total_seconds / self.interrupt_count
        return None

    @property
    def avg_backchannel_seconds(self) -> Optional[float]:
        """Average time per backchannel check LLM call."""
        if self.backchannel_count > 0:
            return self.backchannel_total_seconds / self.backchannel_count
        return None

    @property
    def total_user_sim_seconds(self) -> float:
        """Total time spent in user simulator operations."""
        return (
            self.llm_gen_total_seconds
            + self.tts_total_seconds
            + self.interrupt_total_seconds
            + self.backchannel_total_seconds
        )


# =============================================================================
# Data Collection
# =============================================================================


def find_experiment_folders(data_dir: Path) -> List[Path]:
    """
    Find all experiment folders in the given directory.

    Experiment folders are expected to contain a 'tasks' subdirectory.

    Args:
        data_dir: Directory containing experiment folders

    Returns:
        List of paths to experiment folders
    """
    experiments = []
    for item in data_dir.iterdir():
        if item.is_dir() and (item / "tasks").exists():
            experiments.append(item)
    return sorted(experiments, key=lambda x: x.name)


def get_experiment_short_name(exp_name: str) -> str:
    """
    Convert experiment folder name to a shorter display name.

    Examples:
        retail_control_gemini -> ctrl_gem
        retail_regular_openai -> reg_oai
    """
    # Extract components
    parts = exp_name.split("_")
    if len(parts) < 3:
        return exp_name[:15]

    # Get complexity (control -> ctrl, regular -> reg)
    complexity = parts[1] if len(parts) > 1 else ""
    complexity_short = {
        "control": "ctrl",
        "regular": "reg",
    }.get(complexity, complexity[:4])

    # Get provider (last part)
    provider = parts[-1]
    provider_short = {
        "gemini": "gem",
        "openai": "oai",
        "nova": "nova",
        "xai": "xai",
    }.get(provider, provider[:4])

    return f"{complexity_short}_{provider_short}"


def extract_provider_from_experiment(exp_name: str) -> str:
    """Extract provider name from experiment folder name."""
    parts = exp_name.split("_")
    if len(parts) >= 1:
        return parts[-1]
    return "unknown"


def extract_complexity_from_experiment(exp_name: str) -> str:
    """Extract complexity (control/regular) from experiment folder name."""
    parts = exp_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def count_retries_per_task(exp_path: Path) -> Dict[int, int]:
    """
    Count the number of retries per task in an experiment.

    A retry is counted as (number of sim_* folders - 1) for each task.

    Args:
        exp_path: Path to experiment folder

    Returns:
        Dict mapping task_id -> retry_count
    """
    tasks_dir = exp_path / "tasks"
    if not tasks_dir.exists():
        return {}

    retries = {}
    for task_dir in tasks_dir.iterdir():
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue

        try:
            task_id = int(task_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        # Count sim_* folders
        sim_count = len(
            [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("sim_")]
        )

        retries[task_id] = max(0, sim_count - 1)

    return retries


def get_successful_sim_path(task_path: Path) -> Optional[Path]:
    """
    Find the successful simulation folder for a task.

    A successful simulation has a simulation.json file.

    Args:
        task_path: Path to task folder

    Returns:
        Path to successful sim folder, or None if not found
    """
    for sim_dir in task_path.iterdir():
        if not sim_dir.is_dir() or not sim_dir.name.startswith("sim_"):
            continue
        if (sim_dir / "simulation.json").exists():
            return sim_dir
    return None


def extract_wall_clock_from_simulation(sim_json_path: Path) -> Optional[float]:
    """
    Extract wall clock duration from a simulation.json file.

    Uses the 'duration' field which is set when the simulation completes.

    Args:
        sim_json_path: Path to simulation.json file

    Returns:
        Duration in seconds, or None if unable to extract
    """
    if not sim_json_path.exists():
        return None

    try:
        with open(sim_json_path, "r") as f:
            data = json.load(f)

        # The duration field is wall clock time
        duration = data.get("duration")
        if duration is not None:
            return float(duration)

        return None
    except Exception as e:
        logger.debug(f"Failed to extract duration from {sim_json_path}: {e}")
        return None


def extract_wall_clock_time(task_log_path: Path) -> Optional[float]:
    """
    Extract wall clock duration from a task.log file.

    DEPRECATED: Prefer extract_wall_clock_from_simulation() which uses structured data.
    This function parses timestamps from the log file as a fallback.

    Args:
        task_log_path: Path to task.log file

    Returns:
        Duration in seconds, or None if unable to parse
    """
    if not task_log_path.exists():
        return None

    timestamp_pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"
    fmt = "%Y-%m-%d %H:%M:%S.%f"

    first_ts = None
    last_ts = None

    try:
        with open(task_log_path, "r") as f:
            # Read first 100 lines to find first timestamp
            for i, line in enumerate(f):
                if i >= 100:
                    break
                match = re.match(timestamp_pattern, line)
                if match:
                    first_ts = match.group(1)
                    break

            # Read last 100 lines to find last timestamp
            f.seek(0, 2)  # Go to end
            file_size = f.tell()

            # Read chunks from the end
            chunk_size = 8192
            lines = []
            position = file_size

            while position > 0 and len(lines) < 100:
                position = max(0, position - chunk_size)
                f.seek(position)
                chunk = f.read(min(chunk_size, file_size - position))
                chunk_lines = chunk.splitlines()
                lines = chunk_lines + lines

            # Find last timestamp
            for line in reversed(lines[-100:]):
                match = re.match(timestamp_pattern, line)
                if match:
                    last_ts = match.group(1)
                    break

        if first_ts and last_ts:
            start = datetime.strptime(first_ts, fmt)
            end = datetime.strptime(last_ts, fmt)
            return (end - start).total_seconds()
    except Exception as e:
        logger.debug(f"Failed to parse timestamps from {task_log_path}: {e}")

    return None


def extract_user_simulator_timing(task_log_path: Path) -> Optional[UserSimulatorTiming]:
    """
    Extract user simulator timing breakdown from a task.log file.

    Parses timestamps to measure:
    - User LLM generation time
    - TTS synthesis time
    - Interruption check time
    - Backchannel check time

    Args:
        task_log_path: Path to task.log file

    Returns:
        UserSimulatorTiming with breakdown, or None if unable to parse
    """
    if not task_log_path.exists():
        return None

    timestamp_pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"
    fmt = "%Y-%m-%d %H:%M:%S.%f"

    timing = UserSimulatorTiming()

    try:
        with open(task_log_path, "r", errors="ignore") as f:
            lines = f.readlines()

        # Parse events with timestamps
        generate_starts = []
        generate_ends = []
        tts_starts = []
        tts_ends = []
        interrupt_starts = []
        interrupt_ends = []
        backchannel_starts = []
        backchannel_ends = []

        for line in lines:
            match = re.match(timestamp_pattern, line)
            if not match:
                continue

            try:
                ts = datetime.strptime(match.group(1), fmt)
            except ValueError:
                continue

            # User LLM generation
            if "Performing turn-taking action: action='generate_message'" in line:
                generate_starts.append(ts)
            elif (
                "USER SIMULATOR:" in line
                and "_generate_full_duplex_voice_message" in line
            ):
                generate_ends.append(ts)

            # TTS synthesis
            elif "ElevenLabs TTS: calling API" in line:
                tts_starts.append(ts)
            elif "ElevenLabs TTS: received" in line:
                tts_ends.append(ts)

            # Interruption checks
            elif "CHECKING INTERRUPTION" in line:
                interrupt_starts.append(ts)
            elif "Interruption decision:" in line:
                interrupt_ends.append(ts)

            # Backchannel checks
            elif "CHECKING BACKCHANNEL" in line:
                backchannel_starts.append(ts)
            elif "Backchannel decision:" in line:
                backchannel_ends.append(ts)

        # Calculate LLM generation times
        for start, end in zip(generate_starts, generate_ends):
            duration = (end - start).total_seconds()
            if 0 < duration < 120:  # Sanity check: < 2 minutes
                timing.llm_gen_count += 1
                timing.llm_gen_total_seconds += duration

        # Calculate TTS times
        for start, end in zip(tts_starts, tts_ends):
            duration = (end - start).total_seconds()
            if 0 < duration < 60:  # Sanity check: < 1 minute
                timing.tts_count += 1
                timing.tts_total_seconds += duration

        # Calculate interruption check times
        for start, end in zip(interrupt_starts, interrupt_ends):
            duration = (end - start).total_seconds()
            if 0 < duration < 30:  # Sanity check: < 30 seconds
                timing.interrupt_count += 1
                timing.interrupt_total_seconds += duration

        # Calculate backchannel check times
        for start, end in zip(backchannel_starts, backchannel_ends):
            duration = (end - start).total_seconds()
            if 0 < duration < 30:  # Sanity check: < 30 seconds
                timing.backchannel_count += 1
                timing.backchannel_total_seconds += duration

        return timing

    except Exception as e:
        logger.debug(f"Failed to parse user simulator timing from {task_log_path}: {e}")
        return None


def extract_num_ticks(sim_json_path: Path) -> Optional[int]:
    """
    Extract the number of ticks from a simulation.json file.

    Args:
        sim_json_path: Path to simulation.json file

    Returns:
        Number of ticks, or None if unable to extract
    """
    if not sim_json_path.exists():
        return None

    try:
        with open(sim_json_path, "r") as f:
            data = json.load(f)

        # The ticks are stored in the simulation data
        ticks = data.get("ticks")
        if ticks is not None:
            return len(ticks)

        # Alternative: check if there's a duration field we can use
        # with tick_duration to estimate ticks
        return None
    except Exception as e:
        logger.debug(f"Failed to extract num_ticks from {sim_json_path}: {e}")
        return None


@dataclass
class TickTimingStats:
    """Statistics about tick durations calculated from tick timestamps."""

    total_ticks: int = 0
    # Configured tick duration (from tick metadata, constant)
    configured_tick_duration_ms: Optional[float] = None
    # Wall clock statistics
    avg_tick_ms: float = 0.0
    median_tick_ms: float = 0.0
    max_tick_ms: float = 0.0
    min_tick_ms: float = 0.0
    total_wall_clock_ms: float = 0.0
    slow_tick_count: int = 0  # >500ms
    very_slow_tick_count: int = 0  # >2000ms
    user_gen_count: int = 0
    user_gen_total_cost: float = 0.0
    user_gen_total_prompt_tokens: int = 0
    user_gen_total_completion_tokens: int = 0
    # Timing metadata from TurnTakingAction (when available)
    llm_generation_total_seconds: float = 0.0
    tts_synthesis_total_seconds: float = 0.0
    interrupt_check_total_seconds: float = 0.0
    backchannel_check_total_seconds: float = 0.0
    # Counts for averaging
    llm_generation_count: int = 0
    tts_synthesis_count: int = 0
    interrupt_check_count: int = 0
    backchannel_check_count: int = 0
    # Cost/usage for interrupt and backchannel checks (when available)
    interrupt_check_total_cost: float = 0.0
    interrupt_check_total_prompt_tokens: int = 0
    interrupt_check_total_completion_tokens: int = 0
    backchannel_check_total_cost: float = 0.0
    backchannel_check_total_prompt_tokens: int = 0
    backchannel_check_total_completion_tokens: int = 0

    @property
    def slowdown_ratio(self) -> Optional[float]:
        """Ratio of wall clock time to simulated time (slowdown factor)."""
        if self.configured_tick_duration_ms and self.avg_tick_ms:
            return self.avg_tick_ms / self.configured_tick_duration_ms
        return None

    @property
    def total_simulated_ms(self) -> Optional[float]:
        """Total simulated time in milliseconds."""
        if self.configured_tick_duration_ms:
            return self.total_ticks * self.configured_tick_duration_ms
        return None


def extract_tick_timing_stats(sim_json_path: Path) -> Optional[TickTimingStats]:
    """
    Extract tick timing statistics from simulation.json using tick timestamps.

    This uses structured data instead of log parsing.

    Args:
        sim_json_path: Path to simulation.json file

    Returns:
        TickTimingStats with timing breakdown, or None if unable to extract
    """
    if not sim_json_path.exists():
        return None

    try:
        with open(sim_json_path, "r") as f:
            data = json.load(f)

        ticks = data.get("ticks", [])
        if not ticks:
            return None

        stats = TickTimingStats(total_ticks=len(ticks))

        # Extract configured tick duration from first tick (should be constant)
        first_tick_duration = ticks[0].get("tick_duration_seconds") if ticks else None
        if first_tick_duration is not None:
            stats.configured_tick_duration_ms = first_tick_duration * 1000

        # Calculate tick durations - prefer wall_clock_duration_seconds if available
        tick_durations_ms = []
        for tick in ticks:
            # Try new field first (wall_clock_duration_seconds)
            wc_duration = tick.get("wall_clock_duration_seconds")
            if wc_duration is not None:
                tick_durations_ms.append(wc_duration * 1000)

        # Fall back to calculating from timestamps if new field not available
        if not tick_durations_ms:
            for i in range(1, len(ticks)):
                prev_ts_str = ticks[i - 1].get("timestamp", "")
                curr_ts_str = ticks[i].get("timestamp", "")

                try:
                    prev_ts = datetime.fromisoformat(prev_ts_str.replace("Z", "+00:00"))
                    curr_ts = datetime.fromisoformat(curr_ts_str.replace("Z", "+00:00"))
                    duration_ms = (curr_ts - prev_ts).total_seconds() * 1000
                    tick_durations_ms.append(duration_ms)
                except (ValueError, TypeError):
                    continue

        if tick_durations_ms:
            stats.avg_tick_ms = sum(tick_durations_ms) / len(tick_durations_ms)
            stats.total_wall_clock_ms = sum(tick_durations_ms)
            stats.min_tick_ms = min(tick_durations_ms)
            sorted_durations = sorted(tick_durations_ms)
            stats.median_tick_ms = sorted_durations[len(sorted_durations) // 2]
            stats.max_tick_ms = max(tick_durations_ms)
            stats.slow_tick_count = sum(1 for d in tick_durations_ms if d > 500)
            stats.very_slow_tick_count = sum(1 for d in tick_durations_ms if d > 2000)

        # Count user generation events, costs, and timing from user_chunk
        for tick in ticks:
            user_chunk = tick.get("user_chunk")
            if not user_chunk:
                continue

            action = user_chunk.get("turn_taking_action", {}) or {}
            action_type = action.get("action")

            # Extract timing metadata from TurnTakingAction (if available)
            llm_gen_time = action.get("llm_generation_seconds")
            tts_time = action.get("tts_synthesis_seconds")
            int_time = action.get("interrupt_check_seconds")
            bc_time = action.get("backchannel_check_seconds")

            # Extract cost/usage for interrupt and backchannel checks (if available)
            int_cost = action.get("interrupt_check_cost")
            int_usage = action.get("interrupt_check_usage") or {}
            bc_cost = action.get("backchannel_check_cost")
            bc_usage = action.get("backchannel_check_usage") or {}

            if llm_gen_time is not None:
                stats.llm_generation_total_seconds += llm_gen_time
                stats.llm_generation_count += 1

            if tts_time is not None:
                stats.tts_synthesis_total_seconds += tts_time
                stats.tts_synthesis_count += 1

            if int_time is not None:
                stats.interrupt_check_total_seconds += int_time
                stats.interrupt_check_count += 1

            if int_cost is not None:
                stats.interrupt_check_total_cost += int_cost
                stats.interrupt_check_total_prompt_tokens += (
                    int_usage.get("prompt_tokens", 0) or 0
                )
                stats.interrupt_check_total_completion_tokens += (
                    int_usage.get("completion_tokens", 0) or 0
                )

            if bc_time is not None:
                stats.backchannel_check_total_seconds += bc_time
                stats.backchannel_check_count += 1

            if bc_cost is not None:
                stats.backchannel_check_total_cost += bc_cost
                stats.backchannel_check_total_prompt_tokens += (
                    bc_usage.get("prompt_tokens", 0) or 0
                )
                stats.backchannel_check_total_completion_tokens += (
                    bc_usage.get("completion_tokens", 0) or 0
                )

            # Count generate_message events and costs
            if action_type == "generate_message":
                stats.user_gen_count += 1
                stats.user_gen_total_cost += user_chunk.get("cost", 0) or 0
                usage = user_chunk.get("usage", {}) or {}
                stats.user_gen_total_prompt_tokens += usage.get("prompt_tokens", 0) or 0
                stats.user_gen_total_completion_tokens += (
                    usage.get("completion_tokens", 0) or 0
                )

        return stats

    except Exception as e:
        logger.debug(f"Failed to extract tick timing stats from {sim_json_path}: {e}")
        return None


def get_task_timing(
    sim_path: Path, tick_duration: float = DEFAULT_TICK_DURATION_SECONDS
) -> TaskTiming:
    """
    Get full timing information for a successful simulation.

    Prefers structured data from simulation.json over log parsing.

    Args:
        sim_path: Path to simulation folder
        tick_duration: Duration of each tick in seconds

    Returns:
        TaskTiming with wall clock, audio time, and num_ticks
    """
    timing = TaskTiming()
    sim_json = sim_path / "simulation.json"

    # Prefer wall clock time from simulation.json (structured data)
    timing.wall_clock_seconds = extract_wall_clock_from_simulation(sim_json)

    # Fall back to log parsing if simulation.json doesn't have duration
    if timing.wall_clock_seconds is None:
        task_log = sim_path / "task.log"
        timing.wall_clock_seconds = extract_wall_clock_time(task_log)

    # Extract num_ticks from simulation.json
    timing.num_ticks = extract_num_ticks(sim_json)

    # Calculate audio time
    if timing.num_ticks is not None:
        timing.audio_time_seconds = timing.num_ticks * tick_duration

    return timing


def get_timing_per_task(
    exp_path: Path, tick_duration: float = DEFAULT_TICK_DURATION_SECONDS
) -> Dict[int, Optional[TaskTiming]]:
    """
    Get timing information for each successful task in an experiment.

    Args:
        exp_path: Path to experiment folder
        tick_duration: Duration of each tick in seconds

    Returns:
        Dict mapping task_id -> TaskTiming (None if not successful)
    """
    tasks_dir = exp_path / "tasks"
    if not tasks_dir.exists():
        return {}

    timings = {}
    for task_dir in tasks_dir.iterdir():
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue

        try:
            task_id = int(task_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        # Find successful simulation
        sim_path = get_successful_sim_path(task_dir)
        if sim_path is None:
            timings[task_id] = None
            continue

        timings[task_id] = get_task_timing(sim_path, tick_duration)

    return timings


def count_successful_tasks(exp_path: Path) -> int:
    """
    Count the number of successful tasks in an experiment.

    A task is successful if it has a simulation.json file.
    """
    tasks_dir = exp_path / "tasks"
    if not tasks_dir.exists():
        return 0

    count = 0
    for task_dir in tasks_dir.iterdir():
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        if get_successful_sim_path(task_dir) is not None:
            count += 1

    return count


def count_total_tasks(exp_path: Path) -> int:
    """Count the total number of task folders in an experiment."""
    tasks_dir = exp_path / "tasks"
    if not tasks_dir.exists():
        return 0

    return len(
        [d for d in tasks_dir.iterdir() if d.is_dir() and d.name.startswith("task_")]
    )


def get_user_sim_timing_per_task(
    exp_path: Path,
) -> Dict[int, Optional[UserSimulatorTiming]]:
    """
    Get user simulator timing breakdown for each successful task.

    Args:
        exp_path: Path to experiment folder

    Returns:
        Dict mapping task_id -> UserSimulatorTiming (None if not available)
    """
    tasks_dir = exp_path / "tasks"
    if not tasks_dir.exists():
        return {}

    timings = {}
    for task_dir in tasks_dir.iterdir():
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue

        try:
            task_id = int(task_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        # Find successful simulation
        sim_path = get_successful_sim_path(task_dir)
        if sim_path is None:
            timings[task_id] = None
            continue

        # Extract user simulator timing from task.log
        task_log = sim_path / "task.log"
        timings[task_id] = extract_user_simulator_timing(task_log)

    return timings


# =============================================================================
# Table Generation
# =============================================================================


def format_time(seconds: Optional[float]) -> str:
    """Format seconds as MM:SS string."""
    if seconds is None:
        return "    ."
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:2d}:{secs:02d}"


def calc_stats(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Calculate mean and standard deviation of a list of values."""
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return mean, std


def build_retry_table(
    experiments: List[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build retry count tables.

    Returns:
        Tuple of (full_df, summary_df)
        - full_df: task x experiment matrix with retry counts
        - summary_df: provider summary with mean/std
    """
    # Collect retry data
    data = {}  # exp_name -> {task_id -> retry_count}
    all_tasks = set()

    for exp_path in experiments:
        exp_name = exp_path.name
        retries = count_retries_per_task(exp_path)
        data[exp_name] = retries
        all_tasks.update(retries.keys())

    if not all_tasks:
        return pd.DataFrame(), pd.DataFrame()

    all_tasks = sorted(all_tasks)
    exp_names = [exp.name for exp in experiments]
    short_names = [get_experiment_short_name(name) for name in exp_names]

    # Build full table
    rows = []
    for task_id in all_tasks:
        row = {"Task": task_id}
        for exp_name, short_name in zip(exp_names, short_names):
            retry_count = data.get(exp_name, {}).get(task_id, 0)
            row[short_name] = retry_count if retry_count > 0 else None
        rows.append(row)

    full_df = pd.DataFrame(rows)

    # Add marginal totals
    full_df["Total"] = full_df[short_names].sum(axis=1)

    # Add row totals
    totals_row = {"Task": "Total"}
    for short_name in short_names:
        totals_row[short_name] = full_df[short_name].sum()
    totals_row["Total"] = full_df["Total"].sum()
    full_df = pd.concat([full_df, pd.DataFrame([totals_row])], ignore_index=True)

    # Build summary by provider
    provider_data = defaultdict(lambda: {"control": [], "regular": []})

    for exp_path in experiments:
        exp_name = exp_path.name
        provider = extract_provider_from_experiment(exp_name)
        complexity = extract_complexity_from_experiment(exp_name)

        retries = data.get(exp_name, {})
        total_retries = sum(retries.values())

        if complexity in provider_data[provider]:
            provider_data[provider][complexity].append(total_retries)

    summary_rows = []
    for provider in sorted(provider_data.keys()):
        ctrl_values = provider_data[provider]["control"]
        reg_values = provider_data[provider]["regular"]

        ctrl_total = sum(ctrl_values) if ctrl_values else 0
        reg_total = sum(reg_values) if reg_values else 0

        summary_rows.append(
            {
                "Provider": provider.capitalize(),
                "Control": ctrl_total,
                "Regular": reg_total,
                "Total": ctrl_total + reg_total,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    return full_df, summary_df


def build_timing_table(
    experiments: List[Path],
    tick_duration: float = DEFAULT_TICK_DURATION_SECONDS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build timing tables with wall clock, audio time, and ratio.

    Returns:
        Tuple of (full_df, summary_df)
        - full_df: task x experiment matrix with "WC|Audio|Ratio" format
        - summary_df: provider summary with averages
    """
    # Collect timing data
    data = {}  # exp_name -> {task_id -> TaskTiming}
    all_tasks = set()

    for exp_path in experiments:
        exp_name = exp_path.name
        timings = get_timing_per_task(exp_path, tick_duration)
        data[exp_name] = timings
        all_tasks.update(timings.keys())

    if not all_tasks:
        return pd.DataFrame(), pd.DataFrame()

    all_tasks = sorted(all_tasks)
    exp_names = [exp.name for exp in experiments]
    short_names = [get_experiment_short_name(name) for name in exp_names]

    # Build full table with separate columns for each metric
    rows = []
    for task_id in all_tasks:
        row = {"Task": task_id}
        for exp_name, short_name in zip(exp_names, short_names):
            timing = data.get(exp_name, {}).get(task_id)
            if timing is not None:
                row[f"{short_name}_wc"] = timing.wall_clock_seconds
                row[f"{short_name}_audio"] = timing.audio_time_seconds
                row[f"{short_name}_ratio"] = timing.ratio
            else:
                row[f"{short_name}_wc"] = None
                row[f"{short_name}_audio"] = None
                row[f"{short_name}_ratio"] = None
        rows.append(row)

    full_df = pd.DataFrame(rows)

    # Calculate averages for each column type
    avg_row = {"Task": "Avg"}
    for short_name in short_names:
        for suffix in ["_wc", "_audio", "_ratio"]:
            col = f"{short_name}{suffix}"
            col_values = [v for v in full_df[col] if v is not None and not pd.isna(v)]
            avg_row[col] = sum(col_values) / len(col_values) if col_values else None

    full_df = pd.concat([full_df, pd.DataFrame([avg_row])], ignore_index=True)

    # Build summary by provider
    provider_data = defaultdict(
        lambda: {
            "control": {"wc": [], "audio": [], "ratio": []},
            "regular": {"wc": [], "audio": [], "ratio": []},
        }
    )

    for exp_path in experiments:
        exp_name = exp_path.name
        provider = extract_provider_from_experiment(exp_name)
        complexity = extract_complexity_from_experiment(exp_name)

        timings = data.get(exp_name, {})
        for timing in timings.values():
            if timing is None:
                continue
            if timing.wall_clock_seconds is not None:
                provider_data[provider][complexity]["wc"].append(
                    timing.wall_clock_seconds
                )
            if timing.audio_time_seconds is not None:
                provider_data[provider][complexity]["audio"].append(
                    timing.audio_time_seconds
                )
            if timing.ratio is not None:
                provider_data[provider][complexity]["ratio"].append(timing.ratio)

    summary_rows = []
    for provider in sorted(provider_data.keys()):
        ctrl = provider_data[provider]["control"]
        reg = provider_data[provider]["regular"]

        ctrl_wc_mean, ctrl_wc_std = calc_stats(ctrl["wc"])
        ctrl_audio_mean, _ = calc_stats(ctrl["audio"])
        ctrl_ratio_mean, ctrl_ratio_std = calc_stats(ctrl["ratio"])

        reg_wc_mean, reg_wc_std = calc_stats(reg["wc"])
        reg_audio_mean, _ = calc_stats(reg["audio"])
        reg_ratio_mean, reg_ratio_std = calc_stats(reg["ratio"])

        all_wc = ctrl["wc"] + reg["wc"]
        all_audio = ctrl["audio"] + reg["audio"]
        all_ratio = ctrl["ratio"] + reg["ratio"]

        overall_wc_mean, overall_wc_std = calc_stats(all_wc)
        overall_audio_mean, _ = calc_stats(all_audio)
        overall_ratio_mean, overall_ratio_std = calc_stats(all_ratio)

        summary_rows.append(
            {
                "Provider": provider.capitalize(),
                "Ctrl WC": ctrl_wc_mean,
                "Ctrl Audio": ctrl_audio_mean,
                "Ctrl Ratio": ctrl_ratio_mean,
                "Ctrl N": len(ctrl["wc"]),
                "Reg WC": reg_wc_mean,
                "Reg Audio": reg_audio_mean,
                "Reg Ratio": reg_ratio_mean,
                "Reg N": len(reg["wc"]),
                "All WC": overall_wc_mean,
                "All Audio": overall_audio_mean,
                "All Ratio": overall_ratio_mean,
                "Total N": len(all_wc),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    return full_df, summary_df


def build_normalized_metrics_table(
    experiments: List[Path],
    tick_duration: float = DEFAULT_TICK_DURATION_SECONDS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build normalized metrics tables (metrics independent of conversation length).

    Metrics:
    - Wall clock / Audio time ratio (slowdown factor)
    - Time per tick (ms)
    - Avg time per user LLM call (s)
    - Avg time per TTS call (s)
    - Avg time per interruption check (s)
    - Avg time per backchannel check (s)

    Returns:
        Tuple of (full_df, summary_df)
        - full_df: per-task metrics for each experiment
        - summary_df: summary by provider with mean ± std
    """
    # Collect data for each experiment
    timing_data = {}  # exp_name -> {task_id -> TaskTiming}
    user_sim_data = {}  # exp_name -> {task_id -> UserSimulatorTiming}
    all_tasks = set()

    for exp_path in experiments:
        exp_name = exp_path.name
        timing_data[exp_name] = get_timing_per_task(exp_path, tick_duration)
        user_sim_data[exp_name] = get_user_sim_timing_per_task(exp_path)
        all_tasks.update(timing_data[exp_name].keys())

    if not all_tasks:
        return pd.DataFrame(), pd.DataFrame()

    all_tasks = sorted(all_tasks)
    exp_names = [exp.name for exp in experiments]
    short_names = [get_experiment_short_name(name) for name in exp_names]

    # Build per-task table
    rows = []
    for task_id in all_tasks:
        row = {"Task": task_id}
        for exp_name, short_name in zip(exp_names, short_names):
            timing = timing_data.get(exp_name, {}).get(task_id)
            user_sim = user_sim_data.get(exp_name, {}).get(task_id)

            # Add ratio
            row[f"{short_name}_ratio"] = timing.ratio if timing else None

            # Add time per tick
            row[f"{short_name}_ms_per_tick"] = (
                timing.time_per_tick_ms if timing else None
            )

            # Add user simulator metrics
            if user_sim:
                row[f"{short_name}_llm_avg"] = user_sim.avg_llm_gen_seconds
                row[f"{short_name}_tts_avg"] = user_sim.avg_tts_seconds
                row[f"{short_name}_int_avg"] = user_sim.avg_interrupt_seconds
                row[f"{short_name}_bc_avg"] = user_sim.avg_backchannel_seconds
            else:
                row[f"{short_name}_llm_avg"] = None
                row[f"{short_name}_tts_avg"] = None
                row[f"{short_name}_int_avg"] = None
                row[f"{short_name}_bc_avg"] = None
        rows.append(row)

    full_df = pd.DataFrame(rows)

    # Build summary by provider
    provider_metrics = defaultdict(
        lambda: {
            "control": {
                "ratio": [],
                "ms_per_tick": [],
                "llm_avg": [],
                "tts_avg": [],
                "int_avg": [],
                "bc_avg": [],
            },
            "regular": {
                "ratio": [],
                "ms_per_tick": [],
                "llm_avg": [],
                "tts_avg": [],
                "int_avg": [],
                "bc_avg": [],
            },
        }
    )

    for exp_path in experiments:
        exp_name = exp_path.name
        provider = extract_provider_from_experiment(exp_name)
        complexity = extract_complexity_from_experiment(exp_name)

        for task_id in timing_data.get(exp_name, {}).keys():
            timing = timing_data[exp_name].get(task_id)
            user_sim = user_sim_data.get(exp_name, {}).get(task_id)

            if timing and timing.ratio is not None:
                provider_metrics[provider][complexity]["ratio"].append(timing.ratio)
            if timing and timing.time_per_tick_ms is not None:
                provider_metrics[provider][complexity]["ms_per_tick"].append(
                    timing.time_per_tick_ms
                )
            if user_sim:
                if user_sim.avg_llm_gen_seconds is not None:
                    provider_metrics[provider][complexity]["llm_avg"].append(
                        user_sim.avg_llm_gen_seconds
                    )
                if user_sim.avg_tts_seconds is not None:
                    provider_metrics[provider][complexity]["tts_avg"].append(
                        user_sim.avg_tts_seconds
                    )
                if user_sim.avg_interrupt_seconds is not None:
                    provider_metrics[provider][complexity]["int_avg"].append(
                        user_sim.avg_interrupt_seconds
                    )
                if user_sim.avg_backchannel_seconds is not None:
                    provider_metrics[provider][complexity]["bc_avg"].append(
                        user_sim.avg_backchannel_seconds
                    )

    summary_rows = []
    for provider in sorted(provider_metrics.keys()):
        ctrl = provider_metrics[provider]["control"]
        reg = provider_metrics[provider]["regular"]
        all_data = {
            metric: ctrl[metric] + reg[metric]
            for metric in [
                "ratio",
                "ms_per_tick",
                "llm_avg",
                "tts_avg",
                "int_avg",
                "bc_avg",
            ]
        }

        # Calculate stats for each metric
        row = {"Provider": provider.capitalize()}

        for prefix, data in [("Ctrl", ctrl), ("Reg", reg), ("All", all_data)]:
            ratio_mean, ratio_std = calc_stats(data["ratio"])
            ms_tick_mean, ms_tick_std = calc_stats(data["ms_per_tick"])
            llm_mean, llm_std = calc_stats(data["llm_avg"])
            tts_mean, tts_std = calc_stats(data["tts_avg"])
            int_mean, int_std = calc_stats(data["int_avg"])
            bc_mean, bc_std = calc_stats(data["bc_avg"])

            row[f"{prefix} Ratio"] = ratio_mean
            row[f"{prefix} Ratio Std"] = ratio_std
            row[f"{prefix} ms/tick"] = ms_tick_mean
            row[f"{prefix} ms/tick Std"] = ms_tick_std
            row[f"{prefix} LLM Avg"] = llm_mean
            row[f"{prefix} LLM Std"] = llm_std
            row[f"{prefix} TTS Avg"] = tts_mean
            row[f"{prefix} TTS Std"] = tts_std
            row[f"{prefix} Int Avg"] = int_mean
            row[f"{prefix} Int Std"] = int_std
            row[f"{prefix} BC Avg"] = bc_mean
            row[f"{prefix} BC Std"] = bc_std
            row[f"{prefix} N"] = len(data["ratio"])

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    return full_df, summary_df


def build_success_count_table(experiments: List[Path]) -> pd.DataFrame:
    """
    Build a table showing successful task counts per experiment.

    Returns:
        DataFrame with experiment name, successful count, total count, success rate
    """
    rows = []
    for exp_path in experiments:
        exp_name = exp_path.name
        short_name = get_experiment_short_name(exp_name)
        successful = count_successful_tasks(exp_path)
        total = count_total_tasks(exp_path)

        rows.append(
            {
                "Experiment": short_name,
                "Successful": successful,
                "Total": total,
                "Success Rate": (
                    f"{successful / total * 100:.1f}%" if total > 0 else "N/A"
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Output Formatting
# =============================================================================


def format_retry_table_markdown(df: pd.DataFrame) -> str:
    """Format retry table as markdown."""
    lines = []

    # Header
    cols = df.columns.tolist()
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append(
        "|" + "|".join("-" * max(5, len(str(c)) + 2) + ":" for c in cols) + "|"
    )

    # Rows
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if col == "Task":
                cells.append(f"{val}")
            elif val is None or (isinstance(val, float) and pd.isna(val)):
                cells.append(".")
            elif isinstance(val, float):
                cells.append(f"{int(val)}")
            else:
                cells.append(f"{val}")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def format_timing_table_markdown(
    df: pd.DataFrame,
    short_names: List[str],
) -> str:
    """
    Format timing table as markdown with WC/Audio/Ratio for each experiment.

    Each cell shows: "WC / Audio / Ratio" where WC and Audio are MM:SS and Ratio is Nx
    """
    lines = []

    # Build header
    header = ["Task"]
    for sn in short_names:
        header.append(f"{sn} (WC/Audio/Ratio)")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("-" * max(5, len(h) + 2) + ":" for h in header) + "|")

    # Rows
    for _, row in df.iterrows():
        cells = [str(row["Task"])]
        for sn in short_names:
            wc = row.get(f"{sn}_wc")
            audio = row.get(f"{sn}_audio")
            ratio = row.get(f"{sn}_ratio")

            if wc is None or (isinstance(wc, float) and pd.isna(wc)):
                cells.append(".")
            else:
                wc_str = format_time(wc).strip()
                audio_str = format_time(audio).strip() if audio else "."
                ratio_str = f"{ratio:.1f}x" if ratio else "."
                cells.append(f"{wc_str} / {audio_str} / {ratio_str}")

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def format_timing_summary_markdown(df: pd.DataFrame) -> str:
    """Format timing summary table as markdown."""
    lines = []

    # Header
    lines.append(
        "| Provider | Ctrl WC | Ctrl Audio | Ctrl Ratio | Ctrl N | Reg WC | Reg Audio | Reg Ratio | Reg N | All WC | All Audio | All Ratio | Total N |"
    )
    lines.append(
        "|----------|--------:|-----------:|-----------:|-------:|-------:|----------:|----------:|------:|-------:|----------:|----------:|--------:|"
    )

    # Rows
    for _, row in df.iterrows():
        cells = [
            f"**{row['Provider']}**",
            format_time(row["Ctrl WC"]).strip(),
            format_time(row["Ctrl Audio"]).strip(),
            f"{row['Ctrl Ratio']:.1f}x" if row["Ctrl Ratio"] else ".",
            str(int(row["Ctrl N"])),
            format_time(row["Reg WC"]).strip(),
            format_time(row["Reg Audio"]).strip(),
            f"{row['Reg Ratio']:.1f}x" if row["Reg Ratio"] else ".",
            str(int(row["Reg N"])),
            format_time(row["All WC"]).strip(),
            format_time(row["All Audio"]).strip(),
            f"{row['All Ratio']:.1f}x" if row["All Ratio"] else ".",
            str(int(row["Total N"])),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def format_normalized_metrics_markdown(df: pd.DataFrame) -> str:
    """
    Format normalized metrics summary table as markdown.

    Shows metrics that are independent of conversation length.
    """
    lines = []

    # Header - Focus on key normalized metrics
    lines.append(
        "| Provider | Ctrl Ratio | Ctrl ms/tick | Ctrl LLM | Ctrl TTS | Ctrl N | Reg Ratio | Reg ms/tick | Reg LLM | Reg TTS | Reg N | All Ratio | All ms/tick | All LLM | All TTS | All N |"
    )
    lines.append(
        "|----------|----------:|------------:|--------:|--------:|------:|----------:|-----------:|-------:|-------:|------:|----------:|-----------:|-------:|-------:|------:|"
    )

    # Rows
    for _, row in df.iterrows():

        def fmt_val(val, suffix=""):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "."
            return f"{val:.1f}{suffix}"

        def fmt_time(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "."
            return f"{val:.2f}s"

        def fmt_int(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "."
            return str(int(val))

        cells = [
            f"**{row['Provider']}**",
            # Control
            fmt_val(row.get("Ctrl Ratio"), "x"),
            fmt_val(row.get("Ctrl ms/tick"), ""),
            fmt_time(row.get("Ctrl LLM Avg")),
            fmt_time(row.get("Ctrl TTS Avg")),
            fmt_int(row.get("Ctrl N")),
            # Regular
            fmt_val(row.get("Reg Ratio"), "x"),
            fmt_val(row.get("Reg ms/tick"), ""),
            fmt_time(row.get("Reg LLM Avg")),
            fmt_time(row.get("Reg TTS Avg")),
            fmt_int(row.get("Reg N")),
            # All
            fmt_val(row.get("All Ratio"), "x"),
            fmt_val(row.get("All ms/tick"), ""),
            fmt_time(row.get("All LLM Avg")),
            fmt_time(row.get("All TTS Avg")),
            fmt_int(row.get("All N")),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def format_normalized_metrics_detail_markdown(df: pd.DataFrame) -> str:
    """
    Format detailed normalized metrics summary with std deviations.

    More compact format focusing on mean ± std for key metrics.
    """
    lines = []

    # Header - clarified column names
    lines.append(
        "| Provider | Complexity | N | Ratio WC/Audio (mean±std) | ms/tick (mean±std) | User LLM Gen (mean±std) | TTS Synth (mean±std) |"
    )
    lines.append(
        "|----------|------------|--:|-------------------------:|-----------------:|-----------------------:|--------------------:|"
    )

    # Rows - one per provider/complexity combo
    for _, row in df.iterrows():
        provider = row["Provider"]

        def fmt_mean_std(mean_key, std_key, suffix=""):
            mean = row.get(mean_key)
            std = row.get(std_key)
            if mean is None or (isinstance(mean, float) and pd.isna(mean)):
                return "."
            if std is None or (isinstance(std, float) and pd.isna(std)):
                return f"{mean:.2f}{suffix}"
            return f"{mean:.2f}±{std:.2f}{suffix}"

        # Control row
        cells_ctrl = [
            f"**{provider}**",
            "Control",
            str(int(row.get("Ctrl N", 0))),
            fmt_mean_std("Ctrl Ratio", "Ctrl Ratio Std", "x"),
            fmt_mean_std("Ctrl ms/tick", "Ctrl ms/tick Std", ""),
            fmt_mean_std("Ctrl LLM Avg", "Ctrl LLM Std", "s"),
            fmt_mean_std("Ctrl TTS Avg", "Ctrl TTS Std", "s"),
        ]
        lines.append("| " + " | ".join(cells_ctrl) + " |")

        # Regular row
        cells_reg = [
            f"**{provider}**",
            "Regular",
            str(int(row.get("Reg N", 0))),
            fmt_mean_std("Reg Ratio", "Reg Ratio Std", "x"),
            fmt_mean_std("Reg ms/tick", "Reg ms/tick Std", ""),
            fmt_mean_std("Reg LLM Avg", "Reg LLM Std", "s"),
            fmt_mean_std("Reg TTS Avg", "Reg TTS Std", "s"),
        ]
        lines.append("| " + " | ".join(cells_reg) + " |")

        # All row
        cells_all = [
            f"**{provider}**",
            "**Overall**",
            str(int(row.get("All N", 0))),
            fmt_mean_std("All Ratio", "All Ratio Std", "x"),
            fmt_mean_std("All ms/tick", "All ms/tick Std", ""),
            fmt_mean_std("All LLM Avg", "All LLM Std", "s"),
            fmt_mean_std("All TTS Avg", "All TTS Std", "s"),
        ]
        lines.append("| " + " | ".join(cells_all) + " |")

    return "\n".join(lines)


def format_user_llm_breakdown_markdown(df: pd.DataFrame) -> str:
    """
    Format User LLM call breakdown table showing all types of LLM calls.

    Shows:
    - User LLM Generation: Main response generation
    - Interruption Check: LLM call to decide if user should interrupt
    - Backchannel Check: LLM call to decide if user should backchannel
    """
    lines = []

    # Header
    lines.append(
        "| Provider | Complexity | N | User Gen (mean±std) | Interrupt Check (mean±std) | Backchannel Check (mean±std) |"
    )
    lines.append(
        "|----------|------------|--:|-------------------:|---------------------------:|-----------------------------:|"
    )

    # Rows
    for _, row in df.iterrows():
        provider = row["Provider"]

        def fmt_mean_std(mean_key, std_key, suffix=""):
            mean = row.get(mean_key)
            std = row.get(std_key)
            if mean is None or (isinstance(mean, float) and pd.isna(mean)):
                return "."
            if std is None or (isinstance(std, float) and pd.isna(std)):
                return f"{mean:.2f}{suffix}"
            return f"{mean:.2f}±{std:.2f}{suffix}"

        # Control row
        cells_ctrl = [
            f"**{provider}**",
            "Control",
            str(int(row.get("Ctrl N", 0))),
            fmt_mean_std("Ctrl LLM Avg", "Ctrl LLM Std", "s"),
            fmt_mean_std("Ctrl Int Avg", "Ctrl Int Std", "s"),
            fmt_mean_std("Ctrl BC Avg", "Ctrl BC Std", "s"),
        ]
        lines.append("| " + " | ".join(cells_ctrl) + " |")

        # Regular row
        cells_reg = [
            f"**{provider}**",
            "Regular",
            str(int(row.get("Reg N", 0))),
            fmt_mean_std("Reg LLM Avg", "Reg LLM Std", "s"),
            fmt_mean_std("Reg Int Avg", "Reg Int Std", "s"),
            fmt_mean_std("Reg BC Avg", "Reg BC Std", "s"),
        ]
        lines.append("| " + " | ".join(cells_reg) + " |")

        # All row
        cells_all = [
            f"**{provider}**",
            "**Overall**",
            str(int(row.get("All N", 0))),
            fmt_mean_std("All LLM Avg", "All LLM Std", "s"),
            fmt_mean_std("All Int Avg", "All Int Std", "s"),
            fmt_mean_std("All BC Avg", "All BC Std", "s"),
        ]
        lines.append("| " + " | ".join(cells_all) + " |")

    return "\n".join(lines)


# =============================================================================
# Main Analysis Function
# =============================================================================


def analyze_run_stats(
    data_dir: Path,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Main analysis function for run statistics.

    Args:
        data_dir: Directory containing experiment folders
        output_dir: Directory for output files (default: data_dir)
    """
    logger.info(f"Analyzing run statistics in {data_dir}...")

    # Find experiment folders
    experiments = find_experiment_folders(data_dir)

    if not experiments:
        logger.warning("No experiment folders found. Exiting.")
        return

    logger.info(f"Found {len(experiments)} experiment folders:")
    for exp in experiments:
        logger.info(f"  - {exp.name}")

    # Set up output directory (default: data_dir/stats)
    if output_dir is None:
        output_dir = data_dir / "stats"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get short names for experiments
    exp_names = [exp.name for exp in experiments]
    short_names = [get_experiment_short_name(name) for name in exp_names]

    # Build tables
    logger.info("Building retry count tables...")
    retry_full_df, retry_summary_df = build_retry_table(experiments)

    logger.info("Building timing tables (wall clock + audio time + ratio)...")
    timing_full_df, timing_summary_df = build_timing_table(experiments)

    logger.info("Building normalized metrics tables (length-independent)...")
    normalized_full_df, normalized_summary_df = build_normalized_metrics_table(
        experiments
    )

    logger.info("Building success count table...")
    success_df = build_success_count_table(experiments)

    # Build markdown content
    md_lines = []
    md_lines.append("# Run Statistics Report")
    md_lines.append("")
    md_lines.append(f"**Data Directory:** `{data_dir}`")
    md_lines.append("")

    md_lines.append("## Successful Task Counts")
    md_lines.append("")
    md_lines.append(success_df.to_markdown(index=False))
    md_lines.append("")

    md_lines.append("## Retry Count Table (Full)")
    md_lines.append("")
    md_lines.append(format_retry_table_markdown(retry_full_df))
    md_lines.append("")

    md_lines.append("## Retry Count Summary by Provider")
    md_lines.append("")
    md_lines.append(retry_summary_df.to_markdown(index=False))
    md_lines.append("")

    md_lines.append("## Timing Table (Full)")
    md_lines.append("")
    md_lines.append(
        "*Successful tasks only. Format: Wall Clock / Audio Time / Ratio (times slower than real-time).*"
    )
    md_lines.append("")
    md_lines.append(format_timing_table_markdown(timing_full_df, short_names))
    md_lines.append("")

    md_lines.append("## Timing Summary by Provider")
    md_lines.append("")
    md_lines.append(
        "*WC = Wall Clock, Audio = Simulated Audio Time, Ratio = WC/Audio (times slower than real-time).*"
    )
    md_lines.append("")
    md_lines.append(format_timing_summary_markdown(timing_summary_df))
    md_lines.append("")

    md_lines.append("## Normalized Metrics Summary (Length-Independent)")
    md_lines.append("")
    md_lines.append(
        "*These metrics are independent of conversation length, allowing fair comparison across tasks.*"
    )
    md_lines.append("")
    md_lines.append(
        "- **Ratio WC/Audio**: Wall clock time / Audio time (slowdown factor, lower is better)"
    )
    md_lines.append(
        "- **ms/tick**: Average wall clock milliseconds per 50ms audio tick (50ms = real-time)"
    )
    md_lines.append(
        "- **User LLM Gen**: Average time per user simulator LLM generation call (generates user response text)"
    )
    md_lines.append(
        "- **TTS Synth**: Average time per TTS synthesis call (ElevenLabs, converts text to speech)"
    )
    md_lines.append("")
    md_lines.append(format_normalized_metrics_detail_markdown(normalized_summary_df))
    md_lines.append("")

    md_lines.append("## User Simulator LLM Call Breakdown")
    md_lines.append("")
    md_lines.append(
        "*Breakdown of all LLM calls made by the user simulator (all use GPT-4.1):*"
    )
    md_lines.append("")
    md_lines.append("- **User Gen**: Main LLM call to generate user's response text")
    md_lines.append(
        "- **Interrupt Check**: LLM call to decide if user should interrupt the agent"
    )
    md_lines.append(
        "- **Backchannel Check**: LLM call to decide if user should say 'uh-huh', 'ok', etc."
    )
    md_lines.append("")
    md_lines.append(format_user_llm_breakdown_markdown(normalized_summary_df))
    md_lines.append("")

    md_content = "\n".join(md_lines)

    # Print to console
    print("\n" + "=" * 80)
    print("SUCCESSFUL TASK COUNTS")
    print("=" * 80)
    print(success_df.to_markdown(index=False))

    print("\n" + "=" * 80)
    print("RETRY COUNT TABLE (Full)")
    print("=" * 80)
    print(format_retry_table_markdown(retry_full_df))

    print("\n" + "=" * 80)
    print("RETRY COUNT SUMMARY BY PROVIDER")
    print("=" * 80)
    print(retry_summary_df.to_markdown(index=False))

    print("\n" + "=" * 80)
    print("TIMING TABLE (Full) - Wall Clock / Audio Time / Ratio")
    print("=" * 80)
    print(format_timing_table_markdown(timing_full_df, short_names))

    print("\n" + "=" * 80)
    print("TIMING SUMMARY BY PROVIDER")
    print("=" * 80)
    print(format_timing_summary_markdown(timing_summary_df))

    print("\n" + "=" * 80)
    print("NORMALIZED METRICS SUMMARY (Length-Independent)")
    print("=" * 80)
    print("Metrics independent of conversation length:")
    print("  - Ratio WC/Audio: Wall clock / Audio time (slowdown factor)")
    print("  - ms/tick: Avg wall clock ms per 50ms audio tick")
    print("  - User LLM Gen: Avg time per user simulator LLM generation call")
    print("  - TTS Synth: Avg time per TTS synthesis call (ElevenLabs)")
    print("")
    print(format_normalized_metrics_detail_markdown(normalized_summary_df))

    print("\n" + "=" * 80)
    print("USER SIMULATOR LLM CALL BREAKDOWN")
    print("=" * 80)
    print("All LLM calls made by the user simulator (all use GPT-4.1):")
    print("  - User Gen: Main LLM call to generate user's response text")
    print("  - Interrupt Check: LLM call to decide if user should interrupt")
    print("  - Backchannel Check: LLM call to decide if user should backchannel")
    print("")
    print(format_user_llm_breakdown_markdown(normalized_summary_df))

    # Save markdown report
    md_path = output_dir / "run_stats_report.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info(f"Saved: {md_path}")

    # Save CSVs
    success_csv = output_dir / "run_stats_success_counts.csv"
    success_df.to_csv(success_csv, index=False)
    logger.info(f"Saved: {success_csv}")

    retry_csv = output_dir / "run_stats_retries_full.csv"
    retry_full_df.to_csv(retry_csv, index=False)
    logger.info(f"Saved: {retry_csv}")

    retry_summary_csv = output_dir / "run_stats_retries_summary.csv"
    retry_summary_df.to_csv(retry_summary_csv, index=False)
    logger.info(f"Saved: {retry_summary_csv}")

    timing_csv = output_dir / "run_stats_timing_full.csv"
    timing_full_df.to_csv(timing_csv, index=False)
    logger.info(f"Saved: {timing_csv}")

    timing_summary_csv = output_dir / "run_stats_timing_summary.csv"
    timing_summary_df.to_csv(timing_summary_csv, index=False)
    logger.info(f"Saved: {timing_summary_csv}")

    normalized_csv = output_dir / "run_stats_normalized_full.csv"
    normalized_full_df.to_csv(normalized_csv, index=False)
    logger.info(f"Saved: {normalized_csv}")

    normalized_summary_csv = output_dir / "run_stats_normalized_summary.csv"
    normalized_summary_df.to_csv(normalized_summary_csv, index=False)
    logger.info(f"Saved: {normalized_summary_csv}")

    logger.info("Analysis complete!")


# =============================================================================
# CLI
# =============================================================================


def get_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze tau_voice experiment run statistics (retries, wall clock time, success counts)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing experiment folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output CSV files. Defaults to data_dir/stats.",
    )
    return parser


def main():
    parser = get_cli_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        # Check if path exists relative to current working directory first
        if not data_dir.exists():
            # Try relative to project root (DATA_DIR parent)
            project_root = DATA_DIR.parent
            data_dir = project_root / args.data_dir

    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute() and not output_dir.exists():
            project_root = DATA_DIR.parent
            output_dir = project_root / args.output_dir

    analyze_run_stats(
        data_dir=data_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
