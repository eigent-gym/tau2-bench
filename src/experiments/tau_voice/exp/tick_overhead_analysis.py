#!/usr/bin/env python3
"""
Tick-level overhead analysis for full-duplex voice simulations.

Analyzes wall-clock overhead per tick relative to the configured tick duration.

Usage:
    python -m experiments.tau_voice.exp.tick_overhead_analysis \
        --data-dir data/exp/tau_voice_new_lite/voice

    # Single experiment
    python -m experiments.tau_voice.exp.tick_overhead_analysis \
        --data-dir data/exp/tau_voice_new_lite/voice/airline_regular_gemini_gemini-live-2.5-flash-native-audio
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from experiments.tau_voice.exp.data_loader import load_simulation_results
from tau2.data_model.simulation import Results


def extract_tick_data(
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """Extract per-tick timing data from all simulations."""
    rows = []
    for params, sim_results in results:
        if sim_results.simulations is None:
            continue
        for sim in sim_results.simulations:
            if not sim.ticks:
                continue
            for tick in sim.ticks:
                if tick.wall_clock_duration_seconds is None:
                    continue
                tick_dur = tick.tick_duration_seconds or 0.2

                has_agent_chunk = (
                    tick.agent_chunk is not None and tick.agent_chunk.contains_speech
                )

                user_tt_action = None
                if (
                    tick.user_chunk is not None
                    and tick.user_chunk.turn_taking_action is not None
                ):
                    user_tt_action = tick.user_chunk.turn_taking_action.action

                rows.append(
                    {
                        "llm": params["llm"],
                        "domain": params["domain"],
                        "speech_complexity": params["speech_complexity"],
                        "simulation_id": sim.id,
                        "tick_id": tick.tick_id,
                        "tick_duration": tick_dur,
                        "wall_clock": tick.wall_clock_duration_seconds,
                        "overhead_ratio": tick.wall_clock_duration_seconds / tick_dur,
                        "agent_speaking": has_agent_chunk,
                        "user_turn_taking_action": user_tt_action,
                    }
                )

    return pd.DataFrame(rows)


def print_breakdown(df: pd.DataFrame) -> None:
    """Print overhead breakdown tables to stdout."""

    def _stats(series):
        return {
            "count": len(series),
            "mean": series.mean(),
            "median": series.median(),
            "p95": np.percentile(series, 95),
            "p99": np.percentile(series, 99),
        }

    # Overall
    print("\n=== Overall ===")
    s = _stats(df["overhead_ratio"])
    print(
        f"  Ticks: {s['count']:,}  |  Mean: {s['mean']:.2f}x  |  "
        f"Median: {s['median']:.2f}x  |  P95: {s['p95']:.2f}x  |  P99: {s['p99']:.2f}x"
    )

    # By LLM
    print("\n=== By LLM ===")
    for llm, group in sorted(df.groupby("llm")):
        s = _stats(group["overhead_ratio"])
        print(f"  {llm}")
        print(
            f"    Ticks: {s['count']:,}  |  Mean: {s['mean']:.2f}x  |  "
            f"Median: {s['median']:.2f}x  |  P95: {s['p95']:.2f}x  |  P99: {s['p99']:.2f}x"
        )

    # By user turn-taking action
    valid = df[df["user_turn_taking_action"].notna()]
    if not valid.empty:
        print("\n=== By User Turn-Taking Action ===")
        for action, group in sorted(valid.groupby("user_turn_taking_action")):
            s = _stats(group["overhead_ratio"])
            print(f"  {action}")
            print(
                f"    Ticks: {s['count']:,}  |  Mean: {s['mean']:.2f}x  |  "
                f"Median: {s['median']:.2f}x  |  P95: {s['p95']:.2f}x"
            )

    # By agent speaking
    print("\n=== By Agent Speaking ===")
    for label in ["agent_silent", "agent_speaking"]:
        group = df[df["agent_speaking"] == (label == "agent_speaking")]
        if group.empty:
            continue
        s = _stats(group["overhead_ratio"])
        print(f"  {label}")
        print(
            f"    Ticks: {s['count']:,}  |  Mean: {s['mean']:.2f}x  |  "
            f"Median: {s['median']:.2f}x  |  P95: {s['p95']:.2f}x"
        )

    # By user turn-taking action x agent speaking
    if not valid.empty:
        print("\n=== By User Action x Agent Speaking ===")
        valid = valid.copy()
        valid["agent_label"] = valid["agent_speaking"].map(
            {True: "agent_speaking", False: "agent_silent"}
        )
        for (action, agent_label), group in valid.groupby(
            ["user_turn_taking_action", "agent_label"]
        ):
            s = _stats(group["overhead_ratio"])
            print(f"  {action} / {agent_label}")
            print(
                f"    Ticks: {s['count']:,}  |  Mean: {s['mean']:.2f}x  |  "
                f"Median: {s['median']:.2f}x  |  P95: {s['p95']:.2f}x"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-tick wall-clock overhead in full-duplex voice simulations.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing simulation result folders.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific domains.",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Optional path to save raw tick data as CSV.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results = load_simulation_results(data_dir, args.domains)
    if not results:
        logger.error("No results found.")
        return

    df = extract_tick_data(results)
    if df.empty:
        logger.error("No tick data found.")
        return

    logger.info(
        f"Extracted {len(df):,} ticks from {df['simulation_id'].nunique()} simulations"
    )
    print_breakdown(df)

    if args.save_csv:
        path = Path(args.save_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved: {path}")


if __name__ == "__main__":
    main()
