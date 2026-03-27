#!/usr/bin/env python3
"""
Performance Analysis for Tau Voice Experiments.

This script analyzes results from tau_voice hyperparameter sweeps, generating
pass^k metrics broken down by domain, speech complexity, and agent LLM.

Usage:
    python -m experiments.tau_voice.exp.performance_analysis --data-dir data/simulations

The script expects simulation folders with the naming pattern:
    {DATE}_{DOMAIN}_{NUM_TASKS}_voice_{SPEECH_COMPLEXITY}_{MAX_STEPS}s

For example:
    2026_01_07:23_20_57_retail_20_voice_low_600s
"""

import argparse
import shutil
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# =============================================================================
# Parsing and Data Loading (imported from data_loader.py)
# =============================================================================
from experiments.tau_voice.exp.data_loader import load_simulation_results

# =============================================================================
# Configuration and Styling (imported from plot_style.py)
# =============================================================================
from experiments.tau_voice.exp.plot_style import (
    BAR_STYLE,
    DOMAIN_COLORS,
    DOMAINS,
    SPEECH_COMPLEXITIES,
    SPEECH_COMPLEXITY_COLORS,
    get_bar_style,
    get_complexity_display_name,
    get_complexity_style,
    get_legend_patch,
    get_llm_color,
    style_axis,
)
from tau2.data_model.simulation import Results
from tau2.data_model.voice_personas import ALL_PERSONAS
from tau2.metrics.agent_metrics import compute_metrics, prepare_dfs
from tau2.metrics.break_down_metrics import result_reward_actions_analysis
from tau2.utils.utils import DATA_DIR


def build_metrics_dataframe(
    results: List[Tuple[dict, Results]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build metrics DataFrames from simulation results.

    Returns:
        - df_metrics: Aggregated metrics per simulation (one row per simulation)
        - df_pass_hat_k: Per-task pass^k values
    """
    rows = []
    dfs_pass_hat_k = []

    for params, sim_results in results:
        row = deepcopy(params)
        metrics = compute_metrics(sim_results)
        df, df_pass_hat_k = prepare_dfs(sim_results)

        # Add pass^k columns
        row.update(metrics.as_dict())

        # Compute std for each pass^k across tasks
        for col in df_pass_hat_k.columns:
            if col.startswith("pass^"):
                # Extract k value and compute std
                k = col.replace("pass^", "")
                row[f"pass_hat_{k}_std"] = df_pass_hat_k[col].std()

        rows.append(row)

        # Build per-task dataframe
        df_pass_hat_k.reset_index(inplace=True)
        df_pass_hat_k["llm"] = params["llm"]
        df_pass_hat_k["domain"] = params["domain"]
        df_pass_hat_k["speech_complexity"] = params["speech_complexity"]
        dfs_pass_hat_k.append(df_pass_hat_k)

    df_metrics = pd.DataFrame(rows)
    df_all_pass_hat_k = pd.concat(dfs_pass_hat_k, ignore_index=True)

    return df_metrics, df_all_pass_hat_k


def build_simulation_level_dataframe(
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """
    Build a per-simulation DataFrame that includes speech environment details.

    This dataframe has one row per simulation run with:
    - Standard simulation info (task_id, trial, reward, etc.)
    - Speech environment info (persona_name, background_noise_file)
    - Experiment params (llm, domain, speech_complexity)

    Returns:
        DataFrame with per-simulation data including persona_name and background_noise_file
    """
    from tau2.metrics.agent_metrics import is_successful

    rows = []

    for params, sim_results in results:
        if sim_results.simulations is None:
            logger.warning(
                f"Missing simulations for {params.get('llm', '?')}/{params.get('domain', '?')}/{params.get('speech_complexity', '?')}"
            )
            continue
        for sim in sim_results.simulations:
            row = {
                "simulation_id": sim.id,
                "task_id": sim.task_id,
                "trial": sim.trial,
                "reward": sim.reward_info.reward if sim.reward_info else None,
                "success": (
                    is_successful(sim.reward_info.reward) if sim.reward_info else None
                ),
                "duration": sim.duration,
                "llm": params["llm"],
                "domain": params["domain"],
                "speech_complexity": params["speech_complexity"],
            }

            # Extract speech environment info
            if sim.speech_environment:
                row["persona_name"] = sim.speech_environment.persona_name
                row["background_noise_file"] = (
                    sim.speech_environment.background_noise_file
                )
            else:
                row["persona_name"] = None
                row["background_noise_file"] = None

            # Extract termination reason
            row["termination_reason"] = (
                sim.termination_reason.value
                if hasattr(sim.termination_reason, "value")
                else str(sim.termination_reason)
            )

            rows.append(row)

    return pd.DataFrame(rows)


def compute_pass_k_by_group(
    df: pd.DataFrame,
    group_columns: List[str],
    max_k: int = 5,
) -> pd.DataFrame:
    """
    Compute pass^k metrics for each group defined by group_columns.

    Args:
        df: DataFrame with per-simulation data (must have 'success', 'task_id' columns)
        group_columns: Columns to group by (e.g., ['llm', 'persona_name'])
        max_k: Maximum k value for pass^k computation

    Returns:
        DataFrame with pass^k values for each group
    """
    from tau2.metrics.agent_metrics import pass_hat_k

    results_rows = []

    # Group by the specified columns
    for group_key, group_df in df.groupby(group_columns, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = dict(zip(group_columns, group_key))

        # For each task in this group, compute pass^k
        task_pass_ks = {k: [] for k in range(1, max_k + 1)}

        for task_id, task_df in group_df.groupby("task_id"):
            # Filter to only trials with valid reward_info (success is not None)
            valid_df = task_df[task_df["success"].notna()]
            num_trials = len(valid_df)
            if num_trials == 0:
                continue
            success_count = valid_df["success"].sum()

            for k in range(1, min(num_trials, max_k) + 1):
                try:
                    pk = pass_hat_k(num_trials, success_count, k)
                    task_pass_ks[k].append(pk)
                except ValueError:
                    # Not enough trials for this k
                    pass

        # Average pass^k across tasks (and compute std)
        for k in range(1, max_k + 1):
            if task_pass_ks[k]:
                row[f"pass_hat_{k}"] = np.mean(task_pass_ks[k])
                row[f"pass_hat_{k}_std"] = np.std(task_pass_ks[k])
            else:
                row[f"pass_hat_{k}"] = np.nan
                row[f"pass_hat_{k}_std"] = np.nan

        row["num_tasks"] = group_df["task_id"].nunique()
        row["num_simulations"] = len(group_df)
        row["avg_reward"] = group_df["reward"].mean()

        results_rows.append(row)

    return pd.DataFrame(results_rows)


def compute_pass_k_by_persona(
    results: List[Tuple[dict, Results]],
    max_k: int = 5,
    include_complexity: bool = False,
    include_domain: bool = False,
) -> pd.DataFrame:
    """
    Compute pass^k metrics grouped by persona_name for each LLM.

    Args:
        results: List of (params, Results) tuples
        max_k: Maximum k value for pass^k computation
        include_complexity: If True, also group by speech_complexity
        include_domain: If True, also group by domain

    Returns:
        DataFrame with columns: llm, [domain], persona_name, [speech_complexity], pass_hat_1, ..., num_tasks, avg_reward
    """
    df_sims = build_simulation_level_dataframe(results)
    group_cols = ["llm"]
    if include_domain:
        group_cols.append("domain")
    group_cols.append("persona_name")
    if include_complexity:
        group_cols.append("speech_complexity")
    return compute_pass_k_by_group(df_sims, group_cols, max_k)


def compute_pass_k_by_background_noise(
    results: List[Tuple[dict, Results]],
    max_k: int = 5,
    include_complexity: bool = False,
    include_domain: bool = False,
) -> pd.DataFrame:
    """
    Compute pass^k metrics grouped by background_noise_file for each LLM.

    Args:
        results: List of (params, Results) tuples
        max_k: Maximum k value for pass^k computation
        include_complexity: If True, also group by speech_complexity
        include_domain: If True, also group by domain

    Returns:
        DataFrame with columns: llm, [domain], background_noise_file, [speech_complexity], pass_hat_1, ..., num_tasks, avg_reward
    """
    df_sims = build_simulation_level_dataframe(results)
    group_cols = ["llm"]
    if include_domain:
        group_cols.append("domain")
    group_cols.append("background_noise_file")
    if include_complexity:
        group_cols.append("speech_complexity")
    return compute_pass_k_by_group(df_sims, group_cols, max_k)


# =============================================================================
# Plotting Functions
# =============================================================================


def get_pass_hat_k_columns(df: pd.DataFrame) -> List[str]:
    """Get all pass_hat_k column names from a DataFrame (excluding _std columns)."""
    return [
        col
        for col in df.columns
        if col.startswith("pass_hat_") and not col.endswith("_std")
    ]


def get_k_values(df: pd.DataFrame) -> List[int]:
    """Get sorted list of k values from pass_hat_k columns."""
    cols = get_pass_hat_k_columns(df)
    return sorted([int(col.split("_")[-1]) for col in cols])


def save_pass_k_by_domain_raw(
    output_dir: Path,
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """
    Save raw per-simulation data for pass_k_by_domain analysis.

    Creates raw.csv with columns:
    - simulation_id, task_id, trial
    - domain, speech_complexity, llm, provider
    - reward, success
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for params, sim_results in results:
        llm = params.get("llm", "unknown")
        domain = params.get("domain", "unknown")
        speech_complexity = params.get("speech_complexity", "unknown")
        provider = params.get("provider", "unknown")

        if sim_results.simulations is None:
            logger.warning(
                f"Missing simulations for {llm}/{domain}/{speech_complexity}"
            )
            continue

        for sim in sim_results.simulations:
            reward = sim.reward_info.reward if sim.reward_info else 0.0
            rows.append(
                {
                    "simulation_id": sim.id,
                    "task_id": sim.task_id,
                    "trial": sim.trial,
                    "domain": domain,
                    "speech_complexity": speech_complexity,
                    "llm": llm,
                    "provider": provider,
                    "reward": reward,
                    "success": reward == 1.0,
                }
            )

    df = pd.DataFrame(rows)
    raw_path = output_dir / f"{output_dir.name}_raw.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved: {raw_path}")
    return df


def save_pass_k_by_domain_analysis(
    output_dir: Path,
    df_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Save analysis table for pass_k_by_domain.

    Creates analysis.csv with aggregated pass^k by (domain, llm, speech_complexity).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get pass_hat columns (excluding std columns)
    pass_hat_cols = get_pass_hat_k_columns(df_metrics)
    std_cols = [col for col in df_metrics.columns if col.endswith("_std")]

    # Select relevant columns
    cols_to_keep = (
        ["domain", "llm", "speech_complexity", "provider", "num_tasks"]
        + pass_hat_cols
        + std_cols
        + ["avg_reward"]
    )

    cols_available = [c for c in cols_to_keep if c in df_metrics.columns]
    df_analysis = df_metrics[cols_available].copy()

    analysis_path = output_dir / f"{output_dir.name}_analysis.csv"
    df_analysis.to_csv(analysis_path, index=False)
    logger.info(f"Saved: {analysis_path}")
    return df_analysis


def plot_pass_1_by_domain(
    output_dir: Path,
    df_metrics: pd.DataFrame,
) -> None:
    """
    Create pass^1 plot with one subplot per domain, LLM names on x-axis.

    Shows pass^1 for each LLM with control vs regular comparison using
    consistent styling (alpha, hatch).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "pass_hat_1" not in df_metrics.columns:
        logger.warning("pass_hat_1 not found. Skipping pass^1 by domain plot.")
        return

    llms = sorted(df_metrics["llm"].unique())
    tested_domains = df_metrics["domain"].unique()
    domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    complexities = ["control", "regular"]

    n_domains = len(domains_to_plot)
    if n_domains == 0:
        return

    fig, axes = plt.subplots(n_domains, 1, figsize=(10, 5 * n_domains), squeeze=False)
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains_to_plot):
        ax = axes[d_idx]
        domain_data = df_metrics[df_metrics["domain"] == domain]

        x = np.arange(len(llms))
        bar_width = 0.35

        for i, complexity in enumerate(complexities):
            # Use complexity color instead of LLM color
            complexity_color = SPEECH_COMPLEXITY_COLORS.get(complexity, "#888888")
            values = []

            for llm in llms:
                subset = domain_data[
                    (domain_data["llm"] == llm)
                    & (domain_data["speech_complexity"] == complexity)
                ]
                if len(subset) > 0:
                    values.append(subset["pass_hat_1"].mean())
                else:
                    values.append(np.nan)  # No data available for this config

            ax.bar(
                x + i * bar_width,
                values,
                bar_width,
                color=complexity_color,
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )

            # Add value labels
            for bar_x, val in zip(x + i * bar_width, values):
                if not np.isnan(val):
                    label = f"{val:.2f}"
                    ax.text(
                        bar_x, val + 0.02, label, ha="center", va="bottom", fontsize=9
                    )

        ax.set_xlabel("LLM", fontsize=11)
        ax.set_ylabel("Pass^1", fontsize=11)
        ax.set_title(f"{domain.capitalize()}", fontsize=13, fontweight="bold")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(
            [llm.split(":")[-1][:20] for llm in llms],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax.set_ylim(0, 1.15)
        style_axis(ax)

    # Add legend for control/regular with complexity colors
    legend_elements = [
        get_legend_patch(
            "control", facecolor=SPEECH_COMPLEXITY_COLORS.get("control", "#888888")
        ),
        get_legend_patch(
            "regular", facecolor=SPEECH_COMPLEXITY_COLORS.get("regular", "#888888")
        ),
    ]
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.suptitle("Pass^1 by Domain", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "pass_1.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'pass_1.pdf'}")


def plot_pass_1_headline(
    output_dir: Path,
    df_metrics: pd.DataFrame,
    include_text_baseline: bool = True,
) -> None:
    """
    Create a compact, publication-ready pass^1 figure for the paper headline.

    For each model (including SOTA text baseline on the left), shows grouped bars:
    - Overall average across domains
    - Per-domain scores (Retail, Airline, Telecom)

    Args:
        output_dir: Directory to save the figure
        df_metrics: DataFrame with pass^k metrics
        include_text_baseline: Whether to include SOTA text-based tau-bench on the left
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "pass_hat_1" not in df_metrics.columns:
        logger.warning("pass_hat_1 not found. Skipping headline plot.")
        return

    # Filter to regular mode only
    df_regular = df_metrics[df_metrics["speech_complexity"] == "regular"]
    if df_regular.empty:
        logger.warning("No regular mode data found. Skipping headline plot.")
        return

    llms = sorted(df_regular["llm"].unique())
    tested_domains = df_regular["domain"].unique()
    domains_to_plot = [d for d in DOMAINS if d in tested_domains]

    n_llms = len(llms)
    n_domains = len(domains_to_plot)

    if n_llms == 0 or n_domains == 0:
        logger.warning("Insufficient data for headline plot.")
        return

    # Categories: Overall + each domain
    categories = ["All"] + [d.capitalize() for d in domains_to_plot]
    n_categories = len(categories)

    # Load text-based baselines (hardcoded reference scores)
    text_baselines_data = []
    if include_text_baseline:
        from experiments.tau_voice.exp.text_baselines import DEFAULT_TEXT_BASELINES

        for baseline in DEFAULT_TEXT_BASELINES:
            scores = baseline.get_scores_dict(domains_to_plot)
            text_baselines_data.append((baseline.display_name, scores))
            logger.info(f"Text baseline: {baseline.model_name}")

    # Compute scores for each voice LLM
    llm_scores = {}
    for llm in llms:
        llm_data = df_regular[df_regular["llm"] == llm]
        scores = {}
        # Overall average
        if len(llm_data) > 0:
            scores["All"] = llm_data["pass_hat_1"].mean()
        # Per-domain
        for domain in domains_to_plot:
            domain_data = llm_data[llm_data["domain"] == domain]
            if len(domain_data) > 0:
                scores[domain.capitalize()] = domain_data["pass_hat_1"].mean()
            else:
                scores[domain.capitalize()] = np.nan
        llm_scores[llm] = scores

    # Build list of models to plot (text baselines first if available)
    models_to_plot = []
    n_text_baselines = len(text_baselines_data)
    for name, scores in text_baselines_data:
        models_to_plot.append((f"Text\n({name})", scores, "#666666"))
    for llm in llms:
        # Shorten LLM names
        llm_short = llm.split(":")[-1] if ":" in llm else llm
        # Further shorten common patterns
        llm_short = llm_short.replace("gpt-realtime-", "gpt-rt-")
        llm_short = llm_short.replace("gemini-2.0-flash-live-", "gem-2-live-")
        llm_short = llm_short.replace("grok-2-realtime-", "grok-2-rt-")
        if len(llm_short) > 15:
            llm_short = llm_short[:15]
        models_to_plot.append((llm_short, llm_scores[llm], get_llm_color(llm)))

    n_models = len(models_to_plot)

    # Create figure - more compact
    fig_width = max(6, 1.2 * n_models)
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    # Bar positioning - tighter spacing
    group_width = 0.75
    bar_width = group_width / n_categories
    model_positions = np.arange(n_models) * 0.85  # Compress x-axis spacing

    # Category colors: "All" is muted slate, domains use DOMAIN_COLORS
    category_colors = {"All": "#778899"}  # Soft slate gray for overall
    for domain in domains_to_plot:
        category_colors[domain.capitalize()] = DOMAIN_COLORS.get(domain, "#888888")

    # Plot bars for each model
    for m_idx, (model_name, scores, _) in enumerate(models_to_plot):
        for c_idx, cat in enumerate(categories):
            val = scores.get(cat, np.nan)
            if np.isnan(val):
                continue

            # Bar position within the group
            bar_offset = (c_idx - (n_categories - 1) / 2) * bar_width
            x = model_positions[m_idx] + bar_offset

            # Use category/domain color instead of LLM color
            bar_color = category_colors.get(cat, "#888888")

            ax.bar(
                x,
                val,
                bar_width * 0.85,
                color=bar_color,
                edgecolor="white",
                linewidth=0.5,
            )

            # Add value label on top
            ax.text(
                x,
                val + 0.015,
                f"{val:.0%}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="medium",
            )

    # Add vertical separator after text baselines
    if n_text_baselines > 0 and n_models > n_text_baselines:
        sep_x = (
            model_positions[n_text_baselines - 1] + model_positions[n_text_baselines]
        ) / 2
        ax.axvline(x=sep_x, color="#cccccc", linestyle="-", linewidth=1, zorder=0)

    # Styling
    ax.set_ylabel("Pass^1", fontsize=11, fontweight="medium")
    ax.set_xticks(model_positions)
    ax.set_xticklabels(
        [m[0] for m in models_to_plot],
        fontsize=9,
        fontweight="medium",
        rotation=20,
        ha="right",
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, model_positions[-1] + 0.5)

    # Clean axis style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)

    # Legend for categories with domain colors
    from matplotlib.patches import Patch

    legend_patches = []
    for cat in categories:
        cat_color = category_colors.get(cat, "#888888")
        legend_patches.append(Patch(facecolor=cat_color, edgecolor="white", label=cat))
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=8,
        frameon=False,
        ncol=n_categories,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "pass_1_headline.pdf", format="pdf", bbox_inches="tight", dpi=300
    )
    plt.close()
    logger.info(f"Saved: {output_dir / 'pass_1_headline.pdf'}")


def plot_pass_k_by_domain(fig_dir: Path, df_metrics: pd.DataFrame) -> None:
    """
    Create a pass^k plot with one graph per domain.

    Focuses on control vs regular comparison only. Uses a cleaner layout
    with LLMs grouped together and complexity shown via hatching.
    """
    k_values = get_k_values(df_metrics)
    if not k_values:
        logger.warning("No pass^k columns found. Skipping pass^k by domain plot.")
        return

    llms = sorted(df_metrics["llm"].unique())
    tested_domains = df_metrics["domain"].unique()
    domains_to_plot = [d for d in DOMAINS if d in tested_domains]

    # Focus on control and regular only for cleaner plot
    core_complexities = ["control", "regular"]
    tested_complexities = df_metrics["speech_complexity"].unique()
    complexities_to_plot = [c for c in core_complexities if c in tested_complexities]

    n_llms = len(llms)
    n_domains = len(domains_to_plot)
    n_complexities = len(complexities_to_plot)

    if n_llms == 0 or n_domains == 0 or n_complexities == 0:
        logger.warning(
            f"Insufficient data for pass^k by domain: "
            f"llms={n_llms}, domains={n_domains}, complexities={n_complexities}"
        )
        return

    # Create one subplot per domain (horizontal layout for compactness)
    fig, axes = plt.subplots(1, n_domains, figsize=(5 * n_domains, 4), squeeze=False)
    axes = axes[0, :]

    for d_idx, domain in enumerate(domains_to_plot):
        ax = axes[d_idx]
        domain_data = df_metrics[df_metrics["domain"] == domain]

        # Group by LLM, show complexities as adjacent bars
        x = np.arange(n_llms)
        bar_width = 0.8 / n_complexities

        for c_idx, complexity in enumerate(complexities_to_plot):
            values = []

            # Use complexity color instead of LLM color
            complexity_color = SPEECH_COMPLEXITY_COLORS.get(complexity, "#888888")

            for llm in llms:
                subset = domain_data[
                    (domain_data["llm"] == llm)
                    & (domain_data["speech_complexity"] == complexity)
                ]
                # Use pass^1 for this simplified view
                if len(subset) > 0 and "pass_hat_1" in subset.columns:
                    values.append(subset["pass_hat_1"].mean())
                else:
                    values.append(np.nan)

            bar_offset = (c_idx - (n_complexities - 1) / 2) * bar_width
            x_pos = x + bar_offset

            # Plot all bars for this complexity with the same color
            for xp, val in zip(x_pos, values):
                if not np.isnan(val):
                    ax.bar(
                        xp,
                        val,
                        bar_width * 0.9,
                        color=complexity_color,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    # Add small value label on top
                    ax.text(
                        xp,
                        val + 0.01,
                        f"{val:.0%}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        # X-axis: LLM names
        llm_short_names = [
            llm.split(":")[-1][:12] if ":" in llm else llm[:12] for llm in llms
        ]
        ax.set_xlabel("")
        ax.set_ylabel("Pass^1" if d_idx == 0 else "", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(llm_short_names, fontsize=8, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{domain.capitalize()}", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Add compact legend below the plots with complexity colors
    from matplotlib.patches import Patch

    legend_patches = []
    for complexity in complexities_to_plot:
        complexity_color = SPEECH_COMPLEXITY_COLORS.get(complexity, "#888888")
        legend_patches.append(
            Patch(
                facecolor=complexity_color,
                edgecolor="white",
                label=get_complexity_display_name(complexity),
            )
        )

    fig.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=n_complexities,
        fontsize=9,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(fig_dir / "pass_k.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved: {fig_dir / 'pass_k.pdf'}")


def save_pass_1_complexity_table(
    output_dir: Path,
    df_metrics: pd.DataFrame,
) -> None:
    """
    Save a markdown table of Pass^1 scores by complexity level.

    Columns: LLMs with sub-columns for each domain (Retail, Airline, Telecom, All)
    Rows: Complexity levels (control, ablations, regular)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "pass_hat_1" not in df_metrics.columns:
        logger.warning("pass_hat_1 not found. Skipping complexity table.")
        return

    llms = sorted(df_metrics["llm"].unique())
    tested_domains = df_metrics["domain"].unique()
    domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    tested_complexities = df_metrics["speech_complexity"].unique()
    complexities_to_plot = [c for c in SPEECH_COMPLEXITIES if c in tested_complexities]

    if not llms or not domains_to_plot or not complexities_to_plot:
        logger.warning("Insufficient data for complexity table.")
        return

    # Shorten LLM names for table headers
    llm_short_names = {}
    for llm in llms:
        short = llm.split(":")[-1] if ":" in llm else llm
        short = short.replace("gpt-realtime-", "gpt-rt-")
        short = short.replace("gemini-2.0-flash-live-", "gem-2-live-")
        short = short.replace("grok-2-realtime-", "grok-2-rt-")
        if len(short) > 15:
            short = short[:15]
        llm_short_names[llm] = short

    # Domain labels including "All"
    domain_labels = [d.capitalize() for d in domains_to_plot] + ["All"]

    # Build markdown table
    lines = []
    lines.append("# Pass^1 by Complexity Level\n")

    # Header row 1: LLM names spanning domain columns
    header1_parts = ["Complexity"]
    for llm in llms:
        header1_parts.append(f"**{llm_short_names[llm]}**")
        header1_parts.extend([""] * (len(domain_labels) - 1))
    lines.append("| " + " | ".join(header1_parts) + " |")

    # Header row 2: Domain sub-columns
    header2_parts = [""]
    for llm in llms:
        header2_parts.extend(domain_labels)
    lines.append("| " + " | ".join(header2_parts) + " |")

    # Separator row
    sep_parts = ["---"] * (1 + len(llms) * len(domain_labels))
    lines.append("| " + " | ".join(sep_parts) + " |")

    # Data rows for each complexity
    for complexity in complexities_to_plot:
        display_name = get_complexity_display_name(complexity)
        row_parts = [f"**{display_name}**"]

        for llm in llms:
            llm_data = df_metrics[
                (df_metrics["llm"] == llm)
                & (df_metrics["speech_complexity"] == complexity)
            ]

            # Per-domain scores
            for domain in domains_to_plot:
                domain_data = llm_data[llm_data["domain"] == domain]
                if len(domain_data) > 0 and "pass_hat_1" in domain_data.columns:
                    val = domain_data["pass_hat_1"].mean()
                    row_parts.append(f"{val:.1%}")
                else:
                    row_parts.append("-")

            # All (average across domains)
            if len(llm_data) > 0 and "pass_hat_1" in llm_data.columns:
                val = llm_data["pass_hat_1"].mean()
                row_parts.append(f"{val:.1%}")
            else:
                row_parts.append("-")

        lines.append("| " + " | ".join(row_parts) + " |")

    # Write to file
    md_path = output_dir / "pass_1_complexity_table.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {md_path}")


def plot_pass_1_all_complexities(
    output_dir: Path,
    df_metrics: pd.DataFrame,
) -> None:
    """
    Create a Pass^1 plot showing all complexity levels.

    Layout: One subplot per domain, LLMs on x-axis, complexities as grouped bars.
    Uses hatching/alpha to distinguish complexities.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "pass_hat_1" not in df_metrics.columns:
        logger.warning("pass_hat_1 not found. Skipping all-complexities plot.")
        return

    llms = sorted(df_metrics["llm"].unique())
    tested_domains = df_metrics["domain"].unique()
    domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    tested_complexities = df_metrics["speech_complexity"].unique()
    complexities_to_plot = [c for c in SPEECH_COMPLEXITIES if c in tested_complexities]

    n_llms = len(llms)
    n_domains = len(domains_to_plot)
    n_complexities = len(complexities_to_plot)

    if n_llms == 0 or n_domains == 0 or n_complexities == 0:
        logger.warning("Insufficient data for all-complexities plot.")
        return

    # Create one subplot per domain (horizontal layout)
    fig, axes = plt.subplots(1, n_domains, figsize=(5 * n_domains, 5), squeeze=False)
    axes = axes[0, :]

    for d_idx, domain in enumerate(domains_to_plot):
        ax = axes[d_idx]
        domain_data = df_metrics[df_metrics["domain"] == domain]

        # LLMs on x-axis, complexities as grouped bars
        x = np.arange(n_llms)
        bar_width = 0.8 / n_complexities

        for c_idx, complexity in enumerate(complexities_to_plot):
            # Use complexity color instead of LLM color
            complexity_color = SPEECH_COMPLEXITY_COLORS.get(complexity, "#888888")
            values = []

            for llm in llms:
                subset = domain_data[
                    (domain_data["llm"] == llm)
                    & (domain_data["speech_complexity"] == complexity)
                ]
                if len(subset) > 0 and "pass_hat_1" in subset.columns:
                    values.append(subset["pass_hat_1"].mean())
                else:
                    values.append(np.nan)

            bar_offset = (c_idx - (n_complexities - 1) / 2) * bar_width
            x_pos = x + bar_offset

            # Plot all bars for this complexity with the same color
            for xp, val in zip(x_pos, values):
                if not np.isnan(val):
                    ax.bar(
                        xp,
                        val,
                        bar_width * 0.9,
                        color=complexity_color,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    # Add small value label on top (horizontal, compact format)
                    label = f".{round(val * 100)}" if val < 1 else "1"
                    ax.text(
                        xp,
                        val + 0.01,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )

        # X-axis: LLM names (shortened)
        llm_short_names = [
            llm.split(":")[-1][:10] if ":" in llm else llm[:10] for llm in llms
        ]
        ax.set_ylabel("Pass^1" if d_idx == 0 else "", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(llm_short_names, fontsize=8, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{domain.capitalize()}", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Add legend below the plots showing all complexities with their colors
    from matplotlib.patches import Patch

    legend_patches = []
    for complexity in complexities_to_plot:
        complexity_color = SPEECH_COMPLEXITY_COLORS.get(complexity, "#888888")
        label = get_complexity_display_name(complexity)
        legend_patches.append(
            Patch(
                facecolor=complexity_color,
                edgecolor="white",
                label=label,
            )
        )

    fig.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(n_complexities, 5),
        fontsize=8,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(
        output_dir / "pass_1_all_complexities.pdf", bbox_inches="tight", dpi=300
    )
    plt.close()
    logger.info(f"Saved: {output_dir / 'pass_1_all_complexities.pdf'}")


def plot_pass_1_filtered(
    output_dir: Path,
    df_metrics: pd.DataFrame,
    domains_filter: list[str] | None = None,
    llms_filter: list[str] | None = None,
    filename: str = "pass_1_filtered.pdf",
) -> None:
    """
    Create a filtered Pass^1 plot for specific domains and LLMs.

    Args:
        output_dir: Directory to save the plot
        df_metrics: DataFrame with metrics
        domains_filter: List of domains to include (None = all)
        llms_filter: List of LLM name patterns to include (None = all)
        filename: Output filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "pass_hat_1" not in df_metrics.columns:
        logger.warning("pass_hat_1 not found. Skipping filtered plot.")
        return

    # Filter LLMs by pattern matching
    all_llms = sorted(df_metrics["llm"].unique())
    if llms_filter:
        llms = [
            llm
            for llm in all_llms
            if any(pattern.lower() in llm.lower() for pattern in llms_filter)
        ]
    else:
        llms = all_llms

    # Filter domains
    tested_domains = df_metrics["domain"].unique()
    if domains_filter:
        domains_to_plot = [d for d in domains_filter if d in tested_domains]
    else:
        domains_to_plot = [d for d in DOMAINS if d in tested_domains]

    tested_complexities = df_metrics["speech_complexity"].unique()
    complexities_to_plot = [c for c in SPEECH_COMPLEXITIES if c in tested_complexities]

    n_llms = len(llms)
    n_domains = len(domains_to_plot)
    n_complexities = len(complexities_to_plot)

    if n_llms == 0 or n_domains == 0 or n_complexities == 0:
        logger.warning(
            f"Insufficient data for filtered plot: {n_llms} LLMs, {n_domains} domains, {n_complexities} complexities"
        )
        return

    # Create one subplot per domain (compact for column width)
    fig_width = max(3.5, 1.2 * n_llms) * n_domains
    fig, axes = plt.subplots(
        1, n_domains, figsize=(fig_width / n_domains, 3.5), squeeze=False
    )
    axes = axes[0, :]

    for d_idx, domain in enumerate(domains_to_plot):
        ax = axes[d_idx]
        domain_data = df_metrics[df_metrics["domain"] == domain]

        # LLMs on x-axis, complexities as grouped bars - tighter spacing
        x = np.arange(n_llms) * 0.85
        bar_width = 0.7 / n_complexities

        for c_idx, complexity in enumerate(complexities_to_plot):
            complexity_color = SPEECH_COMPLEXITY_COLORS.get(complexity, "#888888")
            values = []

            for llm in llms:
                subset = domain_data[
                    (domain_data["llm"] == llm)
                    & (domain_data["speech_complexity"] == complexity)
                ]
                if len(subset) > 0 and "pass_hat_1" in subset.columns:
                    values.append(subset["pass_hat_1"].mean())
                else:
                    values.append(np.nan)

            bar_offset = (c_idx - (n_complexities - 1) / 2) * bar_width
            x_pos = x + bar_offset

            for xp, val in zip(x_pos, values):
                if not np.isnan(val):
                    ax.bar(
                        xp,
                        val,
                        bar_width * 0.9,
                        color=complexity_color,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    # Add value label in compact format (horizontal, small)
                    label = f".{round(val * 100):02d}"
                    ax.text(
                        xp,
                        val + 0.02,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=5,
                    )

        # X-axis: LLM names (shortened)
        llm_short_names = []
        for llm in llms:
            short = llm.split(":")[-1] if ":" in llm else llm
            short = short.replace("gpt-realtime-", "gpt-rt-")
            short = short.replace("gemini-2.0-flash-live-", "gem-2-live-")
            short = short.replace("grok-2-realtime-", "grok-2-rt-")
            llm_short_names.append(short[:12])

        ax.set_ylabel("Pass^1" if d_idx == 0 else "", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(llm_short_names, fontsize=8, rotation=35, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_xlim(-0.4, x[-1] + 0.4 if len(x) > 0 else 0.5)
        ax.set_title(
            f"{domain.capitalize()} - Ablation", fontsize=10, fontweight="bold"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch

    legend_patches = []
    for complexity in complexities_to_plot:
        complexity_color = SPEECH_COMPLEXITY_COLORS.get(complexity, "#888888")
        label = get_complexity_display_name(complexity)
        legend_patches.append(
            Patch(facecolor=complexity_color, edgecolor="white", label=label)
        )

    # Legend in upper right of plot area (below title, above bars)
    axes[-1].legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=6,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
    )

    plt.tight_layout()
    plt.savefig(output_dir / filename, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved: {output_dir / filename}")


def save_metrics_csv(fig_dir: Path, df_metrics: pd.DataFrame) -> None:
    """Save the metrics DataFrame to CSV for further analysis."""
    csv_path = fig_dir / "voice_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics to: {csv_path}")


# =============================================================================
# Action Success Analysis (Tool Call Success/Failure)
# =============================================================================


def extract_action_data(
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """
    Extract action (tool call) data from all simulations.

    Returns DataFrame with columns:
    - llm, domain, speech_complexity
    - simulation_id, task_id, trial
    - tick_idx, requestor, action_name
    - arguments, error, error_message
    """
    rows = []

    for params, sim_results in results:
        llm = params.get("llm", "unknown")
        domain = params.get("domain", "unknown")
        speech_complexity = params.get("speech_complexity", "unknown")

        if sim_results.simulations is None:
            logger.warning(
                f"Missing simulations for {llm}/{domain}/{speech_complexity}"
            )
            continue

        for sim in sim_results.simulations:
            sim_id = sim.id
            task_id = sim.task_id
            trial = sim.trial

            for tick_idx, tick in enumerate(sim.ticks):
                # Process agent tool calls
                if tick.agent_tool_calls:
                    for i, tool_call in enumerate(tick.agent_tool_calls):
                        # Find matching result if available
                        error = False
                        error_message = ""
                        if tick.agent_tool_results and i < len(tick.agent_tool_results):
                            result = tick.agent_tool_results[i]
                            error = getattr(result, "error", False)
                            if error:
                                error_message = str(getattr(result, "content", ""))[
                                    :500
                                ]

                        rows.append(
                            {
                                "llm": llm,
                                "domain": domain,
                                "speech_complexity": speech_complexity,
                                "simulation_id": sim_id,
                                "task_id": task_id,
                                "trial": trial,
                                "tick_idx": tick_idx,
                                "requestor": "assistant",
                                "action_name": tool_call.name,
                                "arguments": str(tool_call.arguments)[:500],
                                "error": error,
                                "error_message": error_message,
                            }
                        )

                # Process user tool calls
                if tick.user_tool_calls:
                    for i, tool_call in enumerate(tick.user_tool_calls):
                        # Find matching result if available
                        error = False
                        error_message = ""
                        if tick.user_tool_results and i < len(tick.user_tool_results):
                            result = tick.user_tool_results[i]
                            error = getattr(result, "error", False)
                            if error:
                                error_message = str(getattr(result, "content", ""))[
                                    :500
                                ]

                        rows.append(
                            {
                                "llm": llm,
                                "domain": domain,
                                "speech_complexity": speech_complexity,
                                "simulation_id": sim_id,
                                "task_id": task_id,
                                "trial": trial,
                                "tick_idx": tick_idx,
                                "requestor": "user",
                                "action_name": tool_call.name,
                                "arguments": str(tool_call.arguments)[:500],
                                "error": error,
                                "error_message": error_message,
                            }
                        )

    return pd.DataFrame(rows)


def save_action_success_raw(
    output_dir: Path,
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """Save raw action data to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = extract_action_data(results)
    raw_path = output_dir / f"{output_dir.name}_raw.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved: {raw_path}")
    return df


def save_action_success_analysis(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze action success rates and save results.

    Creates:
    - analysis.csv: Per-action success rates by LLM × [domain] × complexity
    - success_rates.csv: Pivoted table with actions as rows
    - error_summary.csv: Error message counts
    - error_counts.csv: Error counts pivoted by LLM × complexity
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_raw.empty:
        logger.warning("No action data to analyze")
        return pd.DataFrame()

    # Filter to agent actions only
    df_agent = df_raw[df_raw["requestor"] == "assistant"].copy()

    if df_agent.empty:
        logger.warning("No agent actions found")
        return pd.DataFrame()

    # Check if domain column exists
    has_domain = "domain" in df_agent.columns
    group_cols = ["llm"]
    if has_domain:
        group_cols.append("domain")
    group_cols.extend(["speech_complexity", "action_name"])

    # === Analysis: success rate by action, LLM, [domain], complexity ===
    grouped = (
        df_agent.groupby(group_cols)
        .agg(
            num_calls=("error", "count"),
            num_errors=("error", "sum"),
        )
        .reset_index()
    )
    grouped["success_rate"] = 1 - grouped["num_errors"] / grouped["num_calls"]

    analysis_path = output_dir / f"{output_dir.name}_analysis.csv"
    grouped.to_csv(analysis_path, index=False)
    logger.info(f"Saved: {analysis_path}")

    # === Pivoted success rates table ===
    grouped["llm_complexity"] = (
        grouped["llm"].apply(lambda x: x.split(":")[-1][:20] if ":" in x else x[:20])
        + " ("
        + grouped["speech_complexity"]
        + ")"
    )

    pivot_table = grouped.pivot_table(
        index="action_name",
        columns="llm_complexity",
        values="success_rate",
        fill_value=0,
        aggfunc="mean",
    )

    # Add overall success rate
    action_success = df_agent.groupby("action_name").apply(
        lambda x: 1 - x["error"].sum() / len(x), include_groups=False
    )
    pivot_table["OVERALL"] = pivot_table.index.map(action_success)
    pivot_table = pivot_table.sort_values("OVERALL", ascending=True)

    pivot_path = output_dir / "success_rates.csv"
    pivot_table.to_csv(pivot_path)
    logger.info(f"Saved: {pivot_path}")

    # === Error analysis ===
    df_errors = df_agent[df_agent["error"]].copy()

    if not df_errors.empty:
        # Error summary (counts by error message)
        error_summary = df_errors["error_message"].value_counts().reset_index()
        error_summary.columns = ["error_message", "count"]
        summary_path = output_dir / "error_summary.csv"
        error_summary.to_csv(summary_path, index=False)
        logger.info(f"Saved: {summary_path}")

        # Error counts pivoted by LLM × complexity
        df_errors["llm_complexity"] = (
            df_errors["llm"].apply(
                lambda x: x.split(":")[-1][:20] if ":" in x else x[:20]
            )
            + " ("
            + df_errors["speech_complexity"]
            + ")"
        )

        # Categorize errors
        def categorize_error(msg):
            msg_lower = str(msg).lower()
            if "not found" in msg_lower:
                return msg[:60]  # Keep specific message
            elif "missing" in msg_lower and "argument" in msg_lower:
                return "Missing positional arguments"
            else:
                return msg[:60]

        df_errors["error_category"] = df_errors["error_message"].apply(categorize_error)

        error_pivot = (
            df_errors.groupby(["error_category", "llm_complexity"])
            .size()
            .unstack(fill_value=0)
        )
        error_pivot["TOTAL"] = error_pivot.sum(axis=1)
        error_pivot = error_pivot.sort_values("TOTAL", ascending=False)

        counts_path = output_dir / "error_counts.csv"
        error_pivot.to_csv(counts_path)
        logger.info(f"Saved: {counts_path}")

    return grouped


def plot_action_success(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> None:
    """
    Plot action success rates by LLM and complexity.
    Creates separate figures per domain if domain column is present.

    Creates horizontal bar chart with actions ordered from least to most successful.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_raw.empty:
        return

    df_agent = df_raw[df_raw["requestor"] == "assistant"].copy()
    if df_agent.empty:
        return

    # Check if domain column exists and get domains to plot
    has_domain = "domain" in df_agent.columns
    if has_domain:
        tested_domains = df_agent["domain"].unique()
        domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    else:
        domains_to_plot = [None]

    llms = sorted(df_agent["llm"].unique())
    complexities = ["control", "regular"]

    # Create one figure per domain
    for domain in domains_to_plot:
        # Filter data for this domain
        if has_domain and domain is not None:
            domain_df = df_agent[df_agent["domain"] == domain]
        else:
            domain_df = df_agent

        # Get top actions by call count for this domain
        action_counts = domain_df["action_name"].value_counts()
        top_actions = action_counts.head(15).index.tolist()

        df_top = domain_df[domain_df["action_name"].isin(top_actions)].copy()

        if df_top.empty:
            continue

        # Calculate success rate per action
        action_success = (
            df_top.groupby("action_name")
            .apply(lambda x: 1 - x["error"].sum() / len(x), include_groups=False)
            .reset_index()
        )
        action_success.columns = ["action_name", "avg_success"]

        # Sort by average success rate (ascending - least successful first)
        action_success = action_success.sort_values("avg_success", ascending=True)
        sorted_actions = action_success["action_name"].tolist()

        # Calculate per LLM×complexity
        grouped = (
            df_top.groupby(["llm", "speech_complexity", "action_name"])
            .agg(
                num_calls=("error", "count"),
                num_errors=("error", "sum"),
            )
            .reset_index()
        )
        grouped["success_rate"] = 1 - grouped["num_errors"] / grouped["num_calls"]

        fig, axes = plt.subplots(1, len(llms), figsize=(6 * len(llms), 8), sharey=True)
        if len(llms) == 1:
            axes = [axes]

        for ax, llm in zip(axes, llms):
            df_llm = grouped[grouped["llm"] == llm]
            y_pos = np.arange(len(sorted_actions))
            bar_height = 0.35

            for i, complexity in enumerate(complexities):
                df_c = df_llm[df_llm["speech_complexity"] == complexity]
                values = []
                counts = []
                for action in sorted_actions:
                    row = df_c[df_c["action_name"] == action]
                    if len(row) > 0:
                        values.append(row.iloc[0]["success_rate"])
                        counts.append(int(row.iloc[0]["num_calls"]))
                    else:
                        values.append(np.nan)  # No data available for this action
                        counts.append(0)

                style = get_bar_style(complexity, color=get_llm_color(llm))
                ax.barh(y_pos + i * bar_height, values, bar_height, **style)

                # Add count labels (skip for missing data)
                for y, val, count in zip(y_pos + i * bar_height, values, counts):
                    if count > 0 and not np.isnan(val):
                        ax.text(val + 0.02, y, f"n={count}", va="center", fontsize=8)

            ax.set_yticks(y_pos + bar_height / 2)
            ax.set_yticklabels(sorted_actions, fontsize=9)
            ax.set_xlabel("Success Rate", fontsize=11)
            ax.set_title(llm.split(":")[-1][:25], fontsize=12, fontweight="bold")
            ax.set_xlim(0, 1.15)
            style_axis(ax)

        legend_elements = [get_legend_patch("control"), get_legend_patch("regular")]
        axes[-1].legend(handles=legend_elements, loc="lower right", fontsize=9)

        # Title and filename include domain if present
        if domain:
            title = f"Tool Call Success Rate by Action - {domain.capitalize()}"
            filename = f"action_success_{domain}.pdf"
        else:
            title = "Tool Call Success Rate by Action"
            filename = "action_success.pdf"

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / filename, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {output_dir / filename}")


def plot_action_success_summary(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> None:
    """
    Plot overall tool call success rate by LLM.
    Creates one subplot per domain if domain column is present.

    Shows success rate for each LLM with control vs regular comparison.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_raw.empty:
        return

    df_agent = df_raw[df_raw["requestor"] == "assistant"].copy()
    if df_agent.empty:
        return

    llms = sorted(df_agent["llm"].unique())
    complexities = ["control", "regular"]

    # Check if domain column exists and get domains to plot
    has_domain = "domain" in df_agent.columns
    if has_domain:
        tested_domains = df_agent["domain"].unique()
        domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    else:
        domains_to_plot = [None]

    n_domains = len(domains_to_plot)
    if n_domains == 0:
        return

    # Calculate success rate per LLM × [domain] × complexity
    group_cols = ["llm", "speech_complexity"]
    if has_domain:
        group_cols.insert(1, "domain")
    summary = (
        df_agent.groupby(group_cols)
        .agg(
            total_calls=("error", "count"),
            total_errors=("error", "sum"),
        )
        .reset_index()
    )
    summary["success_rate"] = 1 - summary["total_errors"] / summary["total_calls"]
    summary["successful_calls"] = summary["total_calls"] - summary["total_errors"]

    # Create subplots: one per domain (vertical layout)
    fig, axes = plt.subplots(n_domains, 1, figsize=(12, 6 * n_domains), squeeze=False)
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains_to_plot):
        ax = axes[d_idx]

        # Filter data for this domain
        if has_domain and domain is not None:
            domain_summary = summary[summary["domain"] == domain]
        else:
            domain_summary = summary

        x = np.arange(len(llms))
        bar_width = 0.35

        for i, complexity in enumerate(complexities):
            style = get_complexity_style(complexity)
            df_c = domain_summary[domain_summary["speech_complexity"] == complexity]
            values = []
            labels = []

            for llm in llms:
                row = df_c[df_c["llm"] == llm]
                if len(row) > 0:
                    values.append(row.iloc[0]["success_rate"])
                    labels.append(
                        f"{int(row.iloc[0]['successful_calls'])}/{int(row.iloc[0]['total_calls'])}"
                    )
                else:
                    values.append(np.nan)  # No data available for this LLM
                    labels.append("N/A")

            colors = [get_llm_color(llm) for llm in llms]
            ax.bar(
                x + i * bar_width,
                values,
                bar_width,
                color=colors,
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )

            for bar_x, val, label in zip(x + i * bar_width, values, labels):
                if label == "N/A":
                    continue  # Don't show label for missing data
                ax.text(
                    bar_x,
                    val + 0.02,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=0,
                )

        ax.set_xlabel("LLM", fontsize=12)
        ax.set_ylabel("Success Rate", fontsize=12)
        title = f"{domain.capitalize()}" if domain else "Success Rate"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(
            [llm.split(":")[-1][:20] for llm in llms],
            rotation=45,
            ha="right",
            fontsize=10,
        )
        ax.set_ylim(0, 1.1)
        style_axis(ax)

    legend_elements = [get_legend_patch("control"), get_legend_patch("regular")]
    axes[-1].legend(handles=legend_elements, loc="lower right", fontsize=10)

    fig.suptitle(
        "Overall Tool Call Success Rate by LLM", fontsize=14, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "success_summary.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'success_summary.pdf'}")


def save_pass_k_by_persona_raw(
    output_dir: Path,
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """Save raw per-simulation data for persona analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for params, sim_results in results:
        llm = params.get("llm", "unknown")
        domain = params.get("domain", "unknown")
        speech_complexity = params.get("speech_complexity", "unknown")

        if sim_results.simulations is None:
            logger.warning(
                f"Missing simulations for {llm}/{domain}/{speech_complexity}"
            )
            continue

        for sim in sim_results.simulations:
            persona_name = None
            speech_env = getattr(sim, "speech_env_params", None)
            if speech_env and hasattr(speech_env, "persona"):
                persona = speech_env.persona
                if persona:
                    persona_name = getattr(persona, "name", None)

            reward = sim.reward_info.reward if sim.reward_info else 0.0
            rows.append(
                {
                    "simulation_id": sim.id,
                    "task_id": sim.task_id,
                    "trial": sim.trial,
                    "domain": domain,
                    "speech_complexity": speech_complexity,
                    "llm": llm,
                    "persona_name": persona_name,
                    "reward": reward,
                    "success": reward == 1.0,
                }
            )

    df = pd.DataFrame(rows)
    raw_path = output_dir / f"{output_dir.name}_raw.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved: {raw_path}")
    return df


def save_pass_k_by_persona_analysis(
    output_dir: Path,
    df_voice_by_complexity: pd.DataFrame,
) -> pd.DataFrame:
    """Save analysis table for persona pass^k."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_voice_by_complexity.empty:
        return pd.DataFrame()

    analysis_path = output_dir / f"{output_dir.name}_analysis.csv"
    df_voice_by_complexity.to_csv(analysis_path, index=False)
    logger.info(f"Saved: {analysis_path}")
    return df_voice_by_complexity


def plot_pass_k_by_persona(
    fig_dir: Path,
    df_voice: pd.DataFrame,
    df_voice_by_complexity: pd.DataFrame,
    max_k: int = 1,
) -> None:
    """
    Create bar chart showing pass^k for each persona, with control on left
    and regular on right, separated by a dashed vertical line.
    Creates one subplot per domain if domain column is present.

    Args:
        fig_dir: Directory to save figures
        df_voice: DataFrame from compute_pass_k_by_persona (aggregated)
        df_voice_by_complexity: DataFrame from compute_pass_k_by_persona with include_complexity=True
        max_k: Which k value to plot (default: 1 for pass^1)
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    col = f"pass_hat_{max_k}"
    if col not in df_voice_by_complexity.columns:
        logger.warning(f"Column {col} not found in persona metrics. Skipping plot.")
        return

    # Filter out rows with no persona_name
    df = df_voice_by_complexity[df_voice_by_complexity["persona_name"].notna()].copy()

    if len(df) == 0:
        logger.warning("No persona data found. Skipping persona plot.")
        return

    llms = sorted(df["llm"].unique())

    # Check if domain column exists and get domains to plot
    has_domain = "domain" in df.columns
    if has_domain:
        tested_domains = df["domain"].unique()
        domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    else:
        domains_to_plot = [None]  # Single plot without domain filtering

    n_domains = len(domains_to_plot)
    if n_domains == 0:
        logger.warning("No domains found. Skipping persona plot.")
        return

    # Get personas for each complexity (use first domain or all data to determine)
    sample_df = df if not has_domain else df[df["domain"] == domains_to_plot[0]]
    control_personas = (
        sample_df[sample_df["speech_complexity"] == "control"]["persona_name"]
        .unique()
        .tolist()
    )
    regular_personas = (
        sample_df[sample_df["speech_complexity"] == "regular"]["persona_name"]
        .unique()
        .tolist()
    )

    # Build combined x-axis: control personas, then regular personas
    all_personas = control_personas + regular_personas
    n_control = len(control_personas)
    n_regular = len(regular_personas)
    n_total = n_control + n_regular

    if n_total == 0:
        logger.warning("No personas found. Skipping plot.")
        return

    # Create subplots: one per domain (vertical layout)
    fig, axes = plt.subplots(
        n_domains, 1, figsize=(max(14, n_total * 1.2), 6 * n_domains), squeeze=False
    )
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains_to_plot):
        ax = axes[d_idx]

        # Filter data for this domain
        if has_domain and domain is not None:
            domain_df = df[df["domain"] == domain]
        else:
            domain_df = df

        x = np.arange(n_total)
        n_llms = len(llms)
        bar_width = 0.8 / n_llms

        for llm_idx, llm in enumerate(llms):
            llm_color = get_llm_color(llm)
            values = []

            for i, persona in enumerate(all_personas):
                # Determine complexity based on position
                if i < n_control:
                    complexity = "control"
                else:
                    complexity = "regular"

                subset = domain_df[
                    (domain_df["llm"] == llm)
                    & (domain_df["persona_name"] == persona)
                    & (domain_df["speech_complexity"] == complexity)
                ]

                if len(subset) > 0 and col in subset.columns:
                    values.append(subset[col].values[0])  # Keep NaN as-is
                else:
                    values.append(np.nan)  # No data available for this config

            # Calculate bar positions
            bar_offset = (llm_idx - (n_llms - 1) / 2) * bar_width
            x_pos = x + bar_offset

            llm_short = llm.split(":")[-1][:20] if ":" in llm else llm[:20]
            ax.bar(
                x_pos,
                values,
                bar_width,
                color=llm_color,
                alpha=0.85,
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
                label=llm_short if d_idx == 0 else "",
            )

            # Add value labels (show N/A for missing data)
            for bar_x, val in zip(x_pos, values):
                if np.isnan(val):
                    continue  # Don't show label for missing data
                label = f"{val:.2f}"
                ax.text(bar_x, val + 0.02, label, ha="center", va="bottom", fontsize=8)

        # Add dashed vertical separator between control and regular
        if n_control > 0 and n_regular > 0:
            separator_x = n_control - 0.5
            ax.axvline(x=separator_x, color="gray", linestyle="--", linewidth=1.5)

            # Add complexity labels at top
            ax.text(
                (n_control - 1) / 2,
                1.08,
                "Control",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
            ax.text(
                n_control + (n_regular - 1) / 2,
                1.08,
                "Regular",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlabel("Persona", fontsize=11)
        ax.set_ylabel(f"Pass^{max_k}", fontsize=11)
        title = f"{domain.capitalize()}" if domain else f"Pass^{max_k} by Persona"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(all_personas, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        style_axis(ax)

    # Add legend from first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.95),
            fontsize=9,
            title="LLM",
        )

    fig.suptitle(f"Pass^{max_k} by Persona", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    filename = f"pass_{max_k}.pdf"
    plt.savefig(fig_dir / filename, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved: {fig_dir / filename}")


def save_pass_k_by_background_noise_raw(
    output_dir: Path,
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """Save raw per-simulation data for background noise analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for params, sim_results in results:
        llm = params.get("llm", "unknown")
        domain = params.get("domain", "unknown")
        speech_complexity = params.get("speech_complexity", "unknown")

        if sim_results.simulations is None:
            logger.warning(
                f"Missing simulations for {llm}/{domain}/{speech_complexity}"
            )
            continue

        for sim in sim_results.simulations:
            background_noise_file = None
            speech_env = getattr(sim, "speech_env_params", None)
            if speech_env and hasattr(speech_env, "background_audio_file"):
                background_noise_file = speech_env.background_audio_file

            reward = sim.reward_info.reward if sim.reward_info else 0.0
            rows.append(
                {
                    "simulation_id": sim.id,
                    "task_id": sim.task_id,
                    "trial": sim.trial,
                    "domain": domain,
                    "speech_complexity": speech_complexity,
                    "llm": llm,
                    "background_noise_file": background_noise_file,
                    "reward": reward,
                    "success": reward == 1.0,
                }
            )

    df = pd.DataFrame(rows)
    raw_path = output_dir / f"{output_dir.name}_raw.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved: {raw_path}")
    return df


def save_pass_k_by_background_noise_analysis(
    output_dir: Path,
    df_noise_by_complexity: pd.DataFrame,
) -> pd.DataFrame:
    """Save analysis table for background noise pass^k."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_noise_by_complexity.empty:
        return pd.DataFrame()

    analysis_path = output_dir / f"{output_dir.name}_analysis.csv"
    df_noise_by_complexity.to_csv(analysis_path, index=False)
    logger.info(f"Saved: {analysis_path}")
    return df_noise_by_complexity


def plot_pass_k_by_background_noise(
    fig_dir: Path,
    df_noise: pd.DataFrame,
    df_noise_by_complexity: pd.DataFrame,
    max_k: int = 1,
) -> None:
    """
    Create bar chart showing pass^k for each background noise file, with control on left
    and regular on right, separated by a dashed vertical line.
    Creates one subplot per domain if domain column is present.

    Args:
        fig_dir: Directory to save figures
        df_noise: DataFrame from compute_pass_k_by_background_noise (aggregated)
        df_noise_by_complexity: DataFrame from compute_pass_k_by_background_noise with include_complexity=True
        max_k: Which k value to plot (default: 1 for pass^1)
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    col = f"pass_hat_{max_k}"
    if col not in df_noise_by_complexity.columns:
        logger.warning(
            f"Column {col} not found in background noise metrics. Skipping plot."
        )
        return

    # Handle None/NaN background noise (displayed as "No Noise")
    df = df_noise_by_complexity.copy()
    df["background_noise_display"] = df["background_noise_file"].apply(
        lambda x: Path(x).stem if pd.notna(x) and x else "No Noise"
    )

    llms = sorted(df["llm"].unique())

    # Check if domain column exists and get domains to plot
    has_domain = "domain" in df.columns
    if has_domain:
        tested_domains = df["domain"].unique()
        domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    else:
        domains_to_plot = [None]  # Single plot without domain filtering

    n_domains = len(domains_to_plot)
    if n_domains == 0:
        logger.warning("No domains found. Skipping background noise plot.")
        return

    # Get noise files for each complexity (use first domain or all data to determine)
    sample_df = df if not has_domain else df[df["domain"] == domains_to_plot[0]]
    control_noise = (
        sample_df[sample_df["speech_complexity"] == "control"][
            "background_noise_display"
        ]
        .unique()
        .tolist()
    )
    regular_noise = (
        sample_df[sample_df["speech_complexity"] == "regular"][
            "background_noise_display"
        ]
        .unique()
        .tolist()
    )

    # Sort each list (No Noise first)
    control_noise = sorted(control_noise, key=lambda x: (x != "No Noise", x))
    regular_noise = sorted(regular_noise, key=lambda x: (x != "No Noise", x))

    # Build combined x-axis: control noises, then regular noises
    all_noises = control_noise + regular_noise
    n_control = len(control_noise)
    n_regular = len(regular_noise)
    n_total = n_control + n_regular

    if n_total == 0:
        logger.warning("No background noise data found. Skipping plot.")
        return

    # Create subplots: one per domain (vertical layout)
    fig, axes = plt.subplots(
        n_domains, 1, figsize=(max(14, n_total * 1.2), 6 * n_domains), squeeze=False
    )
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains_to_plot):
        ax = axes[d_idx]

        # Filter data for this domain
        if has_domain and domain is not None:
            domain_df = df[df["domain"] == domain]
        else:
            domain_df = df

        x = np.arange(n_total)
        n_llms = len(llms)
        bar_width = 0.8 / n_llms

        for llm_idx, llm in enumerate(llms):
            llm_color = get_llm_color(llm)
            values = []

            for i, noise in enumerate(all_noises):
                # Determine complexity based on position
                if i < n_control:
                    complexity = "control"
                else:
                    complexity = "regular"

                subset = domain_df[
                    (domain_df["llm"] == llm)
                    & (domain_df["background_noise_display"] == noise)
                    & (domain_df["speech_complexity"] == complexity)
                ]

                if len(subset) > 0 and col in subset.columns:
                    values.append(subset[col].values[0])  # Keep NaN as-is
                else:
                    values.append(np.nan)  # No data available for this config

            # Calculate bar positions
            bar_offset = (llm_idx - (n_llms - 1) / 2) * bar_width
            x_pos = x + bar_offset

            llm_short = llm.split(":")[-1][:20] if ":" in llm else llm[:20]
            ax.bar(
                x_pos,
                values,
                bar_width,
                color=llm_color,
                alpha=0.85,
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
                label=llm_short if d_idx == 0 else "",
            )

            # Add value labels (show N/A for missing data)
            for bar_x, val in zip(x_pos, values):
                if np.isnan(val):
                    continue  # Don't show label for missing data
                label = f"{val:.2f}"
                ax.text(bar_x, val + 0.02, label, ha="center", va="bottom", fontsize=8)

        # Add dashed vertical separator between control and regular
        if n_control > 0 and n_regular > 0:
            separator_x = n_control - 0.5
            ax.axvline(x=separator_x, color="gray", linestyle="--", linewidth=1.5)

            # Add complexity labels at top
            ax.text(
                (n_control - 1) / 2,
                1.08,
                "Control",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
            ax.text(
                n_control + (n_regular - 1) / 2,
                1.08,
                "Regular",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlabel("Background Noise", fontsize=11)
        ax.set_ylabel(f"Pass^{max_k}", fontsize=11)
        title = (
            f"{domain.capitalize()}" if domain else f"Pass^{max_k} by Background Noise"
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(all_noises, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        style_axis(ax)

    # Add legend from first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.95),
            fontsize=9,
            title="LLM",
        )

    fig.suptitle(f"Pass^{max_k} by Background Noise", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    filename = f"pass_{max_k}.pdf"
    plt.savefig(fig_dir / filename, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved: {fig_dir / filename}")


def plot_simulation_counts(
    fig_dir: Path,
    results: List[Tuple[dict, Results]],
) -> None:
    """
    Create heatmaps showing simulation counts per LLM × domain × complexity × background_noise × persona.

    Creates a single multi-page PDF with one page per (LLM, domain) combination.
    Each page has subplots for each complexity level.
    Rows = voice names, Columns = background noise files, Cell values = simulation counts.

    Args:
        fig_dir: Directory to save figures
        results: List of (params, Results) tuples
    """
    from matplotlib.backends.backend_pdf import PdfPages

    # Build simulation-level dataframe
    df_sims = build_simulation_level_dataframe(results)

    if len(df_sims) == 0:
        logger.warning("No simulation data found. Skipping simulation counts plot.")
        return

    # Get persona display name
    def get_persona_label(name):
        if name is None or pd.isna(name):
            return "None"
        if name in ALL_PERSONAS:
            return ALL_PERSONAS[name].display_name
        return str(name)

    # Get background noise display name
    def get_noise_label(noise_file):
        if noise_file is None or pd.isna(noise_file):
            return "No Noise"
        return Path(noise_file).stem

    # Add display columns
    df_sims["persona_display"] = df_sims["persona_name"].apply(get_persona_label)
    df_sims["noise_name"] = df_sims["background_noise_file"].apply(get_noise_label)

    # Get unique values for axes
    llms = df_sims["llm"].unique()
    domains = [d for d in DOMAINS if d in df_sims["domain"].unique()]
    complexities = [
        c for c in SPEECH_COMPLEXITIES if c in df_sims["speech_complexity"].unique()
    ]

    # Get all unique voice names and noise names (sorted)
    all_voice_names = sorted(
        df_sims["voice_name"].unique(), key=lambda x: (x == "None", x)
    )
    all_noise_names = sorted(
        df_sims["noise_name"].unique(),
        key=lambda x: (x != "No Noise", x),  # Put "No Noise" first
    )

    # Create a single multi-page PDF for all (LLM, domain) combinations
    pdf_path = fig_dir / "simulation_counts.pdf"
    with PdfPages(pdf_path) as pdf:
        page_count = 0

        for llm in llms:
            llm_df = df_sims[df_sims["llm"] == llm]

            for domain in domains:
                domain_df = llm_df[llm_df["domain"] == domain]

                if len(domain_df) == 0:
                    continue

                n_complexities = len(complexities)
                fig, axes = plt.subplots(
                    1,
                    n_complexities + 1,  # Extra column for colorbar
                    figsize=(
                        5 * n_complexities + 0.5,
                        max(4, 0.5 * len(all_voice_names)),
                    ),
                    squeeze=False,
                    gridspec_kw={"width_ratios": [1] * n_complexities + [0.05]},
                )
                axes = axes[0]
                cbar_ax = axes[-1]
                plot_axes = axes[:-1]

                # Track max count for consistent colorbar
                max_count = 0

                # First pass: compute max count
                for complexity in complexities:
                    subset = domain_df[domain_df["speech_complexity"] == complexity]
                    counts = subset.groupby(["voice_name", "noise_name"]).size()
                    if len(counts) > 0:
                        max_count = max(max_count, counts.max())

                if max_count == 0:
                    max_count = 1  # Avoid division by zero

                im = None
                for c_idx, complexity in enumerate(complexities):
                    ax = plot_axes[c_idx]
                    subset = domain_df[domain_df["speech_complexity"] == complexity]

                    # Build count matrix
                    matrix = np.zeros((len(all_voice_names), len(all_noise_names)))
                    counts = subset.groupby(["voice_name", "noise_name"]).size()

                    for i, voice_name in enumerate(all_voice_names):
                        for j, noise_name in enumerate(all_noise_names):
                            if (voice_name, noise_name) in counts.index:
                                matrix[i, j] = counts[(voice_name, noise_name)]

                    # Create heatmap
                    im = ax.imshow(
                        matrix,
                        cmap="Blues",
                        aspect="auto",
                        vmin=0,
                        vmax=max_count,
                    )

                    # Add count annotations
                    for i in range(len(all_voice_names)):
                        for j in range(len(all_noise_names)):
                            count = int(matrix[i, j])
                            if count > 0:
                                text_color = (
                                    "white" if count > max_count * 0.6 else "black"
                                )
                                ax.text(
                                    j,
                                    i,
                                    str(count),
                                    ha="center",
                                    va="center",
                                    fontsize=9,
                                    color=text_color,
                                    fontweight="bold",
                                )

                    ax.set_xticks(np.arange(len(all_noise_names)))
                    ax.set_xticklabels(
                        all_noise_names, fontsize=8, rotation=45, ha="right"
                    )
                    ax.set_yticks(np.arange(len(all_voice_names)))
                    ax.set_yticklabels(all_voice_names, fontsize=9)
                    ax.set_xlabel("Background Noise", fontsize=10)
                    if c_idx == 0:
                        ax.set_ylabel("Voice", fontsize=10)
                    ax.set_title(get_complexity_display_name(complexity), fontsize=11)

                fig.suptitle(
                    f"Simulation Counts - {llm} - {domain.capitalize()}", fontsize=14
                )
                if im is not None:
                    fig.colorbar(im, cax=cbar_ax, label="Count")
                plt.tight_layout()

                # Save to PDF
                pdf.savefig(fig, bbox_inches="tight", dpi=300)
                plt.close(fig)
                page_count += 1

        logger.info(f"Saved: simulation_counts.pdf ({page_count} pages)")

    # Also save a summary CSV with all counts (including LLM)
    counts_df = (
        df_sims.groupby(
            ["llm", "domain", "speech_complexity", "voice_name", "noise_name"]
        )
        .size()
        .reset_index(name="simulation_count")
    )
    counts_csv_path = fig_dir / "simulation_counts.csv"
    counts_df.to_csv(counts_csv_path, index=False)
    logger.info(f"Saved simulation counts to: {counts_csv_path}")


def plot_task_pass_fail_grid(
    fig_dir: Path,
    results: List[Tuple[dict, Results]],
) -> None:
    """
    Create a grid visualization showing pass/fail status for each task.

    Creates a single multi-page PDF with one page per domain.
    Each page shows a heatmap where:
    - Rows = (LLM, complexity, trial) combinations
    - Columns = tasks
    - Colors: green = passed, red = failed, grey = not run

    Args:
        fig_dir: Directory to save figures
        results: List of (params, Results) tuples
    """
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    # Build simulation-level dataframe
    df_sims = build_simulation_level_dataframe(results)

    if len(df_sims) == 0:
        logger.warning("No simulation data found. Skipping task pass/fail grid.")
        return

    # Get unique values
    llms = sorted(df_sims["llm"].unique())
    domains = [d for d in DOMAINS if d in df_sims["domain"].unique()]
    complexities = [
        c for c in SPEECH_COMPLEXITIES if c in df_sims["speech_complexity"].unique()
    ]

    # Custom colormap: -1 = grey (not run), 0 = red (fail), 1 = green (pass)
    colors = ["#E0E0E0", "#E15759", "#59A14F"]  # Grey, Red, Green
    cmap = ListedColormap(colors)

    # Shorten task names for display
    def shorten_task_name(task_id: str) -> str:
        # Extract the core task name, removing common prefixes
        name = task_id
        if name.startswith("task_"):
            name = name[5:]
        # Truncate if too long
        if len(name) > 25:
            name = name[:22] + "..."
        return name

    # Shorten LLM names for display
    def shorten_llm_name(llm: str) -> str:
        # Extract just the model name after the provider
        if ":" in llm:
            return llm.split(":")[1][:20]
        return llm[:20]

    # Create a single multi-page PDF (one page per domain)
    pdf_path = fig_dir / "task_pass_fail_grid.pdf"
    with PdfPages(pdf_path) as pdf:
        page_count = 0

        for domain in domains:
            domain_df = df_sims[df_sims["domain"] == domain]

            if len(domain_df) == 0:
                continue

            # Get all unique tasks for this domain (sorted)
            all_tasks = sorted(domain_df["task_id"].unique())
            task_labels = [shorten_task_name(t) for t in all_tasks]

            # Get all unique trials
            all_trials = sorted(domain_df["trial"].unique())

            # Build row labels: (LLM, complexity, trial) tuples
            row_configs = []
            for llm in llms:
                for complexity in complexities:
                    for trial in all_trials:
                        row_configs.append((llm, complexity, trial))

            if len(row_configs) == 0:
                continue

            # Build the matrix: rows = (LLM, complexity, trial), cols = tasks
            # Values: -1 = not run, 0 = fail, 1 = pass
            matrix = np.full((len(row_configs), len(all_tasks)), -1, dtype=float)

            for row_idx, (llm, complexity, trial) in enumerate(row_configs):
                subset = domain_df[
                    (domain_df["llm"] == llm)
                    & (domain_df["speech_complexity"] == complexity)
                    & (domain_df["trial"] == trial)
                ]
                for col_idx, task_id in enumerate(all_tasks):
                    task_subset = subset[subset["task_id"] == task_id]
                    if len(task_subset) > 0:
                        # Take the first match (should be unique)
                        success = task_subset["success"].values[0]
                        if success is None or pd.isna(success):
                            # Missing reward_info - show as grey (not evaluated)
                            matrix[row_idx, col_idx] = -1
                        else:
                            matrix[row_idx, col_idx] = 1 if success else 0

            # Create figure
            n_tasks = len(all_tasks)
            n_rows = len(row_configs)

            # Calculate figure size based on content
            fig_width = max(14, 0.4 * n_tasks + 4)
            fig_height = max(6, 0.3 * n_rows + 2)

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # Create heatmap
            ax.imshow(
                matrix,
                cmap=cmap,
                aspect="auto",
                vmin=-1,
                vmax=1,
                interpolation="nearest",
            )

            # Add grid lines
            ax.set_xticks(np.arange(n_tasks + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)

            # Add horizontal separator lines between LLMs
            rows_per_llm = len(complexities) * len(all_trials)
            for i in range(1, len(llms)):
                ax.axhline(
                    y=i * rows_per_llm - 0.5,
                    color="#333333",
                    linestyle="-",
                    linewidth=2,
                )

            # Row labels with complexity color coding
            row_labels = []
            row_colors = []
            for i, (llm, complexity, trial) in enumerate(row_configs):
                # Add LLM name only for first row of each LLM group
                if i % rows_per_llm == 0:
                    llm_short = shorten_llm_name(llm)
                    row_labels.append(
                        f"{llm_short} | {get_complexity_display_name(complexity)} T{trial}"
                    )
                else:
                    row_labels.append(
                        f"{get_complexity_display_name(complexity)} T{trial}"
                    )
                row_colors.append(SPEECH_COMPLEXITY_COLORS.get(complexity, "black"))

            ax.set_yticks(np.arange(n_rows))
            ax.set_yticklabels(row_labels, fontsize=7)

            # Color the y-axis labels by complexity
            for tick_label, color in zip(ax.get_yticklabels(), row_colors):
                tick_label.set_color(color)
                tick_label.set_fontweight("bold")

            # Task labels on x-axis
            ax.set_xticks(np.arange(n_tasks))
            ax.set_xticklabels(task_labels, fontsize=6, rotation=45, ha="right")

            # Labels
            ax.set_xlabel("Task", fontsize=11)
            ax.set_ylabel("LLM / Complexity / Trial", fontsize=11)

            # Add legend
            legend_elements = [
                Patch(facecolor="#59A14F", edgecolor="white", label="Pass"),
                Patch(facecolor="#E15759", edgecolor="white", label="Fail"),
                Patch(facecolor="#E0E0E0", edgecolor="white", label="Not Run"),
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                fontsize=9,
            )

            # Title
            fig.suptitle(
                f"Task Pass/Fail Grid - {domain.capitalize()}",
                fontsize=14,
                fontweight="bold",
            )

            plt.tight_layout()

            # Save to PDF
            pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close(fig)
            page_count += 1

    logger.info(f"Saved: task_pass_fail_grid.pdf ({page_count} pages)")


# =============================================================================
# Termination Analysis
# =============================================================================

# Color palette for termination reasons
TERMINATION_REASON_COLORS = {
    "user_stop": "#59A14F",  # Green - successful completion
    "agent_stop": "#4E79A7",  # Blue - agent decided to stop
    "max_steps": "#F28E2B",  # Orange - ran out of steps
    "too_many_errors": "#E15759",  # Red - errors
    "agent_error": "#B07AA1",  # Purple - agent error
    "user_error": "#FF9DA7",  # Pink - user error
}

TERMINATION_FALLBACK_COLORS = ["#76B7B2", "#EDC948", "#BAB0AC", "#9C755F"]


def get_termination_reason_color(reason: str, all_reasons: list) -> str:
    """Get color for a termination reason."""
    if reason in TERMINATION_REASON_COLORS:
        return TERMINATION_REASON_COLORS[reason]
    idx = list(all_reasons).index(reason) % len(TERMINATION_FALLBACK_COLORS)
    return TERMINATION_FALLBACK_COLORS[idx]


def save_termination_raw(
    output_dir: Path,
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """
    Save raw per-simulation termination data.

    Args:
        output_dir: Directory for output files
        results: List of (params, Results) tuples

    Returns:
        DataFrame with per-simulation termination data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df_sims = build_simulation_level_dataframe(results)

    if df_sims.empty:
        logger.warning("No simulation data found for termination analysis.")
        return df_sims

    # Select relevant columns for raw data
    raw_columns = [
        "llm",
        "domain",
        "speech_complexity",
        "simulation_id",
        "task_id",
        "trial",
        "termination_reason",
        "reward",
        "success",
    ]
    available_cols = [c for c in raw_columns if c in df_sims.columns]
    df_raw = df_sims[available_cols].copy()

    raw_path = output_dir / f"{output_dir.name}_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    logger.info(f"Saved: {raw_path}")

    return df_raw


def save_termination_analysis(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Save aggregated termination analysis.

    Args:
        output_dir: Directory for output files
        df_raw: Raw termination data

    Returns:
        DataFrame with aggregated analysis
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_raw.empty or "termination_reason" not in df_raw.columns:
        return pd.DataFrame()

    # Check if domain column exists
    has_domain = "domain" in df_raw.columns
    group_cols = ["llm"]
    if has_domain:
        group_cols.append("domain")
    group_cols.extend(["speech_complexity", "termination_reason"])

    # Counts by LLM × [domain] × complexity × reason
    counts_df = df_raw.groupby(group_cols).size().reset_index(name="count")

    # Add totals and proportions
    total_group_cols = group_cols[:-1]  # Exclude termination_reason
    totals = df_raw.groupby(total_group_cols).size().reset_index(name="total")
    counts_df = counts_df.merge(totals, on=total_group_cols)
    counts_df["proportion"] = (counts_df["count"] / counts_df["total"]).round(4)

    analysis_path = output_dir / f"{output_dir.name}_analysis.csv"
    counts_df.to_csv(analysis_path, index=False)
    logger.info(f"Saved: {analysis_path}")

    # Create pivoted summary table (aggregate across domains for summary)
    pivot_cols = ["llm", "speech_complexity"]
    if has_domain:
        pivot_cols.insert(1, "domain")
    pivot_df = counts_df.pivot_table(
        index="termination_reason",
        columns=pivot_cols,
        values="proportion",
        fill_value=0,
    )
    # Flatten column names
    if has_domain:
        pivot_df.columns = [
            f"{llm.split(':')[-1][:10]}_{dom}_{comp}"
            for llm, dom, comp in pivot_df.columns
        ]
    else:
        pivot_df.columns = [
            f"{llm.split(':')[-1][:15]} ({comp})" for llm, comp in pivot_df.columns
        ]
    pivot_df["OVERALL"] = counts_df.groupby("termination_reason")["count"].sum() / len(
        df_raw
    )
    pivot_df = pivot_df.sort_values("OVERALL", ascending=False)

    summary_path = output_dir / "summary.csv"
    pivot_df.to_csv(summary_path)
    logger.info(f"Saved: {summary_path}")

    return counts_df


def plot_termination_reasons(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> None:
    """
    Create bar chart showing termination reasons by LLM and complexity.
    Creates one subplot per domain if domain column is present.

    Args:
        output_dir: Directory to save figures
        df_raw: Raw termination data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_raw.empty or "termination_reason" not in df_raw.columns:
        return

    llms = sorted(df_raw["llm"].unique())
    complexities = ["control", "regular"]
    all_reasons = sorted(df_raw["termination_reason"].unique())

    # Check if domain column exists and get domains to plot
    has_domain = "domain" in df_raw.columns
    if has_domain:
        tested_domains = df_raw["domain"].unique()
        domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    else:
        domains_to_plot = [None]

    n_domains = len(domains_to_plot)
    if n_domains == 0:
        return

    # Create subplots: one per domain (vertical layout)
    fig, axes = plt.subplots(n_domains, 1, figsize=(12, 6 * n_domains), squeeze=False)
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains_to_plot):
        ax = axes[d_idx]

        # Filter data for this domain
        if has_domain and domain is not None:
            domain_df = df_raw[df_raw["domain"] == domain]
        else:
            domain_df = df_raw

        x = np.arange(len(llms))
        bar_width = 0.35

        for i, complexity in enumerate(complexities):
            style = get_complexity_style(complexity)
            bottom = np.zeros(len(llms))

            for reason in all_reasons:
                values = []
                for llm in llms:
                    subset = domain_df[
                        (domain_df["llm"] == llm)
                        & (domain_df["speech_complexity"] == complexity)
                    ]
                    total = len(subset)
                    if total > 0:
                        reason_count = len(
                            subset[subset["termination_reason"] == reason]
                        )
                        values.append(reason_count / total)
                    else:
                        values.append(np.nan)  # No data available

                ax.bar(
                    x + i * bar_width,
                    values,
                    bar_width,
                    bottom=bottom,
                    color=get_termination_reason_color(reason, all_reasons),
                    alpha=style["alpha"],
                    hatch=style["hatch"],
                    edgecolor=BAR_STYLE["edgecolor"],
                    linewidth=BAR_STYLE["linewidth"],
                )
                bottom += np.array(values)

        ax.set_xlabel("LLM", fontsize=12)
        ax.set_ylabel("Proportion", fontsize=12)
        title = f"{domain.capitalize()}" if domain else "Termination Reasons"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(
            [llm.split(":")[-1][:20] for llm in llms],
            rotation=45,
            ha="right",
            fontsize=10,
        )
        ax.set_ylim(0, 1.05)
        style_axis(ax)

    # Custom legend (add to last subplot)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=get_termination_reason_color(r, all_reasons),
            label=r.replace("_", " ").title(),
        )
        for r in all_reasons
    ] + [
        get_legend_patch("control", facecolor="gray"),
        get_legend_patch("regular", facecolor="gray"),
    ]
    axes[-1].legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
    )

    fig.suptitle(
        "Termination Reasons by LLM and Speech Complexity",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    plt.savefig(
        output_dir / "termination_reasons.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()
    logger.info(f"Saved: {output_dir / 'termination_reasons.pdf'}")


# =============================================================================
# Duration Analysis
# =============================================================================


def save_duration_raw(
    output_dir: Path,
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """
    Save raw per-simulation duration data.

    Args:
        output_dir: Directory for output files
        results: List of (params, Results) tuples

    Returns:
        DataFrame with per-simulation duration data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for params, sim_results in results:
        if sim_results.simulations is None:
            logger.warning(
                f"Missing simulations for {params.get('llm', '?')}/{params.get('domain', '?')}/{params.get('speech_complexity', '?')}"
            )
            continue
        for sim in sim_results.simulations:
            num_ticks = len(sim.ticks) if sim.ticks else 0
            tick_duration = None
            if sim.ticks:
                tick_duration = sim.ticks[0].tick_duration_seconds
            rows.append(
                {
                    "llm": params["llm"],
                    "domain": params["domain"],
                    "speech_complexity": params["speech_complexity"],
                    "simulation_id": sim.id,
                    "task_id": sim.task_id,
                    "trial": sim.trial,
                    "duration_seconds": sim.duration,
                    "num_ticks": num_ticks,
                    "tick_duration_seconds": tick_duration,
                    "termination_reason": (
                        sim.termination_reason.value
                        if hasattr(sim.termination_reason, "value")
                        else str(sim.termination_reason)
                    ),
                    "success": (
                        sim.reward_info.reward >= 1.0 - 1e-6
                        if sim.reward_info
                        else False
                    ),
                }
            )

    df_raw = pd.DataFrame(rows)

    if df_raw.empty:
        logger.warning("No duration data found.")
        return df_raw

    raw_path = output_dir / f"{output_dir.name}_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    logger.info(f"Saved: {raw_path}")

    return df_raw


def save_duration_analysis(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Save aggregated duration analysis.

    Args:
        output_dir: Directory for output files
        df_raw: Raw duration data

    Returns:
        DataFrame with aggregated analysis
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_raw.empty:
        return pd.DataFrame()

    # Check if domain column exists
    has_domain = "domain" in df_raw.columns
    group_cols = ["llm"]
    if has_domain:
        group_cols.append("domain")
    group_cols.append("speech_complexity")

    # Aggregate by LLM × [domain] × complexity
    analysis = (
        df_raw.groupby(group_cols)
        .agg(
            num_simulations=("simulation_id", "count"),
            mean_duration=("duration_seconds", "mean"),
            std_duration=("duration_seconds", "std"),
            min_duration=("duration_seconds", "min"),
            max_duration=("duration_seconds", "max"),
            median_duration=("duration_seconds", "median"),
            mean_ticks=("num_ticks", "mean"),
        )
        .reset_index()
    )

    # Round for readability
    for col in [
        "mean_duration",
        "std_duration",
        "min_duration",
        "max_duration",
        "median_duration",
        "mean_ticks",
    ]:
        if col in analysis.columns:
            analysis[col] = analysis[col].round(2)

    analysis_path = output_dir / f"{output_dir.name}_analysis.csv"
    analysis.to_csv(analysis_path, index=False)
    logger.info(f"Saved: {analysis_path}")

    # Create pivoted summary
    pivot_cols = ["speech_complexity"]
    if has_domain:
        pivot_cols.insert(0, "domain")
    pivot_df = df_raw.pivot_table(
        index="llm",
        columns=pivot_cols,
        values="duration_seconds",
        aggfunc=["mean", "std"],
    ).round(2)
    if has_domain:
        pivot_df.columns = [
            f"{agg}_{dom}_{comp}" for agg, dom, comp in pivot_df.columns
        ]
    else:
        pivot_df.columns = [f"{agg}_{comp}" for agg, comp in pivot_df.columns]

    summary_path = output_dir / "summary.csv"
    pivot_df.to_csv(summary_path)
    logger.info(f"Saved: {summary_path}")

    return analysis


def plot_duration(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> None:
    """
    Plot conversation duration by LLM and speech complexity.
    Creates one subplot per domain if domain column is present.

    Args:
        output_dir: Directory to save figures
        df_raw: Raw duration data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_raw.empty:
        return

    llms = sorted(df_raw["llm"].unique())
    complexities = ["control", "regular"]

    # Check if domain column exists and get domains to plot
    has_domain = "domain" in df_raw.columns
    if has_domain:
        tested_domains = df_raw["domain"].unique()
        domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    else:
        domains_to_plot = [None]

    n_domains = len(domains_to_plot)
    if n_domains == 0:
        return

    # Aggregate for plotting
    group_cols = ["llm", "speech_complexity"]
    if has_domain:
        group_cols.insert(1, "domain")
    analysis = (
        df_raw.groupby(group_cols)
        .agg(
            mean_duration=("duration_seconds", "mean"),
            std_duration=("duration_seconds", "std"),
        )
        .reset_index()
    )

    # Create subplots: one per domain (vertical layout)
    fig, axes = plt.subplots(n_domains, 1, figsize=(12, 6 * n_domains), squeeze=False)
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains_to_plot):
        ax = axes[d_idx]

        # Filter data for this domain
        if has_domain and domain is not None:
            domain_analysis = analysis[analysis["domain"] == domain]
        else:
            domain_analysis = analysis

        x = np.arange(len(llms))
        bar_width = 0.35

        for i, complexity in enumerate(complexities):
            style = get_complexity_style(complexity)
            df_c = domain_analysis[domain_analysis["speech_complexity"] == complexity]

            means = []
            stds = []
            for llm in llms:
                row = df_c[df_c["llm"] == llm]
                if len(row) > 0:
                    means.append(row.iloc[0]["mean_duration"])
                    stds.append(
                        row.iloc[0]["std_duration"]
                        if pd.notna(row.iloc[0]["std_duration"])
                        else 0
                    )
                else:
                    means.append(0)
                    stds.append(0)

            colors = [get_llm_color(llm) for llm in llms]
            ax.bar(
                x + i * bar_width,
                means,
                bar_width,
                yerr=stds,
                capsize=3,
                color=colors,
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
                error_kw={"elinewidth": 1, "capthick": 1},
            )

            # Add value labels
            for bar_x, mean in zip(x + i * bar_width, means):
                if mean > 0:
                    ax.text(
                        bar_x,
                        mean + 5,
                        f"{mean:.0f}s",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        ax.set_xlabel("LLM", fontsize=12)
        ax.set_ylabel("Duration (seconds)", fontsize=12)
        title = f"{domain.capitalize()}" if domain else "Duration"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(
            [llm.split(":")[-1][:20] for llm in llms],
            rotation=45,
            ha="right",
            fontsize=10,
        )
        style_axis(ax)

    legend_elements = [get_legend_patch("control"), get_legend_patch("regular")]
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=10)

    fig.suptitle(
        "Average Conversation Duration by LLM and Speech Complexity",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "duration.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'duration.pdf'}")


def plot_duration_distribution(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> None:
    """
    Plot duration distribution (box plot) by LLM and speech complexity.
    Creates one subplot per domain if domain column is present.

    Args:
        output_dir: Directory to save figures
        df_raw: Raw duration data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_raw.empty:
        return

    llms = sorted(df_raw["llm"].unique())
    complexities = ["control", "regular"]

    # Check if domain column exists and get domains to plot
    has_domain = "domain" in df_raw.columns
    if has_domain:
        tested_domains = df_raw["domain"].unique()
        domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    else:
        domains_to_plot = [None]

    n_domains = len(domains_to_plot)
    if n_domains == 0:
        return

    # Create subplots: one per domain (vertical layout)
    fig, axes = plt.subplots(n_domains, 1, figsize=(14, 6 * n_domains), squeeze=False)
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains_to_plot):
        ax = axes[d_idx]

        # Filter data for this domain
        if has_domain and domain is not None:
            domain_df = df_raw[df_raw["domain"] == domain]
        else:
            domain_df = df_raw

        positions = []
        data = []
        colors = []
        labels = []

        pos = 0
        for llm in llms:
            for complexity in complexities:
                subset = domain_df[
                    (domain_df["llm"] == llm)
                    & (domain_df["speech_complexity"] == complexity)
                ]
                if not subset.empty:
                    data.append(subset["duration_seconds"].values)
                    positions.append(pos)
                    colors.append(get_llm_color(llm))
                    labels.append(f"{llm.split(':')[-1][:12]}\n({complexity[:4]})")
                pos += 1
            pos += 0.5  # Gap between LLMs

        if data:
            bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)

            for patch, color, pos_idx in zip(bp["boxes"], colors, range(len(data))):
                complexity = "control" if pos_idx % 2 == 0 else "regular"
                style = get_complexity_style(complexity)
                patch.set_facecolor(color)
                patch.set_alpha(style["alpha"])
                patch.set_edgecolor(BAR_STYLE["edgecolor"])
                if style["hatch"]:
                    patch.set_hatch(style["hatch"])

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=8, rotation=0)

        ax.set_ylabel("Duration (seconds)", fontsize=12)
        title = f"{domain.capitalize()}" if domain else "Duration Distribution"
        ax.set_title(title, fontsize=13, fontweight="bold")
        style_axis(ax)

    legend_elements = [get_legend_patch("control"), get_legend_patch("regular")]
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=10)

    fig.suptitle(
        "Conversation Duration Distribution by LLM and Speech Complexity",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        output_dir / "duration_distribution.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()
    logger.info(f"Saved: {output_dir / 'duration_distribution.pdf'}")


def save_duration_by_outcome_analysis(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Save duration analysis broken down by success/failure and complexity.

    Args:
        output_dir: Directory for output files
        df_raw: Raw duration data

    Returns:
        DataFrame with aggregated analysis
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_raw.empty:
        return pd.DataFrame()

    # Check if domain column exists
    has_domain = "domain" in df_raw.columns
    group_cols = ["llm"]
    if has_domain:
        group_cols.append("domain")
    group_cols.extend(["speech_complexity", "success"])

    # Aggregate by LLM × [domain] × complexity × success
    analysis = (
        df_raw.groupby(group_cols)
        .agg(
            num_simulations=("simulation_id", "count"),
            mean_duration=("duration_seconds", "mean"),
            std_duration=("duration_seconds", "std"),
            median_duration=("duration_seconds", "median"),
        )
        .reset_index()
    )

    # Round for readability
    for col in ["mean_duration", "std_duration", "median_duration"]:
        if col in analysis.columns:
            analysis[col] = analysis[col].round(2)

    # Convert success to readable label
    analysis["outcome"] = analysis["success"].map(
        {True: "success", False: "failure", None: "not_evaluated"}
    )

    analysis_path = output_dir / "analysis_by_outcome.csv"
    analysis.to_csv(analysis_path, index=False)
    logger.info(f"Saved: {analysis_path}")

    return analysis


def plot_duration_by_outcome(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> None:
    """
    Plot duration by LLM, complexity, and success/failure outcome.
    Creates a grid of subplots: rows = domains, columns = complexities.

    Args:
        output_dir: Directory to save figures
        df_raw: Raw duration data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_raw.empty:
        return

    llms = sorted(df_raw["llm"].unique())
    complexities = ["control", "regular"]
    outcomes = [True, False]  # success, failure

    # Check if domain column exists and get domains to plot
    has_domain = "domain" in df_raw.columns
    if has_domain:
        tested_domains = df_raw["domain"].unique()
        domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    else:
        domains_to_plot = [None]

    n_domains = len(domains_to_plot)
    n_complexities = len(complexities)

    if n_domains == 0:
        return

    # Aggregate for plotting
    group_cols = ["llm", "speech_complexity", "success"]
    if has_domain:
        group_cols.insert(1, "domain")
    analysis = (
        df_raw.groupby(group_cols)
        .agg(
            mean_duration=("duration_seconds", "mean"),
            std_duration=("duration_seconds", "std"),
            count=("simulation_id", "count"),
        )
        .reset_index()
    )

    # Create grid: rows = domains, columns = complexities
    fig, axes = plt.subplots(
        n_domains,
        n_complexities,
        figsize=(7 * n_complexities, 5 * n_domains),
        squeeze=False,
    )

    for d_idx, domain in enumerate(domains_to_plot):
        for c_idx, complexity in enumerate(complexities):
            ax = axes[d_idx, c_idx]

            # Filter data for this domain
            if has_domain and domain is not None:
                domain_analysis = analysis[analysis["domain"] == domain]
            else:
                domain_analysis = analysis

            x = np.arange(len(llms))
            bar_width = 0.35

            for i, success in enumerate(outcomes):
                df_subset = domain_analysis[
                    (domain_analysis["speech_complexity"] == complexity)
                    & (domain_analysis["success"] == success)
                ]

                means = []
                stds = []
                counts = []
                for llm in llms:
                    row = df_subset[df_subset["llm"] == llm]
                    if len(row) > 0:
                        means.append(row.iloc[0]["mean_duration"])
                        stds.append(
                            row.iloc[0]["std_duration"]
                            if pd.notna(row.iloc[0]["std_duration"])
                            else 0
                        )
                        counts.append(int(row.iloc[0]["count"]))
                    else:
                        means.append(0)
                        stds.append(0)
                        counts.append(0)

                colors = [get_llm_color(llm) for llm in llms]
                outcome_alpha = 0.9 if success else 0.5
                outcome_hatch = "" if success else "xx"

                ax.bar(
                    x + i * bar_width,
                    means,
                    bar_width,
                    yerr=stds,
                    capsize=3,
                    color=colors,
                    alpha=outcome_alpha,
                    hatch=outcome_hatch,
                    edgecolor=BAR_STYLE["edgecolor"],
                    linewidth=BAR_STYLE["linewidth"],
                    error_kw={"elinewidth": 1, "capthick": 1},
                )

                # Add count labels
                for bar_x, mean, count in zip(x + i * bar_width, means, counts):
                    if count > 0:
                        ax.text(
                            bar_x,
                            mean + 10,
                            f"n={count}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

            ax.set_xlabel("LLM", fontsize=12)
            ax.set_ylabel("Duration (seconds)", fontsize=12)
            # Title shows domain and complexity
            if domain:
                title = (
                    f"{domain.capitalize()} - {get_complexity_display_name(complexity)}"
                )
            else:
                title = get_complexity_display_name(complexity)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xticks(x + bar_width / 2)
            ax.set_xticklabels(
                [llm.split(":")[-1][:15] for llm in llms],
                rotation=45,
                ha="right",
                fontsize=10,
            )
            style_axis(ax)

    # Custom legend (add to last subplot)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="gray", alpha=0.9, label="Success"),
        Patch(facecolor="gray", alpha=0.5, hatch="xx", label="Failure"),
    ]
    axes[-1, -1].legend(handles=legend_elements, loc="upper right", fontsize=10)

    fig.suptitle(
        "Conversation Duration by Outcome (Success vs Failure)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        output_dir / "duration_by_outcome.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()
    logger.info(f"Saved: {output_dir / 'duration_by_outcome.pdf'}")


# =============================================================================
# Tool Call Count Analysis
# =============================================================================


def extract_tool_call_counts(
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """
    Extract tool call counts from all simulations.

    Args:
        results: List of (params, Results) tuples

    Returns:
        DataFrame with per-simulation tool call counts
    """
    rows = []
    for params, sim_results in results:
        if sim_results.simulations is None:
            logger.warning(
                f"Missing simulations for {params.get('llm', '?')}/{params.get('domain', '?')}/{params.get('speech_complexity', '?')}"
            )
            continue
        for sim in sim_results.simulations:
            agent_calls = 0
            agent_calls_success = 0
            agent_calls_failed = 0
            user_calls = 0

            # Count from ticks (full-duplex mode)
            if sim.ticks:
                for tick in sim.ticks:
                    if tick.agent_tool_calls:
                        agent_calls += len(tick.agent_tool_calls)
                        # Match with results to count success/failure
                        if tick.agent_tool_results:
                            for result in tick.agent_tool_results:
                                if result.error:
                                    agent_calls_failed += 1
                                else:
                                    agent_calls_success += 1
                    if tick.user_tool_calls:
                        user_calls += len(tick.user_tool_calls)

            rows.append(
                {
                    "llm": params["llm"],
                    "domain": params["domain"],
                    "speech_complexity": params["speech_complexity"],
                    "simulation_id": sim.id,
                    "task_id": sim.task_id,
                    "trial": sim.trial,
                    "agent_tool_calls": agent_calls,
                    "agent_calls_success": agent_calls_success,
                    "agent_calls_failed": agent_calls_failed,
                    "user_tool_calls": user_calls,
                    "total_tool_calls": agent_calls + user_calls,
                    "duration_seconds": sim.duration,
                    "task_success": (
                        sim.reward_info.reward >= 1.0 - 1e-6
                        if sim.reward_info
                        else False
                    ),
                }
            )

    return pd.DataFrame(rows)


def save_tool_calls_raw(
    output_dir: Path,
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """
    Save raw per-simulation tool call data.

    Args:
        output_dir: Directory for output files
        results: List of (params, Results) tuples

    Returns:
        DataFrame with per-simulation tool call data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = extract_tool_call_counts(results)

    if df_raw.empty:
        logger.warning("No tool call data found.")
        return df_raw

    raw_path = output_dir / f"{output_dir.name}_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    logger.info(f"Saved: {raw_path}")

    return df_raw


def save_tool_calls_analysis(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Save aggregated tool call analysis.

    Args:
        output_dir: Directory for output files
        df_raw: Raw tool call data

    Returns:
        DataFrame with aggregated analysis
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_raw.empty:
        return pd.DataFrame()

    # Check if domain column exists
    has_domain = "domain" in df_raw.columns
    group_cols = ["llm"]
    if has_domain:
        group_cols.append("domain")
    group_cols.append("speech_complexity")

    # Aggregate by LLM × [domain] × complexity
    analysis = (
        df_raw.groupby(group_cols)
        .agg(
            num_simulations=("simulation_id", "count"),
            mean_agent_calls=("agent_tool_calls", "mean"),
            std_agent_calls=("agent_tool_calls", "std"),
            mean_calls_success=("agent_calls_success", "mean"),
            mean_calls_failed=("agent_calls_failed", "mean"),
            total_calls_success=("agent_calls_success", "sum"),
            total_calls_failed=("agent_calls_failed", "sum"),
        )
        .reset_index()
    )

    # Calculate success rate
    analysis["tool_success_rate"] = (
        analysis["total_calls_success"]
        / (analysis["total_calls_success"] + analysis["total_calls_failed"])
    ).round(4)

    # Round for readability
    for col in analysis.columns:
        if col.startswith("mean_") or col.startswith("std_"):
            analysis[col] = analysis[col].round(2)

    analysis_path = output_dir / f"{output_dir.name}_analysis.csv"
    analysis.to_csv(analysis_path, index=False)
    logger.info(f"Saved: {analysis_path}")

    # Create pivoted summary
    pivot_cols = ["speech_complexity"]
    if has_domain:
        pivot_cols.insert(0, "domain")
    pivot_df = df_raw.pivot_table(
        index="llm",
        columns=pivot_cols,
        values="agent_tool_calls",
        aggfunc=["mean", "std"],
    ).round(2)
    if has_domain:
        pivot_df.columns = [
            f"{agg}_{dom}_{comp}" for agg, dom, comp in pivot_df.columns
        ]
    else:
        pivot_df.columns = [f"{agg}_{comp}" for agg, comp in pivot_df.columns]

    summary_path = output_dir / "summary.csv"
    pivot_df.to_csv(summary_path)
    logger.info(f"Saved: {summary_path}")

    return analysis


def save_tool_calls_by_outcome_analysis(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Save tool call analysis broken down by task success/failure.

    Args:
        output_dir: Directory for output files
        df_raw: Raw tool call data

    Returns:
        DataFrame with aggregated analysis
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_raw.empty:
        return pd.DataFrame()

    # Check if domain column exists
    has_domain = "domain" in df_raw.columns
    group_cols = ["llm"]
    if has_domain:
        group_cols.append("domain")
    group_cols.extend(["speech_complexity", "task_success"])

    # Aggregate by LLM × [domain] × complexity × task_success
    analysis = (
        df_raw.groupby(group_cols)
        .agg(
            num_simulations=("simulation_id", "count"),
            mean_agent_calls=("agent_tool_calls", "mean"),
            std_agent_calls=("agent_tool_calls", "std"),
            mean_calls_success=("agent_calls_success", "mean"),
            mean_calls_failed=("agent_calls_failed", "mean"),
        )
        .reset_index()
    )

    for col in analysis.columns:
        if col.startswith("mean_") or col.startswith("std_"):
            analysis[col] = analysis[col].round(2)

    analysis["task_outcome"] = analysis["task_success"].map(
        {True: "success", False: "failure"}
    )

    analysis_path = output_dir / "analysis_by_outcome.csv"
    analysis.to_csv(analysis_path, index=False)
    logger.info(f"Saved: {analysis_path}")

    return analysis


def plot_tool_calls(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> None:
    """
    Plot agent tool calls by LLM and speech complexity.
    Creates one subplot per domain if domain column is present.

    Args:
        output_dir: Directory to save figures
        df_raw: Raw tool call data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_raw.empty:
        return

    llms = sorted(df_raw["llm"].unique())
    complexities = ["control", "regular"]

    # Check if domain column exists and get domains to plot
    has_domain = "domain" in df_raw.columns
    if has_domain:
        tested_domains = df_raw["domain"].unique()
        domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    else:
        domains_to_plot = [None]

    n_domains = len(domains_to_plot)
    if n_domains == 0:
        return

    # Aggregate for plotting
    group_cols = ["llm", "speech_complexity"]
    if has_domain:
        group_cols.insert(1, "domain")
    analysis = (
        df_raw.groupby(group_cols)
        .agg(
            mean_calls=("agent_tool_calls", "mean"),
            std_calls=("agent_tool_calls", "std"),
        )
        .reset_index()
    )

    # Create subplots: one per domain (vertical layout)
    fig, axes = plt.subplots(n_domains, 1, figsize=(12, 6 * n_domains), squeeze=False)
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains_to_plot):
        ax = axes[d_idx]

        # Filter data for this domain
        if has_domain and domain is not None:
            domain_analysis = analysis[analysis["domain"] == domain]
        else:
            domain_analysis = analysis

        x = np.arange(len(llms))
        bar_width = 0.35

        for i, complexity in enumerate(complexities):
            style = get_complexity_style(complexity)
            df_c = domain_analysis[domain_analysis["speech_complexity"] == complexity]

            means = []
            stds = []
            for llm in llms:
                row = df_c[df_c["llm"] == llm]
                if len(row) > 0:
                    means.append(row.iloc[0]["mean_calls"])
                    stds.append(
                        row.iloc[0]["std_calls"]
                        if pd.notna(row.iloc[0]["std_calls"])
                        else 0
                    )
                else:
                    means.append(0)
                    stds.append(0)

            colors = [get_llm_color(llm) for llm in llms]
            ax.bar(
                x + i * bar_width,
                means,
                bar_width,
                yerr=stds,
                capsize=3,
                color=colors,
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
                error_kw={"elinewidth": 1, "capthick": 1},
            )

            # Add value labels
            for bar_x, mean in zip(x + i * bar_width, means):
                if mean > 0:
                    ax.text(
                        bar_x,
                        mean + 1,
                        f"{mean:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        ax.set_xlabel("LLM", fontsize=12)
        ax.set_ylabel("Agent Tool Calls", fontsize=12)
        title = f"{domain.capitalize()}" if domain else "Tool Calls"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(
            [llm.split(":")[-1][:20] for llm in llms],
            rotation=45,
            ha="right",
            fontsize=10,
        )
        style_axis(ax)

    legend_elements = [get_legend_patch("control"), get_legend_patch("regular")]
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=10)

    fig.suptitle(
        "Average Agent Tool Calls by LLM and Speech Complexity",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "tool_calls.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'tool_calls.pdf'}")


def plot_tool_calls_by_outcome(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> None:
    """
    Plot tool calls by LLM, complexity, and task success/failure outcome.
    Creates a grid of subplots: rows = domains, columns = complexities.

    Args:
        output_dir: Directory to save figures
        df_raw: Raw tool call data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_raw.empty:
        return

    llms = sorted(df_raw["llm"].unique())
    complexities = ["control", "regular"]
    outcomes = [True, False]  # task success, failure

    # Check if domain column exists and get domains to plot
    has_domain = "domain" in df_raw.columns
    if has_domain:
        tested_domains = df_raw["domain"].unique()
        domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    else:
        domains_to_plot = [None]

    n_domains = len(domains_to_plot)
    n_complexities = len(complexities)

    if n_domains == 0:
        return

    # Aggregate for plotting
    group_cols = ["llm", "speech_complexity", "task_success"]
    if has_domain:
        group_cols.insert(1, "domain")
    analysis = (
        df_raw.groupby(group_cols)
        .agg(
            mean_calls=("agent_tool_calls", "mean"),
            std_calls=("agent_tool_calls", "std"),
            count=("simulation_id", "count"),
        )
        .reset_index()
    )

    # Create grid: rows = domains, columns = complexities
    fig, axes = plt.subplots(
        n_domains,
        n_complexities,
        figsize=(7 * n_complexities, 5 * n_domains),
        squeeze=False,
    )

    for d_idx, domain in enumerate(domains_to_plot):
        for c_idx, complexity in enumerate(complexities):
            ax = axes[d_idx, c_idx]

            # Filter data for this domain
            if has_domain and domain is not None:
                domain_analysis = analysis[analysis["domain"] == domain]
            else:
                domain_analysis = analysis

            x = np.arange(len(llms))
            bar_width = 0.35

            for i, task_success in enumerate(outcomes):
                df_subset = domain_analysis[
                    (domain_analysis["speech_complexity"] == complexity)
                    & (domain_analysis["task_success"] == task_success)
                ]

                means = []
                stds = []
                counts = []
                for llm in llms:
                    row = df_subset[df_subset["llm"] == llm]
                    if len(row) > 0:
                        means.append(row.iloc[0]["mean_calls"])
                        stds.append(
                            row.iloc[0]["std_calls"]
                            if pd.notna(row.iloc[0]["std_calls"])
                            else 0
                        )
                        counts.append(int(row.iloc[0]["count"]))
                    else:
                        means.append(0)
                        stds.append(0)
                        counts.append(0)

                colors = [get_llm_color(llm) for llm in llms]
                outcome_alpha = 0.9 if task_success else 0.5
                outcome_hatch = "" if task_success else "xx"

                ax.bar(
                    x + i * bar_width,
                    means,
                    bar_width,
                    yerr=stds,
                    capsize=3,
                    color=colors,
                    alpha=outcome_alpha,
                    hatch=outcome_hatch,
                    edgecolor=BAR_STYLE["edgecolor"],
                    linewidth=BAR_STYLE["linewidth"],
                    error_kw={"elinewidth": 1, "capthick": 1},
                )

                # Add count labels
                for bar_x, mean, count in zip(x + i * bar_width, means, counts):
                    if count > 0:
                        ax.text(
                            bar_x,
                            mean + 0.5,
                            f"n={count}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

            ax.set_xlabel("LLM", fontsize=12)
            ax.set_ylabel("Agent Tool Calls", fontsize=12)
            # Title shows domain and complexity
            if domain:
                title = (
                    f"{domain.capitalize()} - {get_complexity_display_name(complexity)}"
                )
            else:
                title = get_complexity_display_name(complexity)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xticks(x + bar_width / 2)
            ax.set_xticklabels(
                [llm.split(":")[-1][:15] for llm in llms],
                rotation=45,
                ha="right",
                fontsize=10,
            )
            style_axis(ax)

    # Custom legend (add to last subplot)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="gray", alpha=0.9, label="Task Success"),
        Patch(facecolor="gray", alpha=0.5, hatch="xx", label="Task Failure"),
    ]
    axes[-1, -1].legend(handles=legend_elements, loc="upper right", fontsize=10)

    fig.suptitle(
        "Agent Tool Calls by Task Outcome",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        output_dir / "tool_calls_by_task_outcome.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()
    logger.info(f"Saved: {output_dir / 'tool_calls_by_task_outcome.pdf'}")


def plot_tool_calls_success_failure(
    output_dir: Path,
    df_raw: pd.DataFrame,
) -> None:
    """
    Plot breakdown of successful vs failed tool calls by LLM, complexity, and task outcome.

    Creates a 2x2 grid per domain: rows = complexity (control/regular), cols = task outcome (success/failure)
    For multiple domains, creates separate figures.

    Args:
        output_dir: Directory to save figures
        df_raw: Raw tool call data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_raw.empty:
        return

    llms = sorted(df_raw["llm"].unique())
    complexities = ["control", "regular"]
    task_outcomes = [True, False]  # task success, task failure

    # Check if domain column exists and get domains to plot
    has_domain = "domain" in df_raw.columns
    if has_domain:
        tested_domains = df_raw["domain"].unique()
        domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    else:
        domains_to_plot = [None]

    # Aggregate for plotting
    group_cols = ["llm", "speech_complexity", "task_success"]
    if has_domain:
        group_cols.insert(1, "domain")
    analysis = (
        df_raw.groupby(group_cols)
        .agg(
            mean_success=("agent_calls_success", "mean"),
            mean_failed=("agent_calls_failed", "mean"),
            count=("simulation_id", "count"),
        )
        .reset_index()
    )

    # Create one figure per domain
    for domain in domains_to_plot:
        # Filter data for this domain
        if has_domain and domain is not None:
            domain_analysis = analysis[analysis["domain"] == domain]
        else:
            domain_analysis = analysis

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

        for row_idx, complexity in enumerate(complexities):
            for col_idx, task_success in enumerate(task_outcomes):
                ax = axes[row_idx, col_idx]

                df_subset = domain_analysis[
                    (domain_analysis["speech_complexity"] == complexity)
                    & (domain_analysis["task_success"] == task_success)
                ]

                x = np.arange(len(llms))
                bar_width = 0.6

                success_vals = []
                failed_vals = []
                counts = []
                for llm in llms:
                    row = df_subset[df_subset["llm"] == llm]
                    if len(row) > 0:
                        success_vals.append(row.iloc[0]["mean_success"])
                        failed_vals.append(row.iloc[0]["mean_failed"])
                        counts.append(int(row.iloc[0]["count"]))
                    else:
                        success_vals.append(0)
                        failed_vals.append(0)
                        counts.append(0)

                colors = [get_llm_color(llm) for llm in llms]

                # Stacked bars: success on bottom, failed on top
                ax.bar(
                    x,
                    success_vals,
                    bar_width,
                    color=colors,
                    edgecolor=BAR_STYLE["edgecolor"],
                    linewidth=BAR_STYLE["linewidth"],
                )

                ax.bar(
                    x,
                    failed_vals,
                    bar_width,
                    bottom=success_vals,
                    color="#E15759",  # Red for failed
                    alpha=0.8,
                    edgecolor=BAR_STYLE["edgecolor"],
                    linewidth=BAR_STYLE["linewidth"],
                )

                # Add labels
                for bar_x, succ, fail, n in zip(x, success_vals, failed_vals, counts):
                    total = succ + fail
                    if total > 0:
                        # Failed count on top
                        ax.text(
                            bar_x,
                            total + 0.2,
                            f"{fail:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            color="#E15759",
                            fontweight="bold",
                        )
                        # Sample size
                        ax.text(
                            bar_x,
                            -0.5,
                            f"n={n}",
                            ha="center",
                            va="top",
                            fontsize=8,
                            color="gray",
                        )

                task_label = "Task Success" if task_success else "Task Failure"
                ax.set_title(
                    f"{get_complexity_display_name(complexity)} - {task_label}",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [llm.split(":")[-1][:15] for llm in llms],
                    rotation=45,
                    ha="right",
                    fontsize=10,
                )
                if col_idx == 0:
                    ax.set_ylabel("Mean Tool Calls", fontsize=11)
                if row_idx == 1:
                    ax.set_xlabel("LLM", fontsize=11)
                style_axis(ax)

        # Custom legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="gray", label="Successful Calls"),
            Patch(facecolor="#E15759", alpha=0.8, label="Failed Calls"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize=10,
        )

        # Title includes domain if present
        if domain:
            title = f"Tool Call Success/Failure - {domain.capitalize()}"
            filename = f"tool_calls_success_failure_{domain}.pdf"
        else:
            title = "Tool Call Success/Failure by Complexity and Task Outcome"
            filename = "tool_calls_success_failure.pdf"

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_dir / filename, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {output_dir / filename}")


# =============================================================================
# Reward-Based Action Success Analysis
# =============================================================================


def extract_reward_actions_data(
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """
    Extract reward-based action success data from all simulations.

    This analyzes whether the agent performed the expected actions
    as specified in the task's reward criteria.

    Args:
        results: List of (params, Results) tuples

    Returns:
        DataFrame with action match data per simulation
    """
    all_dfs = []

    for params, sim_results in results:
        llm = params.get("llm", "unknown")
        domain = params.get("domain", "unknown")
        speech_complexity = params.get("speech_complexity", "unknown")

        try:
            df_actions = result_reward_actions_analysis(sim_results)
            if df_actions is not None and not df_actions.empty:
                df_actions["llm"] = llm
                df_actions["domain"] = domain
                df_actions["speech_complexity"] = speech_complexity
                all_dfs.append(df_actions)
        except Exception as e:
            logger.warning(
                f"Failed to extract reward actions for {llm}/{domain}/{speech_complexity}: {e}"
            )

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def save_reward_action_analysis(
    output_dir: Path,
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """
    Analyze and save reward-based action success rates.

    Creates a pivoted table showing:
    - Rows: action names
    - Columns: LLM × speech_complexity combinations
    - Values: success rate (proportion of action_match == True)

    Args:
        output_dir: Directory for output files
        results: List of (params, Results) tuples

    Returns:
        The analysis DataFrame
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df_actions = extract_reward_actions_data(results)

    if df_actions.empty:
        logger.warning("No reward action data found")
        return pd.DataFrame()

    # Save raw data
    raw_path = output_dir / f"{output_dir.name}_raw.csv"
    df_actions.to_csv(raw_path, index=False)
    logger.info(f"Saved: {raw_path}")

    # Filter to agent actions only
    df_agent = df_actions[df_actions["requestor"] == "assistant"].copy()

    if df_agent.empty:
        logger.warning("No agent actions in reward data")
        return pd.DataFrame()

    # Calculate success rate per action, LLM, complexity
    grouped = (
        df_agent.groupby(["llm", "speech_complexity", "action_name"])
        .agg(
            num_expected=("action_match", "count"),
            num_matched=("action_match", "sum"),
        )
        .reset_index()
    )

    grouped["success_rate"] = grouped["num_matched"] / grouped["num_expected"]

    # Save detailed analysis
    analysis_path = output_dir / f"{output_dir.name}_analysis.csv"
    grouped.to_csv(analysis_path, index=False)
    logger.info(f"Saved: {analysis_path}")

    # Create pivoted table: rows = action_name, columns = LLM×complexity
    grouped["llm_complexity"] = (
        grouped["llm"].apply(lambda x: x.split(":")[-1][:20] if ":" in x else x[:20])
        + " ("
        + grouped["speech_complexity"]
        + ")"
    )

    # Pivot for success rate
    pivot_table = grouped.pivot_table(
        index="action_name",
        columns="llm_complexity",
        values="success_rate",
        fill_value=0,
        aggfunc="mean",
    )

    # Add overall success rate column
    action_success = df_agent.groupby("action_name")["action_match"].mean()
    pivot_table["OVERALL"] = pivot_table.index.map(action_success)

    # Sort by overall success rate ascending (worst first)
    pivot_table = pivot_table.sort_values("OVERALL", ascending=True)

    pivot_path = output_dir / "success_rates.csv"
    pivot_table.to_csv(pivot_path)
    logger.info(f"Saved: {pivot_path}")

    return pivot_table


# =============================================================================
# Review Analysis (LLM-based conversation review)
# =============================================================================


def extract_review_data(
    results: List[Tuple[dict, Results]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract review data from all simulations.

    Returns two DataFrames:
    1. Simulation-level summary (one row per simulation)
    2. Error-level detail (one row per ReviewError)

    Args:
        results: List of (params, Results) tuples

    Returns:
        Tuple of (df_summary, df_errors)
    """
    summary_rows = []
    error_rows = []

    for params, sim_results in results:
        llm = params.get("llm", "unknown")
        domain = params.get("domain", "unknown")
        speech_complexity = params.get("speech_complexity", "unknown")

        if sim_results.simulations is None:
            logger.warning(
                f"Missing simulations for {llm}/{domain}/{speech_complexity}"
            )
            continue

        for sim in sim_results.simulations:
            sim_id = sim.id
            task_id = sim.task_id
            trial = sim.trial

            review = sim.review
            if review is None:
                continue

            # Find first critical error source
            def get_error_position(e):
                """Get position for sorting: tick_start for full-duplex, turn_idx for turn-based."""
                if e.tick_start is not None:
                    return e.tick_start
                if e.turn_idx is not None:
                    return e.turn_idx
                return float("inf")

            critical_errors = [
                e
                for e in review.errors
                if (e.source == "agent" and e.severity == "critical")
                or (
                    e.source == "user"
                    and e.severity in ("critical_helped", "critical_hindered")
                )
            ]
            if critical_errors:
                first_critical = min(critical_errors, key=get_error_position)
                first_critical_source = first_critical.source
                first_critical_position = get_error_position(first_critical)
                first_critical_tag = (
                    first_critical.error_tags[0] if first_critical.error_tags else ""
                )
            else:
                first_critical_source = "none"
                first_critical_position = None
                first_critical_tag = ""

            # Simulation-level summary
            summary_rows.append(
                {
                    "llm": llm,
                    "domain": domain,
                    "speech_complexity": speech_complexity,
                    "simulation_id": sim_id,
                    "task_id": task_id,
                    "trial": trial,
                    "has_errors": review.has_errors,
                    "agent_error": review.agent_error,
                    "user_error": review.user_error,
                    "critical_user_error": review.critical_user_error,
                    "num_errors": len(review.errors),
                    "first_critical_source": first_critical_source,
                    "first_critical_position": first_critical_position,
                    "first_critical_tag": first_critical_tag,
                    "summary": review.summary,
                }
            )

            # Error-level detail
            for i, error in enumerate(review.errors):
                error_rows.append(
                    {
                        "llm": llm,
                        "domain": domain,
                        "speech_complexity": speech_complexity,
                        "simulation_id": sim_id,
                        "task_id": task_id,
                        "trial": trial,
                        "error_idx": i,
                        "source": error.source,
                        "error_type": error.error_type,
                        "severity": error.severity,
                        "error_tags": (
                            ",".join(error.error_tags) if error.error_tags else ""
                        ),
                        "tick_start": error.tick_start,
                        "tick_end": error.tick_end,
                        "turn_idx": error.turn_idx,
                        "reasoning": error.reasoning,
                        "correct_behavior": error.correct_behavior,
                    }
                )

    df_summary = pd.DataFrame(summary_rows)
    df_errors = pd.DataFrame(error_rows)

    return df_summary, df_errors


def save_review_analysis(
    output_dir: Path,
    results: List[Tuple[dict, Results]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze and save review data.

    Creates:
    - raw_summary.csv: Simulation-level review summary
    - raw_errors.csv: Individual error details
    - error_rates.csv: Error rates by LLM × complexity
    - error_types.csv: Breakdown by error type and severity
    - error_tags.csv: Breakdown by error tags

    Args:
        output_dir: Directory for output files
        results: List of (params, Results) tuples

    Returns:
        Tuple of (df_summary, df_errors)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df_summary, df_errors = extract_review_data(results)

    if df_summary.empty:
        logger.warning("No review data found")
        return df_summary, df_errors

    # Save raw data
    summary_path = output_dir / "raw_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"Saved: {summary_path}")

    if not df_errors.empty:
        errors_path = output_dir / "raw_errors.csv"
        df_errors.to_csv(errors_path, index=False)
        logger.info(f"Saved: {errors_path}")

    # === Error rates by LLM × complexity ===
    error_rates = (
        df_summary.groupby(["llm", "speech_complexity"])
        .agg(
            num_simulations=("simulation_id", "count"),
            simulations_with_errors=("has_errors", "sum"),
            simulations_with_agent_error=("agent_error", "sum"),
            simulations_with_user_error=("user_error", "sum"),
            simulations_with_critical_user_error=("critical_user_error", "sum"),
            total_errors=("num_errors", "sum"),
        )
        .reset_index()
    )

    error_rates["error_rate"] = (
        error_rates["simulations_with_errors"] / error_rates["num_simulations"]
    )
    error_rates["agent_error_rate"] = (
        error_rates["simulations_with_agent_error"] / error_rates["num_simulations"]
    )
    error_rates["user_error_rate"] = (
        error_rates["simulations_with_user_error"] / error_rates["num_simulations"]
    )
    error_rates["avg_errors_per_sim"] = (
        error_rates["total_errors"] / error_rates["num_simulations"]
    )

    rates_path = output_dir / "error_rates.csv"
    error_rates.to_csv(rates_path, index=False)
    logger.info(f"Saved: {rates_path}")

    if not df_errors.empty:
        # === Error breakdown by source and severity ===
        df_errors["llm_complexity"] = (
            df_errors["llm"].apply(
                lambda x: x.split(":")[-1][:20] if ":" in x else x[:20]
            )
            + " ("
            + df_errors["speech_complexity"]
            + ")"
        )

        # By source (agent vs user)
        source_counts = (
            df_errors.groupby(["source", "llm_complexity"]).size().unstack(fill_value=0)
        )
        source_counts["TOTAL"] = source_counts.sum(axis=1)
        source_path = output_dir / "error_by_source.csv"
        source_counts.to_csv(source_path)
        logger.info(f"Saved: {source_path}")

        # By severity
        severity_counts = (
            df_errors.groupby(["severity", "llm_complexity"])
            .size()
            .unstack(fill_value=0)
        )
        severity_counts["TOTAL"] = severity_counts.sum(axis=1)
        severity_path = output_dir / "error_by_severity.csv"
        severity_counts.to_csv(severity_path)
        logger.info(f"Saved: {severity_path}")

        # By error type
        type_counts = (
            df_errors.groupby(["error_type", "llm_complexity"])
            .size()
            .unstack(fill_value=0)
        )
        type_counts["TOTAL"] = type_counts.sum(axis=1)
        type_path = output_dir / "error_by_type.csv"
        type_counts.to_csv(type_path)
        logger.info(f"Saved: {type_path}")

        # By error tags (explode comma-separated tags)
        df_tags = df_errors.copy()
        df_tags["error_tags"] = df_tags["error_tags"].str.split(",")
        df_tags = df_tags.explode("error_tags")
        df_tags = df_tags[df_tags["error_tags"] != ""]

        if not df_tags.empty:
            tag_counts = (
                df_tags.groupby(["error_tags", "llm_complexity"])
                .size()
                .unstack(fill_value=0)
            )
            tag_counts["TOTAL"] = tag_counts.sum(axis=1)
            tag_counts = tag_counts.sort_values("TOTAL", ascending=False)
            tags_path = output_dir / "error_by_tags.csv"
            tag_counts.to_csv(tags_path)
            logger.info(f"Saved: {tags_path}")

    # === First critical source analysis ===
    if "first_critical_source" in df_summary.columns:
        first_critical = (
            df_summary.groupby(["llm", "speech_complexity", "first_critical_source"])
            .size()
            .unstack(fill_value=0)
        )
        # Calculate percentages
        first_critical_pct = (
            first_critical.div(first_critical.sum(axis=1), axis=0) * 100
        )
        first_critical_pct = first_critical_pct.round(1)
        # Add count suffix
        for col in first_critical.columns:
            first_critical_pct[f"{col}_count"] = first_critical[col]
        first_critical_path = output_dir / "first_critical_source.csv"
        first_critical_pct.to_csv(first_critical_path)
        logger.info(f"Saved: {first_critical_path}")

    return df_summary, df_errors


def plot_agent_errors_per_simulation(
    output_dir: Path,
    df_summary: pd.DataFrame,
    df_errors: pd.DataFrame,
) -> None:
    """
    Plot agent errors per simulation by LLM and speech complexity.

    Creates a single PDF with 3 subplots:
    - All agent errors
    - Critical errors only
    - Minor errors only
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_summary.empty:
        logger.warning("No review data to plot")
        return

    sim_counts = (
        df_summary.groupby(["llm", "speech_complexity"])
        .size()
        .reset_index(name="num_simulations")
    )

    if not df_errors.empty:
        df_agent = df_errors[df_errors["source"] == "agent"].copy()
    else:
        df_agent = pd.DataFrame()

    llms = df_summary["llm"].unique()
    complexities = ["control", "regular"]

    severity_configs = [
        ("All Errors", lambda df: df),
        ("Critical Errors", lambda df: df[df["severity"] == "critical"]),
        ("Minor Errors", lambda df: df[df["severity"] == "minor"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, (title, filter_func) in enumerate(severity_configs):
        ax = axes[ax_idx]
        x = np.arange(len(llms))
        bar_width = 0.35

        for i, complexity in enumerate(complexities):
            counts_per_sim = []
            colors = []
            style = get_complexity_style(complexity)

            for llm in llms:
                mask = (sim_counts["llm"] == llm) & (
                    sim_counts["speech_complexity"] == complexity
                )
                num_sims = sim_counts.loc[mask, "num_simulations"].values
                num_sims = num_sims[0] if len(num_sims) > 0 else 1

                if not df_agent.empty:
                    df_filtered = filter_func(df_agent)
                    df_llm = df_filtered[
                        (df_filtered["llm"] == llm)
                        & (df_filtered["speech_complexity"] == complexity)
                    ]
                    error_count = len(df_llm)
                else:
                    error_count = 0

                counts_per_sim.append(error_count / num_sims)
                colors.append(get_llm_color(llm))

            ax.bar(
                x + i * bar_width,
                counts_per_sim,
                bar_width,
                color=colors,
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )

            for bar_x, val in zip(x + i * bar_width, counts_per_sim):
                if val > 0:
                    ax.text(
                        bar_x,
                        val + 0.05,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        ax.set_xlabel("LLM", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Errors per Simulation", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(
            [llm.split(":")[-1][:20] for llm in llms],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        style_axis(ax)

    legend_elements = [get_legend_patch("control"), get_legend_patch("regular")]
    axes[-1].legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9,
        title="Speech Complexity",
    )

    fig.suptitle(
        "Agent Errors per Simulation by Severity", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plot_path = output_dir / "agent_errors_per_simulation.pdf"
    plt.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {plot_path}")


def plot_agent_error_breakdown(
    output_dir: Path,
    df_errors: pd.DataFrame,
) -> None:
    """Plot breakdown of agent error types by LLM."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_errors.empty:
        return

    df_agent = df_errors[df_errors["source"] == "agent"].copy()
    if df_agent.empty:
        return

    df_agent["error_tags"] = df_agent["error_tags"].str.split(",")
    df_exploded = df_agent.explode("error_tags")
    df_exploded = df_exploded[df_exploded["error_tags"] != ""]

    if df_exploded.empty:
        return

    df_exploded["llm_short"] = df_exploded["llm"].apply(
        lambda x: x.split(":")[-1][:25] if ":" in x else x[:25]
    )

    tag_totals = df_exploded["error_tags"].value_counts()
    top_tags = tag_totals.head(8).index.tolist()

    df_exploded["error_tag_grouped"] = df_exploded["error_tags"].apply(
        lambda x: x if x in top_tags else "other"
    )

    counts = (
        df_exploded.groupby(["llm_short", "error_tag_grouped"])
        .size()
        .unstack(fill_value=0)
    )

    tag_order = [t for t in top_tags if t in counts.columns]
    if "other" in counts.columns and "other" not in tag_order:
        tag_order.append("other")
    counts = counts[tag_order]

    tag_colors = {
        "incorrect_interpretation": "#e74c3c",
        "tool_call_argument_error": "#e67e22",
        "missed_required_action": "#f39c12",
        "guideline_violation": "#9b59b6",
        "premature_termination": "#3498db",
        "hallucination": "#1abc9c",
        "inconsistent_behavior": "#2ecc71",
        "tool_call_schema_error": "#95a5a6",
        "other": "#bdc3c7",
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    llms = counts.index.tolist()
    y_pos = np.arange(len(llms))
    bar_height = 0.6
    left = np.zeros(len(llms))

    for tag in tag_order:
        if tag in counts.columns:
            values = counts[tag].values
            color = tag_colors.get(tag, "#bdc3c7")
            ax.barh(
                y_pos,
                values,
                bar_height,
                left=left,
                label=tag.replace("_", " ").title(),
                color=color,
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )
            left += values

    for i, total in enumerate(left):
        ax.text(
            total + 2,
            i,
            f"{int(total)}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(llms, fontsize=11)
    ax.set_xlabel("Number of Errors", fontsize=12)
    ax.set_title("Agent Error Types by LLM", fontsize=14, fontweight="bold")
    style_axis(ax, grid=False)
    ax.legend(
        loc="upper right",
        fontsize=9,
        title="Error Type",
        title_fontsize=10,
        framealpha=0.9,
    )
    ax.set_xlim(0, left.max() * 1.15)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(
        output_dir / "agent_error_breakdown.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()
    logger.info(f"Saved: {output_dir / 'agent_error_breakdown.pdf'}")


def plot_agent_error_by_severity(output_dir: Path, df_errors: pd.DataFrame) -> None:
    """Plot breakdown of agent errors by severity for each LLM."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_errors.empty:
        return

    df_agent = df_errors[df_errors["source"] == "agent"].copy()
    if df_agent.empty:
        return

    df_agent["llm_short"] = df_agent["llm"].apply(
        lambda x: x.split(":")[-1][:20] if ":" in x else x[:20]
    )
    df_agent["llm_complexity"] = (
        df_agent["llm_short"] + " (" + df_agent["speech_complexity"] + ")"
    )

    counts = (
        df_agent.groupby(["llm_complexity", "severity"]).size().unstack(fill_value=0)
    )

    severity_order = ["critical", "minor"]
    severity_colors = {"critical": "#e74c3c", "minor": "#f39c12"}

    counts = counts[[s for s in severity_order if s in counts.columns]]
    counts = counts.sort_index()

    fig, ax = plt.subplots(figsize=(12, 7))
    labels = counts.index.tolist()
    y_pos = np.arange(len(labels))
    bar_height = 0.6
    left = np.zeros(len(labels))

    for severity in severity_order:
        if severity in counts.columns:
            values = counts[severity].values
            color = severity_colors.get(severity, "#bdc3c7")
            ax.barh(
                y_pos,
                values,
                bar_height,
                left=left,
                label=severity.replace("_", " ").title(),
                color=color,
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )
            left += values

    for i, total in enumerate(left):
        ax.text(
            total + 1,
            i,
            f"{int(total)}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Number of Errors", fontsize=12)
    ax.set_title(
        "Agent Error Severity by LLM and Speech Complexity",
        fontsize=14,
        fontweight="bold",
    )
    style_axis(ax, grid=False)
    ax.legend(loc="lower right", fontsize=10, title="Severity", title_fontsize=11)
    ax.set_xlim(0, left.max() * 1.15)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(
        output_dir / "agent_error_severity.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()
    logger.info(f"Saved: {output_dir / 'agent_error_severity.pdf'}")


def plot_agent_error_breakdown_by_complexity(
    output_dir: Path, df_errors: pd.DataFrame
) -> None:
    """Plot breakdown of agent error types by LLM and speech complexity."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_errors.empty:
        return

    df_agent = df_errors[df_errors["source"] == "agent"].copy()
    if df_agent.empty:
        return

    df_agent["error_tags"] = df_agent["error_tags"].str.split(",")
    df_exploded = df_agent.explode("error_tags")
    df_exploded = df_exploded[df_exploded["error_tags"] != ""]

    if df_exploded.empty:
        return

    df_exploded["llm_short"] = df_exploded["llm"].apply(
        lambda x: x.split(":")[-1][:20] if ":" in x else x[:20]
    )

    tag_totals = df_exploded["error_tags"].value_counts()
    top_tags = tag_totals.head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    llms = df_exploded["llm_short"].unique()
    complexities = ["control", "regular"]
    x = np.arange(len(llms))
    bar_width = 0.35

    for idx, tag in enumerate(top_tags):
        ax = axes[idx]
        df_tag = df_exploded[df_exploded["error_tags"] == tag]

        for i, complexity in enumerate(complexities):
            style = get_complexity_style(complexity)
            df_c = df_tag[df_tag["speech_complexity"] == complexity]
            tag_counts = df_c.groupby("llm_short").size()
            values = [tag_counts.get(llm, 0) for llm in llms]
            colors = [
                get_llm_color(llm)
                for llm in df_exploded[df_exploded["llm_short"].isin(llms)][
                    "llm"
                ].unique()
            ]

            ax.bar(
                x + i * bar_width,
                values,
                bar_width,
                color=colors[: len(llms)],
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )

        ax.set_title(tag.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(llms, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Count", fontsize=10)
        style_axis(ax)

    legend_elements = [get_legend_patch("control"), get_legend_patch("regular")]
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.suptitle(
        "Agent Error Types: Control vs Regular Speech", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "agent_error_by_complexity.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()
    logger.info(f"Saved: {output_dir / 'agent_error_by_complexity.pdf'}")


# =============================================================================
# User Error Analysis Plots
# =============================================================================


def plot_user_errors_per_simulation(
    output_dir: Path,
    df_summary: pd.DataFrame,
    df_errors: pd.DataFrame,
) -> None:
    """Plot user errors per simulation by LLM and speech complexity."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_summary.empty:
        return

    sim_counts = (
        df_summary.groupby(["llm", "speech_complexity"])
        .size()
        .reset_index(name="num_simulations")
    )

    if not df_errors.empty:
        df_user = df_errors[df_errors["source"] == "user"].copy()
    else:
        df_user = pd.DataFrame()

    llms = df_summary["llm"].unique()
    complexities = ["control", "regular"]

    severity_configs = [
        ("All Errors", lambda df: df),
        (
            "Critical Errors",
            lambda df: df[
                df["severity"].isin(["critical_helped", "critical_hindered"])
            ],
        ),
        ("Minor Errors", lambda df: df[df["severity"] == "minor"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, (title, filter_func) in enumerate(severity_configs):
        ax = axes[ax_idx]
        x = np.arange(len(llms))
        bar_width = 0.35

        for i, complexity in enumerate(complexities):
            counts_per_sim = []
            colors = []
            style = get_complexity_style(complexity)

            for llm in llms:
                mask = (sim_counts["llm"] == llm) & (
                    sim_counts["speech_complexity"] == complexity
                )
                num_sims = sim_counts.loc[mask, "num_simulations"].values
                num_sims = num_sims[0] if len(num_sims) > 0 else 1

                if not df_user.empty:
                    df_filtered = filter_func(df_user)
                    df_llm = df_filtered[
                        (df_filtered["llm"] == llm)
                        & (df_filtered["speech_complexity"] == complexity)
                    ]
                    error_count = len(df_llm)
                else:
                    error_count = 0

                counts_per_sim.append(error_count / num_sims)
                colors.append(get_llm_color(llm))

            ax.bar(
                x + i * bar_width,
                counts_per_sim,
                bar_width,
                color=colors,
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )

            for bar_x, val in zip(x + i * bar_width, counts_per_sim):
                if val > 0:
                    ax.text(
                        bar_x,
                        val + 0.02,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        ax.set_xlabel("LLM", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Errors per Simulation", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(
            [llm.split(":")[-1][:20] for llm in llms],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        style_axis(ax)

    legend_elements = [get_legend_patch("control"), get_legend_patch("regular")]
    axes[-1].legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9,
        title="Speech Complexity",
    )

    fig.suptitle(
        "User Simulator Errors per Simulation by Severity",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "user_errors_per_simulation.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()
    logger.info(f"Saved: {output_dir / 'user_errors_per_simulation.pdf'}")


def plot_user_error_breakdown(output_dir: Path, df_errors: pd.DataFrame) -> None:
    """Plot breakdown of user error types by LLM."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_errors.empty:
        return

    df_user = df_errors[df_errors["source"] == "user"].copy()
    if df_user.empty:
        return

    df_user["error_tags"] = df_user["error_tags"].str.split(",")
    df_exploded = df_user.explode("error_tags")
    df_exploded = df_exploded[df_exploded["error_tags"] != ""]

    if df_exploded.empty:
        return

    df_exploded["llm_short"] = df_exploded["llm"].apply(
        lambda x: x.split(":")[-1][:25] if ":" in x else x[:25]
    )

    tag_totals = df_exploded["error_tags"].value_counts()
    top_tags = tag_totals.head(8).index.tolist()

    df_exploded["error_tag_grouped"] = df_exploded["error_tags"].apply(
        lambda x: x if x in top_tags else "other"
    )

    counts = (
        df_exploded.groupby(["llm_short", "error_tag_grouped"])
        .size()
        .unstack(fill_value=0)
    )

    tag_order = [t for t in top_tags if t in counts.columns]
    if "other" in counts.columns and "other" not in tag_order:
        tag_order.append("other")
    counts = counts[tag_order]

    tag_colors = {
        "premature_termination": "#3498db",
        "inconsistent_behavior": "#2ecc71",
        "guideline_violation": "#9b59b6",
        "incorrect_interpretation": "#e74c3c",
        "hallucination": "#1abc9c",
        "interruption_error": "#e67e22",
        "other": "#bdc3c7",
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    llms = counts.index.tolist()
    y_pos = np.arange(len(llms))
    bar_height = 0.6
    left = np.zeros(len(llms))

    for tag in tag_order:
        if tag in counts.columns:
            values = counts[tag].values
            color = tag_colors.get(tag, "#bdc3c7")
            ax.barh(
                y_pos,
                values,
                bar_height,
                left=left,
                label=tag.replace("_", " ").title(),
                color=color,
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )
            left += values

    for i, total in enumerate(left):
        ax.text(
            total + 0.5,
            i,
            f"{int(total)}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(llms, fontsize=11)
    ax.set_xlabel("Number of Errors", fontsize=12)
    ax.set_title("User Simulator Error Types by LLM", fontsize=14, fontweight="bold")
    style_axis(ax, grid=False)
    ax.legend(
        loc="upper right",
        fontsize=9,
        title="Error Type",
        title_fontsize=10,
        framealpha=0.9,
    )
    ax.set_xlim(0, left.max() * 1.15 if left.max() > 0 else 1)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(
        output_dir / "user_error_breakdown.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()
    logger.info(f"Saved: {output_dir / 'user_error_breakdown.pdf'}")


def plot_user_error_by_severity(output_dir: Path, df_errors: pd.DataFrame) -> None:
    """Plot breakdown of user errors by severity for each LLM."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_errors.empty:
        return

    df_user = df_errors[df_errors["source"] == "user"].copy()
    if df_user.empty:
        return

    df_user["llm_short"] = df_user["llm"].apply(
        lambda x: x.split(":")[-1][:20] if ":" in x else x[:20]
    )
    df_user["llm_complexity"] = (
        df_user["llm_short"] + " (" + df_user["speech_complexity"] + ")"
    )

    counts = (
        df_user.groupby(["llm_complexity", "severity"]).size().unstack(fill_value=0)
    )

    severity_order = ["critical_hindered", "critical_helped", "minor"]
    severity_colors = {
        "critical_hindered": "#e74c3c",
        "critical_helped": "#f39c12",
        "minor": "#95a5a6",
    }

    counts = counts[[s for s in severity_order if s in counts.columns]]
    counts = counts.sort_index()

    fig, ax = plt.subplots(figsize=(12, 7))
    labels = counts.index.tolist()
    y_pos = np.arange(len(labels))
    bar_height = 0.6
    left = np.zeros(len(labels))

    for severity in severity_order:
        if severity in counts.columns:
            values = counts[severity].values
            color = severity_colors.get(severity, "#bdc3c7")
            ax.barh(
                y_pos,
                values,
                bar_height,
                left=left,
                label=severity.replace("_", " ").title(),
                color=color,
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )
            left += values

    for i, total in enumerate(left):
        ax.text(
            total + 0.3,
            i,
            f"{int(total)}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Number of Errors", fontsize=12)
    ax.set_title(
        "User Simulator Error Severity by LLM and Speech Complexity",
        fontsize=14,
        fontweight="bold",
    )
    style_axis(ax, grid=False)
    ax.legend(loc="lower right", fontsize=10, title="Severity", title_fontsize=11)
    ax.set_xlim(0, left.max() * 1.15 if left.max() > 0 else 1)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(
        output_dir / "user_error_severity.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()
    logger.info(f"Saved: {output_dir / 'user_error_severity.pdf'}")


def plot_first_critical_source(
    output_dir: Path,
    df_summary: pd.DataFrame,
) -> None:
    """
    Plot first critical error source by LLM and speech complexity.

    Shows who caused the first critical error: agent, user, or none.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_summary.empty or "first_critical_source" not in df_summary.columns:
        return

    # Aggregate by LLM, complexity, and first critical source
    grouped = (
        df_summary.groupby(["llm", "speech_complexity", "first_critical_source"])
        .size()
        .unstack(fill_value=0)
    )

    if grouped.empty:
        return

    # Calculate percentages
    grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100

    llms = df_summary["llm"].unique()
    complexities = ["control", "regular"]
    sources = ["agent", "user", "none"]
    source_colors = {
        "agent": "#C44E52",  # Red
        "user": "#4C72B0",  # Blue
        "none": "#55A868",  # Green
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(llms))
    bar_width = 0.35

    for i, complexity in enumerate(complexities):
        style = get_complexity_style(complexity)
        bottom = np.zeros(len(llms))

        for source in sources:
            values = []
            for llm in llms:
                key = (llm, complexity)
                if key in grouped_pct.index and source in grouped_pct.columns:
                    values.append(grouped_pct.loc[key, source])
                else:
                    values.append(np.nan)  # No data available

            ax.bar(
                x + i * bar_width,
                values,
                bar_width,
                bottom=bottom,
                label=f"{source} ({complexity})" if i == 0 else "",
                color=source_colors.get(source, "gray"),
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )
            bottom += np.array(values)

    ax.set_xlabel("LLM", fontsize=12)
    ax.set_ylabel("Percentage of Simulations", fontsize=12)
    ax.set_title(
        "First Critical Error Source by LLM and Speech Complexity",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(
        [llm.split(":")[-1][:20] for llm in llms], rotation=45, ha="right", fontsize=10
    )
    ax.set_ylim(0, 105)

    # Custom legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=source_colors["agent"], label="Agent"),
        Patch(facecolor=source_colors["user"], label="User"),
        Patch(facecolor=source_colors["none"], label="None (no critical error)"),
        get_legend_patch("control", facecolor="gray"),
        get_legend_patch("regular", facecolor="gray"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
    )
    style_axis(ax)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(
        output_dir / "first_critical_source.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()
    logger.info(f"Saved: {output_dir / 'first_critical_source.pdf'}")


# =============================================================================
# Authentication Analysis
# =============================================================================


def extract_auth_data(
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """
    Extract authentication classification data from all simulations.

    Returns a DataFrame with one row per simulation containing auth status.
    """
    rows = []

    for params, sim_results in results:
        llm = params.get("llm", "unknown")
        domain = params.get("domain", "unknown")
        speech_complexity = params.get("speech_complexity", "unknown")

        if sim_results.simulations is None:
            logger.warning(
                f"Missing simulations for {llm}/{domain}/{speech_complexity}"
            )
            continue

        for sim in sim_results.simulations:
            auth_status = "not_checked"
            if sim.auth_classification:
                auth_status = sim.auth_classification.status or "unknown"

            rows.append(
                {
                    "llm": llm,
                    "domain": domain,
                    "speech_complexity": speech_complexity,
                    "simulation_id": sim.id,
                    "task_id": sim.task_id,
                    "trial": sim.trial,
                    "auth_status": auth_status,
                }
            )

    return pd.DataFrame(rows)


def save_auth_analysis(
    output_dir: Path,
    results: List[Tuple[dict, Results]],
) -> pd.DataFrame:
    """
    Analyze and save authentication data.

    Creates:
    - auth_raw.csv: Per-simulation auth status
    - auth_summary.csv: Auth rates by LLM × [domain] × complexity

    Returns:
        DataFrame with auth data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df_auth = extract_auth_data(results)

    if df_auth.empty:
        logger.warning("No auth data found")
        return df_auth

    # Save raw data
    raw_path = output_dir / "auth_raw.csv"
    df_auth.to_csv(raw_path, index=False)
    logger.info(f"Saved: {raw_path}")

    # Check if domain column exists
    has_domain = "domain" in df_auth.columns
    group_cols = ["llm"]
    if has_domain:
        group_cols.append("domain")
    group_cols.extend(["speech_complexity", "auth_status"])

    # Summary by LLM × [domain] × complexity
    summary = (
        df_auth.groupby(group_cols[:-1] + ["auth_status"]).size().unstack(fill_value=0)
    )

    # Calculate totals and rates
    summary["total"] = summary.sum(axis=1)
    if "succeeded" in summary.columns:
        summary["success_rate"] = (summary["succeeded"] / summary["total"] * 100).round(
            1
        )
    if "failed" in summary.columns:
        summary["failure_rate"] = (summary["failed"] / summary["total"] * 100).round(1)

    summary_path = output_dir / "auth_summary.csv"
    summary.to_csv(summary_path)
    logger.info(f"Saved: {summary_path}")

    return df_auth


def plot_auth_success(
    output_dir: Path,
    df_auth: pd.DataFrame,
) -> None:
    """
    Plot authentication success/failure rates by LLM and speech complexity.
    Creates one subplot per domain if domain column is present.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_auth.empty:
        return

    llms = sorted(df_auth["llm"].unique())
    complexities = ["control", "regular"]

    # Check if domain column exists and get domains to plot
    has_domain = "domain" in df_auth.columns
    if has_domain:
        tested_domains = df_auth["domain"].unique()
        domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    else:
        domains_to_plot = [None]

    n_domains = len(domains_to_plot)
    if n_domains == 0:
        return

    # Status colors
    status_colors = {
        "succeeded": "#55A868",  # Green
        "failed": "#C44E52",  # Red
        "not_needed": "#8172B3",  # Purple
        "not_checked": "#CCCCCC",  # Gray
    }

    # Create subplots: one per domain (vertical layout)
    fig, axes = plt.subplots(n_domains, 1, figsize=(14, 6 * n_domains), squeeze=False)
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains_to_plot):
        ax = axes[d_idx]

        # Filter data for this domain
        if has_domain and domain is not None:
            domain_df = df_auth[df_auth["domain"] == domain]
        else:
            domain_df = df_auth

        # Aggregate by LLM, complexity, and auth status
        grouped = (
            domain_df.groupby(["llm", "speech_complexity", "auth_status"])
            .size()
            .unstack(fill_value=0)
        )

        if grouped.empty:
            continue

        x = np.arange(len(llms))
        bar_width = 0.35

        for i, complexity in enumerate(complexities):
            style = get_complexity_style(complexity)
            bottom = np.zeros(len(llms))

            for status in ["succeeded", "failed", "not_needed", "not_checked"]:
                if status not in grouped.columns:
                    continue

                values = []
                for llm in llms:
                    key = (llm, complexity)
                    if key in grouped.index:
                        values.append(grouped.loc[key, status])
                    else:
                        values.append(np.nan)  # No data available

                ax.bar(
                    x + i * bar_width,
                    values,
                    bar_width,
                    bottom=bottom,
                    color=status_colors.get(status, "gray"),
                    alpha=style["alpha"],
                    hatch=style["hatch"],
                    edgecolor=BAR_STYLE["edgecolor"],
                    linewidth=BAR_STYLE["linewidth"],
                )
                bottom += np.array(values)

        ax.set_xlabel("LLM", fontsize=12)
        ax.set_ylabel("Number of Simulations", fontsize=12)
        title = f"{domain.capitalize()}" if domain else "Auth Outcome"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(
            [llm.split(":")[-1][:20] for llm in llms],
            rotation=45,
            ha="right",
            fontsize=10,
        )
        style_axis(ax)

    # Custom legend (add to last subplot)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=status_colors["succeeded"], label="Succeeded"),
        Patch(facecolor=status_colors["failed"], label="Failed"),
        Patch(facecolor=status_colors["not_needed"], label="Not Needed"),
        get_legend_patch("control", facecolor="gray"),
        get_legend_patch("regular", facecolor="gray"),
    ]
    axes[-1].legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
    )

    fig.suptitle(
        "Authentication Outcome by LLM and Speech Complexity",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    plt.savefig(output_dir / "auth_outcome.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'auth_outcome.pdf'}")


def plot_auth_success_rate(
    output_dir: Path,
    df_auth: pd.DataFrame,
) -> None:
    """
    Plot authentication success rate by LLM and speech complexity.
    Creates one subplot per domain if domain column is present.

    Only considers simulations where auth was needed (succeeded or failed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_auth.empty:
        return

    # Filter to only auth-relevant simulations
    df_auth_relevant = df_auth[df_auth["auth_status"].isin(["succeeded", "failed"])]

    if df_auth_relevant.empty:
        logger.info("No simulations with auth succeeded/failed status")
        return

    llms = sorted(df_auth_relevant["llm"].unique())
    complexities = ["control", "regular"]

    # Check if domain column exists and get domains to plot
    has_domain = "domain" in df_auth_relevant.columns
    if has_domain:
        tested_domains = df_auth_relevant["domain"].unique()
        domains_to_plot = [d for d in DOMAINS if d in tested_domains]
    else:
        domains_to_plot = [None]

    n_domains = len(domains_to_plot)
    if n_domains == 0:
        return

    # Calculate success rate
    group_cols = ["llm", "speech_complexity"]
    if has_domain:
        group_cols.insert(1, "domain")
    summary = (
        df_auth_relevant.groupby(group_cols)
        .agg(
            total=("auth_status", "count"),
            succeeded=("auth_status", lambda x: (x == "succeeded").sum()),
        )
        .reset_index()
    )
    summary["success_rate"] = summary["succeeded"] / summary["total"]

    # Create subplots: one per domain (vertical layout)
    fig, axes = plt.subplots(n_domains, 1, figsize=(12, 6 * n_domains), squeeze=False)
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains_to_plot):
        ax = axes[d_idx]

        # Filter data for this domain
        if has_domain and domain is not None:
            domain_summary = summary[summary["domain"] == domain]
        else:
            domain_summary = summary

        x = np.arange(len(llms))
        bar_width = 0.35

        for i, complexity in enumerate(complexities):
            style = get_complexity_style(complexity)
            df_c = domain_summary[domain_summary["speech_complexity"] == complexity]

            values = []
            labels = []
            for llm in llms:
                row = df_c[df_c["llm"] == llm]
                if len(row) > 0:
                    values.append(row.iloc[0]["success_rate"])
                    labels.append(
                        f"{int(row.iloc[0]['succeeded'])}/{int(row.iloc[0]['total'])}"
                    )
                else:
                    values.append(np.nan)  # No data available
                    labels.append("N/A")

            colors = [get_llm_color(llm) for llm in llms]
            ax.bar(
                x + i * bar_width,
                values,
                bar_width,
                color=colors,
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )

            # Add count labels (skip for missing data)
            for bar_x, val, label in zip(x + i * bar_width, values, labels):
                if label == "N/A":
                    continue
                ax.text(bar_x, val + 0.02, label, ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("LLM", fontsize=12)
        ax.set_ylabel("Authentication Success Rate", fontsize=12)
        title = f"{domain.capitalize()}" if domain else "Auth Success Rate"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(
            [llm.split(":")[-1][:20] for llm in llms],
            rotation=45,
            ha="right",
            fontsize=10,
        )
        ax.set_ylim(0, 1.15)
        style_axis(ax)

    legend_elements = [get_legend_patch("control"), get_legend_patch("regular")]
    axes[-1].legend(handles=legend_elements, loc="lower right", fontsize=10)

    fig.suptitle(
        "Authentication Success Rate by LLM and Speech Complexity",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "auth_success_rate.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'auth_success_rate.pdf'}")


# =============================================================================
# Paper Output Generation
# =============================================================================


def generate_paper_outputs(
    output_dir: Path,
    df_metrics: pd.DataFrame,
) -> None:
    """
    Generate paper-ready figures and LaTeX tables.

    Outputs are saved to output_dir/../paper/ (shared with voice_analysis)
    """
    # Paper outputs go to shared paper directory at analysis level
    paper_dir = output_dir.parent / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating paper outputs in {paper_dir}")

    if "pass_hat_1" not in df_metrics.columns:
        logger.warning("pass_hat_1 not found. Skipping paper outputs.")
        return

    # -------------------------------------------------------------------------
    # 1. Headline figure (copy from pass_k_by_domain)
    # -------------------------------------------------------------------------
    plot_pass_1_headline(paper_dir, df_metrics)

    # -------------------------------------------------------------------------
    # 2. Main results table (LaTeX): Pass^1 by domain, provider, control/regular
    # -------------------------------------------------------------------------
    _generate_main_results_table(paper_dir, df_metrics)

    # -------------------------------------------------------------------------
    # 3. Ablation table (LaTeX): Retail only, realtime LLMs, all complexities
    # -------------------------------------------------------------------------
    _generate_ablation_table(paper_dir, df_metrics)

    # -------------------------------------------------------------------------
    # 4. Ablation figure
    # -------------------------------------------------------------------------
    plot_pass_1_filtered(
        paper_dir,
        df_metrics,
        domains_filter=["retail"],
        llms_filter=["gemini", "gpt-realtime", "grok", "xai"],
        filename="ablation_retail.pdf",
    )

    logger.info(f"Paper outputs saved to {paper_dir}")


def _generate_main_results_table(
    output_dir: Path,
    df_metrics: pd.DataFrame,
) -> None:
    """
    Generate LaTeX table: Pass^1 by domain, provider, control vs regular with delta.

    Format matches results.tex Table 1 style.
    """
    from experiments.tau_voice.exp.plot_style import (
        get_domain_task_counts,
        get_model_sort_key,
        get_provider_key,
        get_short_llm_name,
    )

    DOMAIN_TASK_COUNTS = get_domain_task_counts()

    domains = [d for d in DOMAINS if d in df_metrics["domain"].unique()]

    # Aggregate per model (by llm, not by provider)
    agg_df = (
        df_metrics.groupby(["domain", "llm", "speech_complexity"])["pass_hat_1"]
        .mean()
        .reset_index()
    )
    agg_df["provider_key"] = agg_df["llm"].apply(get_provider_key)

    rows = []
    for domain in domains:
        domain_data = agg_df[agg_df["domain"] == domain]
        n_tasks = DOMAIN_TASK_COUNTS.get(domain, "?")

        for llm in sorted(domain_data["llm"].unique()):
            llm_data = domain_data[domain_data["llm"] == llm]
            provider_key = llm_data["provider_key"].iloc[0]

            control_rows = llm_data[llm_data["speech_complexity"] == "control"]
            regular_rows = llm_data[llm_data["speech_complexity"] == "regular"]

            control_val = (
                control_rows["pass_hat_1"].values[0] if len(control_rows) > 0 else None
            )
            regular_val = (
                regular_rows["pass_hat_1"].values[0] if len(regular_rows) > 0 else None
            )

            if control_val is None and regular_val is None:
                continue

            model_name = get_short_llm_name(llm, max_len=25)

            rows.append(
                {
                    "domain": domain,
                    "n_tasks": n_tasks,
                    "llm": llm,
                    "provider_key": provider_key,
                    "model": model_name,
                    "control": control_val,
                    "regular": regular_val,
                }
            )

    # Generate LaTeX
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(
        r"\caption{Task completion (pass\^{}1) by model, domain, and condition. \textbf{Bold} indicates best per domain/condition.}"
    )
    lines.append(r"\label{tab:main-results}")
    lines.append(r"\centering")
    lines.append(r"\begin{small}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Domain} & \textbf{Model} & \textbf{Control} & \textbf{Regular} & \textbf{$\Delta$} \\"
    )
    lines.append(r"\midrule")

    for domain in domains:
        domain_rows = [r for r in rows if r["domain"] == domain]
        domain_rows = sorted(
            domain_rows,
            key=lambda r: get_model_sort_key(r["llm"]),
        )
        n_tasks = domain_rows[0]["n_tasks"] if domain_rows else "?"

        # Find best values for bolding
        control_vals = [r["control"] for r in domain_rows if r["control"] is not None]
        regular_vals = [r["regular"] for r in domain_rows if r["regular"] is not None]
        best_control = max(control_vals) if control_vals else None
        best_regular = max(regular_vals) if regular_vals else None

        for i, row in enumerate(domain_rows):
            # Domain label (only on first row of group)
            if i == 0:
                domain_label = rf"\multirow{{{len(domain_rows)}}}{{*}}{{{domain.capitalize()} ({n_tasks})}}"
            else:
                domain_label = ""

            # Format values
            def fmt_val(val, is_best):
                if val is None:
                    return "--"
                pct = f"{round(val * 100)}\\%"
                return rf"\textbf{{{pct}}}" if is_best else pct

            control_str = fmt_val(
                row["control"],
                row["control"] == best_control and row["control"] is not None,
            )
            regular_str = fmt_val(
                row["regular"],
                row["regular"] == best_regular and row["regular"] is not None,
            )

            # Delta
            if row["control"] is not None and row["regular"] is not None:
                delta = round((row["regular"] - row["control"]) * 100)
                delta_str = f"+{delta}\\%" if delta >= 0 else f"$-${abs(delta)}\\%"
            else:
                delta_str = "--"

            lines.append(
                f"{domain_label} & {row['model']} & {control_str} & {regular_str} & {delta_str} \\\\"
            )

        # Add midrule between domains (except after last)
        if domain != domains[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{small}")
    lines.append(r"\end{table}")

    # Write to file
    tex_path = output_dir / "main_results_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")


def _generate_ablation_table(
    output_dir: Path,
    df_metrics: pd.DataFrame,
) -> None:
    """
    Generate LaTeX table: Ablation results for retail domain.

    Shows all complexity levels for realtime LLMs.
    """
    # Filter to retail and realtime LLMs
    retail_data = df_metrics[df_metrics["domain"] == "retail"]

    # Get realtime LLMs
    all_llms = sorted(retail_data["llm"].unique())
    realtime_llms = [
        llm
        for llm in all_llms
        if any(p in llm.lower() for p in ["gemini", "gpt-realtime", "grok", "xai"])
    ]

    if not realtime_llms:
        logger.warning("No realtime LLMs found for ablation table.")
        return

    from experiments.tau_voice.exp.plot_style import (
        get_model_sort_key,
        get_short_llm_name,
    )

    # Complexity display names for ablation
    complexity_labels = {
        "control": "Control (baseline)",
        "control_audio": "+ Background noise",
        "control_accents": "+ Diverse accents",
        "control_behavior": "+ Interruptions",
        "regular": "Regular (all effects)",
    }

    # Get available complexities in order
    tested_complexities = retail_data["speech_complexity"].unique()
    complexities = [c for c in SPEECH_COMPLEXITIES if c in tested_complexities]

    # Sort models by provider order, then model name
    realtime_llms = sorted(realtime_llms, key=get_model_sort_key)

    # Build data matrix keyed by llm
    data = {}
    for llm in realtime_llms:
        data[llm] = {}
        llm_data = retail_data[retail_data["llm"] == llm]
        for complexity in complexities:
            c_data = llm_data[llm_data["speech_complexity"] == complexity]
            data[llm][complexity] = (
                c_data["pass_hat_1"].mean() if len(c_data) > 0 else None
            )

    model_names = [get_short_llm_name(llm, max_len=20) for llm in realtime_llms]

    # Generate LaTeX
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(
        r"\caption{Ablation: impact of acoustic factors on pass\^{}1 (Retail domain).}"
    )
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\centering")
    lines.append(r"\begin{small}")

    # Header with "All" column
    header_cols = " & ".join([rf"\textbf{{{m}}}" for m in model_names])
    lines.append(r"\begin{tabular}{l" + "c" * len(realtime_llms) + "c}")
    lines.append(r"\toprule")
    lines.append(rf"\textbf{{Condition}} & {header_cols} & \textbf{{All}} \\")
    lines.append(r"\midrule")

    # Data rows with deltas from control
    for complexity in complexities:
        label = complexity_labels.get(complexity, complexity)
        vals = []
        model_vals = []
        for llm in realtime_llms:
            val = data[llm].get(complexity)
            control_val = data[llm].get("control")
            if val is not None:
                model_vals.append(val)
                pct = round(val * 100)
                if complexity == "control":
                    vals.append(f"{pct}\\%")
                elif control_val is not None:
                    delta = round((val - control_val) * 100)
                    delta_str = f"+{delta}" if delta >= 0 else str(delta)
                    vals.append(f"{pct}\\% ({delta_str})")
                else:
                    vals.append(f"{pct}\\%")
            else:
                vals.append("--")

        # Compute "All" column (average across models)
        if model_vals:
            avg_val = sum(model_vals) / len(model_vals)
            avg_pct = round(avg_val * 100)
            if complexity == "control":
                vals.append(f"{avg_pct}\\%")
            else:
                control_vals = [
                    data[llm].get("control")
                    for llm in realtime_llms
                    if data[llm].get("control") is not None
                ]
                if control_vals:
                    avg_control = sum(control_vals) / len(control_vals)
                    avg_delta = round((avg_val - avg_control) * 100)
                    delta_str = f"+{avg_delta}" if avg_delta >= 0 else str(avg_delta)
                    vals.append(f"{avg_pct}\\% ({delta_str})")
                else:
                    vals.append(f"{avg_pct}\\%")
        else:
            vals.append("--")

        vals_str = " & ".join(vals)
        lines.append(f"{label} & {vals_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{small}")
    lines.append(r"\end{table}")

    # Write to file
    tex_path = output_dir / "ablation_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")


# =============================================================================
# Main Analysis Function
# =============================================================================


def analyze_results(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    filter_domains: Optional[List[str]] = None,
    results: Optional[List[Tuple[dict, Results]]] = None,
) -> None:
    """
    Main analysis function for tau_voice experiments.

    Args:
        data_dir: Directory containing simulation folders
        output_dir: Directory for output figures (default: data_dir/figs)
        filter_domains: Optional list of domains to include
        results: Optional pre-loaded results to avoid reloading data
    """
    logger.info(f"Analyzing tau_voice results in {data_dir}...")

    # Load results if not provided
    if results is None:
        results = load_simulation_results(data_dir, filter_domains)

    if not results:
        logger.warning("No results found. Exiting.")
        return

    logger.info(f"Analyzing {len(results)} simulation results.")

    # Build metrics DataFrames
    df_metrics, df_pass_hat_k = build_metrics_dataframe(results)

    # Log summary
    unique_llms = df_metrics["llm"].unique()
    unique_domains = df_metrics["domain"].unique()
    unique_complexities = df_metrics["speech_complexity"].unique()
    logger.info(f"LLMs: {list(unique_llms)}")
    logger.info(f"Domains: {list(unique_domains)}")
    logger.info(f"Speech Complexities: {list(unique_complexities)}")

    # Set up output directory
    if output_dir is None:
        output_dir = data_dir / "analysis/performance_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving figures to: {output_dir}")

    # ==========================================================================
    # Pass^k by Domain Analysis (subdirectory)
    # ==========================================================================
    try:
        domain_dir = output_dir / "pass_k_by_domain"
        save_pass_k_by_domain_raw(domain_dir, results)
        save_pass_k_by_domain_analysis(domain_dir, df_metrics)
        plot_pass_k_by_domain(domain_dir, df_metrics)
        plot_pass_1_by_domain(domain_dir, df_metrics)
        plot_pass_1_headline(domain_dir, df_metrics)
        plot_pass_1_all_complexities(domain_dir, df_metrics)
        # Filtered plot: retail only, main realtime LLMs
        plot_pass_1_filtered(
            domain_dir,
            df_metrics,
            domains_filter=["retail"],
            llms_filter=["gemini", "gpt-realtime", "grok", "xai"],
            filename="pass_1_retail_realtime.pdf",
        )
        save_pass_1_complexity_table(domain_dir, df_metrics)
    except Exception as e:
        logger.error(f"Failed to generate pass@k by domain analysis: {e}")

    # Speech environment analysis (persona_name, background_noise_file)
    try:
        df_voice = compute_pass_k_by_persona(results, include_domain=True)
        df_voice_by_complexity = compute_pass_k_by_persona(
            results, include_complexity=True, include_domain=True
        )
        df_noise = compute_pass_k_by_background_noise(results, include_domain=True)
        df_noise_by_complexity = compute_pass_k_by_background_noise(
            results, include_complexity=True, include_domain=True
        )

        # Log speech environment summary
        df_sims = build_simulation_level_dataframe(results)
        unique_personas = df_sims["persona_name"].dropna().nunique()
        unique_noise_files = df_sims["background_noise_file"].dropna().nunique()
        logger.info(f"Unique personas: {unique_personas}")
        logger.info(f"Unique background noise files: {unique_noise_files}")

        # ==========================================================================
        # Pass^k by Persona Analysis (subdirectory)
        # ==========================================================================
        persona_dir = output_dir / "pass_k_by_persona"
        save_pass_k_by_persona_raw(persona_dir, results)
        save_pass_k_by_persona_analysis(persona_dir, df_voice_by_complexity)
        plot_pass_k_by_persona(persona_dir, df_voice, df_voice_by_complexity, max_k=1)

        # ==========================================================================
        # Pass^k by Background Noise Analysis (subdirectory)
        # ==========================================================================
        noise_dir = output_dir / "pass_k_by_background_noise"
        save_pass_k_by_background_noise_raw(noise_dir, results)
        save_pass_k_by_background_noise_analysis(noise_dir, df_noise_by_complexity)
        plot_pass_k_by_background_noise(
            noise_dir, df_noise, df_noise_by_complexity, max_k=1
        )

    except Exception as e:
        logger.error(f"Failed to generate speech environment analysis: {e}")

    # Simulation counts breakdown
    try:
        plot_simulation_counts(output_dir, results)
    except Exception as e:
        logger.error(f"Failed to generate simulation counts plot: {e}")

    # Task pass/fail grid
    try:
        plot_task_pass_fail_grid(output_dir, results)
    except Exception as e:
        logger.error(f"Failed to generate task pass/fail grid: {e}")

    # ==========================================================================
    # Termination Analysis
    # ==========================================================================
    try:
        termination_dir = output_dir / "termination_analysis"
        df_termination = save_termination_raw(termination_dir, results)
        save_termination_analysis(termination_dir, df_termination)
        plot_termination_reasons(termination_dir, df_termination)
    except Exception as e:
        logger.error(f"Failed to generate termination analysis: {e}")

    # ==========================================================================
    # Duration Analysis
    # ==========================================================================
    try:
        duration_dir = output_dir / "duration_analysis"
        df_duration = save_duration_raw(duration_dir, results)
        save_duration_analysis(duration_dir, df_duration)
        plot_duration(duration_dir, df_duration)
        plot_duration_distribution(duration_dir, df_duration)
        save_duration_by_outcome_analysis(duration_dir, df_duration)
        plot_duration_by_outcome(duration_dir, df_duration)
    except Exception as e:
        logger.error(f"Failed to generate duration analysis: {e}")

    # ==========================================================================
    # Tool Call Count Analysis
    # ==========================================================================
    try:
        tool_calls_dir = output_dir / "tool_calls_analysis"
        df_tool_calls = save_tool_calls_raw(tool_calls_dir, results)
        save_tool_calls_analysis(tool_calls_dir, df_tool_calls)
        save_tool_calls_by_outcome_analysis(tool_calls_dir, df_tool_calls)
        plot_tool_calls(tool_calls_dir, df_tool_calls)
        plot_tool_calls_by_outcome(tool_calls_dir, df_tool_calls)
        plot_tool_calls_success_failure(tool_calls_dir, df_tool_calls)
    except Exception as e:
        logger.error(f"Failed to generate tool calls analysis: {e}")

    # ==========================================================================
    # Action Success Analysis (Tool Call Success/Failure)
    # ==========================================================================
    try:
        action_success_dir = output_dir / "action_success"
        df_action_raw = save_action_success_raw(action_success_dir, results)
        save_action_success_analysis(action_success_dir, df_action_raw)
        plot_action_success(action_success_dir, df_action_raw)
        plot_action_success_summary(action_success_dir, df_action_raw)
    except Exception as e:
        logger.error(f"Failed to generate action success analysis: {e}")

    # ==========================================================================
    # Reward-Based Action Analysis (expected vs actual actions)
    # ==========================================================================
    try:
        reward_action_dir = output_dir / "reward_action_success"
        save_reward_action_analysis(reward_action_dir, results)
    except Exception as e:
        logger.error(f"Failed to generate reward action analysis: {e}")

    # ==========================================================================
    # Review Analysis (LLM-based conversation review)
    # ==========================================================================
    try:
        review_dir = output_dir / "review_analysis"
        df_review_summary, df_review_errors = save_review_analysis(review_dir, results)
        # Agent error analysis
        plot_agent_errors_per_simulation(
            review_dir, df_review_summary, df_review_errors
        )
        plot_agent_error_breakdown(review_dir, df_review_errors)
        plot_agent_error_by_severity(review_dir, df_review_errors)
        plot_agent_error_breakdown_by_complexity(review_dir, df_review_errors)
        # User error analysis
        plot_user_errors_per_simulation(review_dir, df_review_summary, df_review_errors)
        plot_user_error_breakdown(review_dir, df_review_errors)
        plot_user_error_by_severity(review_dir, df_review_errors)
        # First critical error source analysis
        plot_first_critical_source(review_dir, df_review_summary)
    except Exception as e:
        logger.error(f"Failed to generate review analysis: {e}")

    # ==========================================================================
    # Authentication Analysis
    # ==========================================================================
    try:
        auth_dir = output_dir / "auth_analysis"
        df_auth = save_auth_analysis(auth_dir, results)
        plot_auth_success(auth_dir, df_auth)
        plot_auth_success_rate(auth_dir, df_auth)
    except Exception as e:
        logger.error(f"Failed to generate auth analysis: {e}")

    # NOTE: Voice/turn-taking metrics analysis has moved to voice_analysis.py
    # Run: python -m experiments.tau_voice.exp.voice_analysis --data-dir <path>

    # Save CSV
    save_metrics_csv(output_dir, df_metrics)

    # Note: Paper outputs are now generated by paper_outputs.py via run_all_analysis.py

    logger.info("Analysis complete!")


# =============================================================================
# Plots-Only Mode (regenerate from existing CSVs)
# =============================================================================


def regenerate_plots_from_csv(output_dir: Path) -> None:
    """
    Regenerate all plots from existing CSV files without reloading data.

    This is useful for iterating on plot styling without recomputing data.

    Args:
        output_dir: Directory containing the analysis subdirectories with CSV files.
    """
    logger.info(f"Regenerating plots from existing CSVs in {output_dir}...")

    # ==========================================================================
    # Pass^k by Domain
    # ==========================================================================
    try:
        domain_dir = output_dir / "pass_k_by_domain"
        analysis_csv = domain_dir / f"{domain_dir.name}_analysis.csv"
        if analysis_csv.exists():
            df_metrics = pd.read_csv(analysis_csv)
            plot_pass_k_by_domain(domain_dir, df_metrics)
            plot_pass_1_by_domain(domain_dir, df_metrics)
            plot_pass_1_headline(domain_dir, df_metrics)
            plot_pass_1_all_complexities(domain_dir, df_metrics)
            # Filtered plot: retail only, main realtime LLMs
            plot_pass_1_filtered(
                domain_dir,
                df_metrics,
                domains_filter=["retail"],
                llms_filter=["gemini", "gpt-realtime", "grok", "xai"],
                filename="pass_1_retail_realtime.pdf",
            )
            save_pass_1_complexity_table(domain_dir, df_metrics)
            logger.info("Regenerated: pass_k_by_domain plots")
        else:
            logger.warning(f"Skipping pass_k_by_domain: {analysis_csv} not found")
    except Exception as e:
        logger.error(f"Failed to regenerate pass_k_by_domain plots: {e}")

    # ==========================================================================
    # Pass^k by Persona
    # ==========================================================================
    try:
        persona_dir = output_dir / "pass_k_by_persona"
        analysis_csv = persona_dir / f"{persona_dir.name}_analysis.csv"
        if analysis_csv.exists():
            df_voice_by_complexity = pd.read_csv(analysis_csv)
            # Also need the non-complexity version for the plot
            # Aggregate to remove complexity dimension
            group_cols = ["llm", "persona_name"]
            if "domain" in df_voice_by_complexity.columns:
                group_cols.insert(0, "domain")
            pass_cols = [
                c for c in df_voice_by_complexity.columns if c.startswith("pass_hat_")
            ]
            if pass_cols:
                df_voice = (
                    df_voice_by_complexity.groupby(group_cols)[pass_cols]
                    .mean()
                    .reset_index()
                )
            else:
                df_voice = df_voice_by_complexity
            plot_pass_k_by_persona(
                persona_dir, df_voice, df_voice_by_complexity, max_k=1
            )
            logger.info("Regenerated: pass_k_by_persona plots")
        else:
            logger.warning(f"Skipping pass_k_by_persona: {analysis_csv} not found")
    except Exception as e:
        logger.error(f"Failed to regenerate pass_k_by_persona plots: {e}")

    # ==========================================================================
    # Pass^k by Background Noise
    # ==========================================================================
    try:
        noise_dir = output_dir / "pass_k_by_background_noise"
        analysis_csv = noise_dir / f"{noise_dir.name}_analysis.csv"
        if analysis_csv.exists():
            df_noise_by_complexity = pd.read_csv(analysis_csv)
            # Aggregate to remove complexity dimension
            group_cols = ["llm", "background_noise_file"]
            if "background_noise_display" in df_noise_by_complexity.columns:
                group_cols = ["llm", "background_noise_display"]
            if "domain" in df_noise_by_complexity.columns:
                group_cols.insert(0, "domain")
            pass_cols = [
                c for c in df_noise_by_complexity.columns if c.startswith("pass_hat_")
            ]
            if pass_cols:
                df_noise = (
                    df_noise_by_complexity.groupby(group_cols)[pass_cols]
                    .mean()
                    .reset_index()
                )
            else:
                df_noise = df_noise_by_complexity
            plot_pass_k_by_background_noise(
                noise_dir, df_noise, df_noise_by_complexity, max_k=1
            )
            logger.info("Regenerated: pass_k_by_background_noise plots")
        else:
            logger.warning(
                f"Skipping pass_k_by_background_noise: {analysis_csv} not found"
            )
    except Exception as e:
        logger.error(f"Failed to regenerate pass_k_by_background_noise plots: {e}")

    # ==========================================================================
    # Termination Analysis
    # ==========================================================================
    try:
        termination_dir = output_dir / "termination_analysis"
        raw_csv = termination_dir / f"{termination_dir.name}_raw.csv"
        if raw_csv.exists():
            df_termination = pd.read_csv(raw_csv)
            plot_termination_reasons(termination_dir, df_termination)
            logger.info("Regenerated: termination_analysis plots")
        else:
            logger.warning(f"Skipping termination_analysis: {raw_csv} not found")
    except Exception as e:
        logger.error(f"Failed to regenerate termination_analysis plots: {e}")

    # ==========================================================================
    # Duration Analysis
    # ==========================================================================
    try:
        duration_dir = output_dir / "duration_analysis"
        raw_csv = duration_dir / f"{duration_dir.name}_raw.csv"
        if raw_csv.exists():
            df_duration = pd.read_csv(raw_csv)
            plot_duration(duration_dir, df_duration)
            plot_duration_distribution(duration_dir, df_duration)
            plot_duration_by_outcome(duration_dir, df_duration)
            logger.info("Regenerated: duration_analysis plots")
        else:
            logger.warning(f"Skipping duration_analysis: {raw_csv} not found")
    except Exception as e:
        logger.error(f"Failed to regenerate duration_analysis plots: {e}")

    # ==========================================================================
    # Tool Calls Analysis
    # ==========================================================================
    try:
        tool_calls_dir = output_dir / "tool_calls_analysis"
        raw_csv = tool_calls_dir / f"{tool_calls_dir.name}_raw.csv"
        if raw_csv.exists():
            df_tool_calls = pd.read_csv(raw_csv)
            plot_tool_calls(tool_calls_dir, df_tool_calls)
            plot_tool_calls_by_outcome(tool_calls_dir, df_tool_calls)
            plot_tool_calls_success_failure(tool_calls_dir, df_tool_calls)
            logger.info("Regenerated: tool_calls_analysis plots")
        else:
            logger.warning(f"Skipping tool_calls_analysis: {raw_csv} not found")
    except Exception as e:
        logger.error(f"Failed to regenerate tool_calls_analysis plots: {e}")

    # ==========================================================================
    # Action Success Analysis
    # ==========================================================================
    try:
        action_success_dir = output_dir / "action_success"
        raw_csv = action_success_dir / f"{action_success_dir.name}_raw.csv"
        if raw_csv.exists():
            df_action_raw = pd.read_csv(raw_csv)
            plot_action_success(action_success_dir, df_action_raw)
            plot_action_success_summary(action_success_dir, df_action_raw)
            logger.info("Regenerated: action_success plots")
        else:
            logger.warning(f"Skipping action_success: {raw_csv} not found")
    except Exception as e:
        logger.error(f"Failed to regenerate action_success plots: {e}")

    # ==========================================================================
    # Review Analysis
    # ==========================================================================
    try:
        review_dir = output_dir / "review_analysis"
        summary_csv = review_dir / "review_summary.csv"
        errors_csv = review_dir / "review_errors.csv"
        if summary_csv.exists() and errors_csv.exists():
            df_review_summary = pd.read_csv(summary_csv)
            df_review_errors = pd.read_csv(errors_csv)
            plot_agent_errors_per_simulation(
                review_dir, df_review_summary, df_review_errors
            )
            plot_agent_error_breakdown(review_dir, df_review_errors)
            plot_agent_error_by_severity(review_dir, df_review_errors)
            plot_agent_error_breakdown_by_complexity(review_dir, df_review_errors)
            plot_user_errors_per_simulation(
                review_dir, df_review_summary, df_review_errors
            )
            plot_user_error_breakdown(review_dir, df_review_errors)
            plot_user_error_by_severity(review_dir, df_review_errors)
            plot_first_critical_source(review_dir, df_review_summary)
            logger.info("Regenerated: review_analysis plots")
        else:
            logger.warning(f"Skipping review_analysis: CSV files not found")
    except Exception as e:
        logger.error(f"Failed to regenerate review_analysis plots: {e}")

    # ==========================================================================
    # Authentication Analysis
    # ==========================================================================
    try:
        auth_dir = output_dir / "auth_analysis"
        raw_csv = auth_dir / "auth_raw.csv"
        if raw_csv.exists():
            df_auth = pd.read_csv(raw_csv)
            plot_auth_success(auth_dir, df_auth)
            plot_auth_success_rate(auth_dir, df_auth)
            logger.info("Regenerated: auth_analysis plots")
        else:
            logger.warning(f"Skipping auth_analysis: {raw_csv} not found")
    except Exception as e:
        logger.error(f"Failed to regenerate auth_analysis plots: {e}")

    # Note: Paper outputs are now generated by paper_outputs.py via run_all_analysis.py

    logger.info("Plot regeneration complete!")


# =============================================================================
# CLI
# =============================================================================


def get_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze tau_voice experiment results and generate pass^k plots."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing simulation result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output figures. Defaults to data_dir/analysis/performance_analysis.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific domains (e.g., retail airline telecom).",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plots from existing CSV files without reloading data.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directory contents before running analysis.",
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

    # Handle --clean flag: delete output directory contents before running
    if args.clean and not args.plots_only:
        # Determine output_dir for cleaning (same logic as analyze_results)
        clean_dir = (
            output_dir if output_dir else data_dir / "analysis/performance_analysis"
        )
        if clean_dir.exists():
            logger.warning(f"Cleaning output directory: {clean_dir}")
            shutil.rmtree(clean_dir)
            logger.info(f"Deleted: {clean_dir}")

    # Handle --plots-only mode
    if args.plots_only:
        # In plots-only mode, output_dir must exist (or default to data_dir/analysis/performance_analysis)
        if output_dir is None:
            output_dir = data_dir / "analysis/performance_analysis"
        if not output_dir.exists():
            logger.error(
                f"Output directory {output_dir} does not exist. Cannot use --plots-only."
            )
            return
        regenerate_plots_from_csv(output_dir)
    else:
        analyze_results(
            data_dir=data_dir,
            output_dir=output_dir,
            filter_domains=args.domains,
        )


if __name__ == "__main__":
    main()
