#!/usr/bin/env python3
"""
Paper Output Generation for Tau Voice Experiments.

Generates publication-ready figures and LaTeX tables from existing CSV data.
All outputs go to a shared `paper/` directory at the analysis level.

Usage:
    python -m experiments.tau_voice.exp.paper_outputs --output-dir ./data/exp/tau-voice-results/analysis
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from experiments.tau_voice.exp.plot_style import (
    DOMAINS,
    SPEECH_COMPLEXITIES,
    SPEECH_COMPLEXITY_COLORS,
    get_complexity_display_name,
    get_domain_task_counts,
    get_model_sort_key,
    get_provider_display,
    get_provider_key,
    get_short_llm_name,
)

# =============================================================================
# Constants
# =============================================================================

# Task counts per domain - loaded from registry
DOMAIN_TASK_COUNTS = get_domain_task_counts()


# =============================================================================
# Main Entry Point
# =============================================================================


def generate_all_paper_outputs(
    analysis_dir: Path,
    performance_only: bool = False,
    voice_only: bool = False,
    copy_to_paper_dir: Optional[Path] = None,
) -> None:
    """
    Generate all paper outputs from existing CSV data.

    Args:
        analysis_dir: Root analysis directory containing performance_analysis/ and voice_analysis/
        performance_only: Only generate performance outputs
        voice_only: Only generate voice outputs
        copy_to_paper_dir: If provided, copy outputs to this directory (e.g., papers/tau-voice/results/)
    """
    paper_dir = analysis_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating paper outputs in {paper_dir}")

    if not voice_only:
        _generate_performance_paper_outputs(analysis_dir, paper_dir)

    if not performance_only:
        _generate_voice_paper_outputs(analysis_dir, paper_dir)

    logger.info(f"Paper outputs saved to {paper_dir}")

    # Copy to paper directory if requested
    if copy_to_paper_dir:
        import shutil

        copy_to_paper_dir = Path(copy_to_paper_dir)
        copy_to_paper_dir.mkdir(parents=True, exist_ok=True)

        # Copy all .tex, .csv, and .pdf files
        for pattern in ["*.tex", "*.csv", "*.pdf"]:
            for src_file in paper_dir.glob(pattern):
                dst_file = copy_to_paper_dir / src_file.name
                shutil.copy2(src_file, dst_file)
                logger.info(f"Copied: {src_file.name} -> {copy_to_paper_dir}")

        logger.info(f"Paper outputs copied to {copy_to_paper_dir}")


def _generate_performance_paper_outputs(analysis_dir: Path, paper_dir: Path) -> None:
    """Generate performance analysis paper outputs."""

    perf_dir = analysis_dir / "performance_analysis"
    domain_csv = perf_dir / "pass_k_by_domain" / "pass_k_by_domain_analysis.csv"

    if not domain_csv.exists():
        logger.warning(f"Performance CSV not found: {domain_csv}")
        return

    df_metrics = pd.read_csv(domain_csv)
    logger.info("Generating performance paper outputs...")

    # 1. Main results table
    _generate_main_results_table(paper_dir, df_metrics)

    # 2. Ablation table
    _generate_ablation_table(paper_dir, df_metrics)

    # 2b. Ablation table (single factor only, no pairwise)
    _generate_ablation_table_single(paper_dir, df_metrics)

    # 3. Headline figures
    _plot_pass_1_headline(paper_dir, df_metrics)
    _plot_pass_1_headline_simple(paper_dir, df_metrics)

    # 4. Ablation figure
    _plot_ablation_figure(paper_dir, df_metrics)

    # 5. Speech complexity conditions table
    _generate_conditions_table(paper_dir)

    # 6. Voice vs Text comparison table
    _generate_voice_vs_text_table(paper_dir, df_metrics)

    # 7. Combined Text/Control/Realistic comparison table
    _generate_combined_comparison_table(paper_dir, df_metrics)

    logger.info("  ✓ Performance paper outputs complete")


def _generate_voice_paper_outputs(analysis_dir: Path, paper_dir: Path) -> None:
    """Generate voice analysis paper outputs.

    Uses the unified voice_quality_analysis.csv which is computed from
    exactly the same event extraction as the speech timeline visualization.
    """
    voice_dir = analysis_dir / "voice_analysis"

    # Try new unified voice quality CSV first (preferred)
    voice_quality_csv = voice_dir / "voice_quality" / "voice_quality_analysis.csv"
    interruption_csv = (
        voice_dir / "interruption_handling" / "interruption_handling_analysis.csv"
    )

    if voice_quality_csv.exists():
        df_voice = pd.read_csv(voice_quality_csv)
        # Load interruption data for agent interruption rate
        df_interruption = None
        if interruption_csv.exists():
            df_interruption = pd.read_csv(interruption_csv)
        logger.info(
            "Generating voice paper outputs from unified voice_quality_analysis.csv..."
        )
        _generate_voice_quality_table_unified(paper_dir, df_voice, df_interruption)
        _generate_voice_quality_aggregated_table(paper_dir, df_voice, df_interruption)
        _generate_vertical_voice_quality_table(paper_dir, df_voice, df_interruption)
        _generate_core_metrics_table(paper_dir, df_voice, df_interruption)
        _generate_full_voice_quality_table(paper_dir, df_voice, df_interruption)
        logger.info("  ✓ Voice paper outputs complete")
        return

    # Fallback to legacy CSVs for backwards compatibility
    latency_csv = voice_dir / "response_latency" / "response_latency_analysis.csv"
    interruption_csv = (
        voice_dir / "interruption_handling" / "interruption_handling_analysis.csv"
    )

    if not latency_csv.exists() or not interruption_csv.exists():
        logger.warning("Voice CSVs not found (neither unified nor legacy)")
        return

    df_latency = pd.read_csv(latency_csv)
    df_interruption = pd.read_csv(interruption_csv)
    logger.info("Generating voice paper outputs from legacy CSVs...")

    # Voice quality table (legacy)
    _generate_voice_quality_table(paper_dir, df_latency, df_interruption)

    logger.info("  ✓ Voice paper outputs complete")


# =============================================================================
# Performance Tables
# =============================================================================


def _generate_main_results_table(output_dir: Path, df_metrics: pd.DataFrame) -> None:
    """Generate LaTeX table: Pass^1 by domain, provider, control vs regular with delta."""
    # Aggregate per model (by llm, not by provider)
    df_metrics = df_metrics.copy()
    agg_df = (
        df_metrics.groupby(["domain", "llm", "speech_complexity"])["pass_hat_1"]
        .mean()
        .reset_index()
    )

    domains = [d for d in DOMAINS if d in df_metrics["domain"].unique()]

    # Build one row per model per domain
    rows = []
    for domain in domains:
        domain_data = agg_df[agg_df["domain"] == domain]
        n_tasks = DOMAIN_TASK_COUNTS.get(domain, "?")

        for llm in sorted(domain_data["llm"].unique(), key=get_model_sort_key):
            llm_data = domain_data[domain_data["llm"] == llm]

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
                    "provider": get_provider_display(get_provider_key(llm)),
                    "model": model_name,
                    "control": control_val,
                    "regular": regular_val,
                }
            )

    # Build "All" aggregate rows (average across domains per model)
    all_rows = []
    total_tasks = sum(DOMAIN_TASK_COUNTS.get(d, 0) for d in domains)
    llms_in_data = sorted(set(r["llm"] for r in rows), key=get_model_sort_key)
    for llm in llms_in_data:
        llm_rows = [r for r in rows if r["llm"] == llm]
        control_vals = [r["control"] for r in llm_rows if r["control"] is not None]
        regular_vals = [r["regular"] for r in llm_rows if r["regular"] is not None]

        control_avg = sum(control_vals) / len(control_vals) if control_vals else None
        regular_avg = sum(regular_vals) / len(regular_vals) if regular_vals else None

        if control_avg is not None or regular_avg is not None:
            all_rows.append(
                {
                    "domain": "all",
                    "n_tasks": total_tasks,
                    "llm": llm,
                    "provider": get_provider_display(get_provider_key(llm)),
                    "model": get_short_llm_name(llm, max_len=25),
                    "control": control_avg,
                    "regular": regular_avg,
                }
            )

    # Generate LaTeX
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(
        r"\caption{Task completion (pass@1) by model, domain, and condition. \textbf{Bold} indicates best per domain/condition.}"
    )
    lines.append(r"\label{tab:main-results}")
    lines.append(r"\centering")
    lines.append(r"\begin{small}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Domain} & \textbf{Model} & \textbf{Control} & \textbf{Realistic} & \textbf{$\Delta$} & \textbf{$\Delta_{\%}$} \\"
    )
    lines.append(r"\midrule")

    def fmt_val(val, is_best):
        if val is None:
            return "--"
        pct = f"{round(val * 100)}\\%"
        return rf"\textbf{{{pct}}}" if is_best else pct

    def _render_group(group_rows, domain_label_text):
        control_vals = [r["control"] for r in group_rows if r["control"] is not None]
        regular_vals = [r["regular"] for r in group_rows if r["regular"] is not None]
        best_control = max(control_vals) if control_vals else None
        best_regular = max(regular_vals) if regular_vals else None

        for i, row in enumerate(group_rows):
            if i == 0:
                domain_label = (
                    rf"\multirow{{{len(group_rows)}}}{{*}}{{{domain_label_text}}}"
                )
            else:
                domain_label = ""

            control_str = fmt_val(
                row["control"],
                row["control"] == best_control and row["control"] is not None,
            )
            regular_str = fmt_val(
                row["regular"],
                row["regular"] == best_regular and row["regular"] is not None,
            )

            if row["control"] is not None and row["regular"] is not None:
                delta = round((row["regular"] - row["control"]) * 100)
                delta_str = f"+{delta}\\%" if delta >= 0 else f"$-${abs(delta)}\\%"
                if row["control"] != 0:
                    delta_rel = (row["regular"] - row["control"]) / row["control"] * 100
                    delta_rel_str = (
                        f"+{delta_rel:.1f}\\%"
                        if delta_rel >= 0
                        else f"$-${abs(delta_rel):.1f}\\%"
                    )
                else:
                    delta_rel_str = "--"
            else:
                delta_str = "--"
                delta_rel_str = "--"

            lines.append(
                f"{domain_label} & {row['model']} & {control_str} & {regular_str} & {delta_str} & {delta_rel_str} \\\\"
            )

    # Add "All" rows first
    if all_rows:
        _render_group(all_rows, f"All ({total_tasks})")
        lines.append(r"\midrule")

    # Add domain-specific rows
    for domain in domains:
        domain_rows = sorted(
            [r for r in rows if r["domain"] == domain],
            key=lambda r: get_model_sort_key(r["llm"]),
        )
        n_tasks = domain_rows[0]["n_tasks"] if domain_rows else "?"
        _render_group(domain_rows, f"{domain.capitalize()} ({n_tasks})")

        if domain != domains[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{small}")
    lines.append(r"\end{table}")

    tex_path = output_dir / "main_results_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")

    # Also save as CSV
    csv_rows = []
    for row in all_rows + rows:
        delta = None
        delta_rel = None
        if row["control"] is not None and row["regular"] is not None:
            delta = row["regular"] - row["control"]
            if row["control"] != 0:
                delta_rel = (row["regular"] - row["control"]) / row["control"]
        csv_rows.append(
            {
                "domain": "All" if row["domain"] == "all" else row["domain"],
                "n_tasks": row["n_tasks"],
                "provider": row["provider"],
                "model": row["model"],
                "control": row["control"],
                "regular": row["regular"],
                "delta": delta,
                "delta_relative": delta_rel,
            }
        )
    df_csv = pd.DataFrame(csv_rows)
    csv_path = output_dir / "main_results_table.csv"
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")


def _render_ablation_table(
    llms: list,
    model_names: list,
    data: dict,
    complexities: list,
    complexity_labels: dict,
    caption: str,
    label: str,
) -> tuple:
    """Shared renderer for ablation tables (full and single-factor).

    Returns (lines, csv_rows) where lines is a list of LaTeX strings
    and csv_rows is a list of dicts for CSV output.
    """
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\centering")
    lines.append(r"\begin{small}")

    header_cols = " & ".join([rf"\textbf{{{m}}}" for m in model_names])
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * len(llms) + "|c}")
    lines.append(r"\toprule")
    lines.append(rf"\textbf{{Condition}} & {header_cols} & \textbf{{All}} \\")
    lines.append(r"\midrule")

    csv_rows = []
    for complexity in complexities:
        lbl = complexity_labels.get(complexity, complexity)

        row_values = []
        for llm in llms:
            val = data[llm].get(complexity)
            control_val = data[llm].get("control")
            row_values.append((llm, val, control_val))

        model_vals = [v for _, v, _ in row_values if v is not None]
        if model_vals:
            avg_val = sum(model_vals) / len(model_vals)
            control_vals_list = [
                data[llm].get("control")
                for llm in llms
                if data[llm].get("control") is not None
            ]
            avg_control = (
                sum(control_vals_list) / len(control_vals_list)
                if control_vals_list
                else None
            )
            row_values.append(("All", avg_val, avg_control))
        else:
            row_values.append(("All", None, None))

        valid_vals = [v for _, v, _ in row_values if v is not None]
        best_val = max(valid_vals) if valid_vals else None

        vals = []
        for name, val, control_val in row_values:
            if val is not None:
                pct = round(val * 100)
                is_best = val == best_val

                if complexity == "control":
                    cell = f"{pct}\\%"
                elif control_val is not None:
                    delta = round((val - control_val) * 100)
                    delta_str = f"+{delta}" if delta >= 0 else str(delta)
                    if control_val != 0:
                        delta_rel = (val - control_val) / control_val * 100
                        delta_rel_str = (
                            f"+{delta_rel:.1f}\\%"
                            if delta_rel >= 0
                            else f"{delta_rel:.1f}\\%"
                        )
                        cell = f"{pct}\\% ({delta_str}, {delta_rel_str})"
                    else:
                        cell = f"{pct}\\% ({delta_str})"
                else:
                    cell = f"{pct}\\%"

                if is_best:
                    vals.append(rf"\textbf{{{cell}}}")
                else:
                    vals.append(cell)
            else:
                vals.append("--")

        vals_str = " & ".join(vals)
        lines.append(f"{lbl} & {vals_str} \\\\")

        # CSV row
        row_data = {"condition": complexity}
        control_vals_for_avg = []
        model_vals_for_avg = []
        for llm, mname in zip(llms, model_names):
            val = data[llm].get(complexity)
            control_val = data[llm].get("control")
            row_data[f"{mname}_value"] = val
            if val is not None:
                model_vals_for_avg.append(val)
            if complexity != "control" and control_val is not None and val is not None:
                row_data[f"{mname}_delta"] = val - control_val
                row_data[f"{mname}_delta_rel"] = (
                    (val - control_val) / control_val if control_val != 0 else None
                )
            else:
                row_data[f"{mname}_delta"] = None
                row_data[f"{mname}_delta_rel"] = None
            if control_val is not None:
                control_vals_for_avg.append(control_val)

        if model_vals_for_avg:
            avg_val = sum(model_vals_for_avg) / len(model_vals_for_avg)
            row_data["all_value"] = avg_val
            if complexity != "control" and control_vals_for_avg:
                avg_control = sum(control_vals_for_avg) / len(control_vals_for_avg)
                row_data["all_delta"] = avg_val - avg_control
                row_data["all_delta_rel"] = (
                    (avg_val - avg_control) / avg_control if avg_control != 0 else None
                )
            else:
                row_data["all_delta"] = None
                row_data["all_delta_rel"] = None
        else:
            row_data["all_value"] = None
            row_data["all_delta"] = None
            row_data["all_delta_rel"] = None
        csv_rows.append(row_data)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{small}")
    lines.append(r"\end{table}")

    return lines, csv_rows


def _generate_ablation_table(output_dir: Path, df_metrics: pd.DataFrame) -> None:
    """Generate LaTeX table: Ablation results for retail domain."""

    # Filter to retail and realtime LLMs
    retail_data = df_metrics[df_metrics["domain"] == "retail"]

    all_llms = sorted(retail_data["llm"].unique())
    realtime_llms = [
        llm
        for llm in all_llms
        if any(p in llm.lower() for p in ["gemini", "gpt-realtime", "grok", "xai"])
    ]

    if not realtime_llms:
        logger.warning("No realtime LLMs found for ablation table.")
        return

    complexity_labels = {
        "control": "Clean",
        "control_audio": "+ Noise",
        "control_accents": "+ Accents",
        "control_behavior": "+ Interrupts",
        "control_audio_accents": "+ Noise + Accents",
        "control_audio_behavior": "+ Noise + Interrupts",
        "control_accents_behavior": "+ Accents + Interrupts",
        "regular": "Realistic",
    }

    tested_complexities = retail_data["speech_complexity"].unique()
    complexities = [c for c in SPEECH_COMPLEXITIES if c in tested_complexities]

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

    lines, csv_rows = _render_ablation_table(
        realtime_llms,
        model_names,
        data,
        complexities,
        complexity_labels,
        caption=r"Ablation: impact of acoustic factors on pass@1 (Retail domain).",
        label="tab:ablation",
    )

    tex_path = output_dir / "ablation_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")

    df_csv = pd.DataFrame(csv_rows)
    csv_path = output_dir / "ablation_table.csv"
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")


def _generate_ablation_table_single(output_dir: Path, df_metrics: pd.DataFrame) -> None:
    """Generate LaTeX table: Single-factor ablation results for retail domain (no pairwise)."""

    # Filter to retail and realtime LLMs
    retail_data = df_metrics[df_metrics["domain"] == "retail"]

    all_llms = sorted(retail_data["llm"].unique())
    realtime_llms = [
        llm
        for llm in all_llms
        if any(p in llm.lower() for p in ["gemini", "gpt-realtime", "grok", "xai"])
    ]

    if not realtime_llms:
        logger.warning("No realtime LLMs found for single ablation table.")
        return

    single_complexities = [
        "control",
        "control_audio",
        "control_accents",
        "control_behavior",
        "regular",
    ]

    complexity_labels = {
        "control": "Clean",
        "control_audio": "+ Noise",
        "control_accents": "+ Accents",
        "control_behavior": "+ Interrupts",
        "regular": "Realistic",
    }

    tested_complexities = retail_data["speech_complexity"].unique()
    complexities = [c for c in single_complexities if c in tested_complexities]

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

    lines, csv_rows = _render_ablation_table(
        realtime_llms,
        model_names,
        data,
        complexities,
        complexity_labels,
        caption=r"Ablation: impact of individual acoustic factors on pass@1 (Retail domain).",
        label="tab:ablation-single",
    )

    tex_path = output_dir / "ablation_table_single.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")

    df_csv = pd.DataFrame(csv_rows)
    csv_path = output_dir / "ablation_table_single.csv"
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")


def _generate_conditions_table(output_dir: Path) -> None:
    """Generate LaTeX tables showing speech complexity conditions from presets.

    Generates two tables:
    1. conditions_table.tex - Control vs Realistic with actual values
    2. conditions_ablation_table.tex - All conditions with checkmarks
    """
    from tau2.user_simulation_voice_presets import COMPLEXITY_CONFIGS

    # Define all conditions
    all_conditions = [
        "control",
        # Single-feature ablations
        "control_audio",
        "control_accents",
        "control_behavior",
        # Pairwise ablations
        "control_audio_accents",
        "control_audio_behavior",
        "control_accents_behavior",
        # Realistic
        "regular",
    ]
    condition_labels = {
        "control": "Cln",
        # Single-feature ablations
        "control_audio": "+N",
        "control_accents": "+A",
        "control_behavior": "+I",
        # Pairwise ablations
        "control_audio_accents": "+NA",
        "control_audio_behavior": "+NI",
        "control_accents_behavior": "+AI",
        # Realistic
        "regular": "Real",
    }

    # Define settings grouped by category (matching ablation conditions)
    setting_groups = [
        (
            "Accents",
            [
                (
                    "personas",
                    "Personas",
                    lambda c: _get_persona_desc(c),
                    lambda c: c.get("persona_names") != _get_control_personas(),
                ),
            ],
        ),
        (
            "Audio/Channel",
            [
                (
                    "background_noise",
                    "Background noise",
                    lambda c: _get_enabled_desc(
                        c, "enable_background_noise", "Indoor/outdoor", "None"
                    ),
                    lambda c: c.get("enable_background_noise", False),
                ),
                (
                    "burst_noise",
                    "Burst noise",
                    lambda c: _get_burst_desc(c),
                    lambda c: c.get("enable_burst_noise", False),
                ),
                (
                    "frame_drops",
                    "Frame drops",
                    lambda c: _get_frame_drop_desc(c),
                    lambda c: c.get("frame_drop_rate", 0) > 0,
                ),
                (
                    "telephony",
                    "Telephony",
                    lambda c: "G.711 $\\mu$-law 8kHz"
                    if c.get("telephony_enabled", True)
                    else "None",
                    lambda c: c.get("telephony_enabled", True),
                ),
                (
                    "muffling",
                    "Muffling",
                    lambda c: _get_enabled_desc(
                        c, "enable_muffling", "Dynamic", "None"
                    ),
                    lambda c: c.get("enable_muffling", False),
                ),
            ],
        ),
        (
            "User Behavior",
            [
                # vocal_tics and non_directed default to True in voice_config.py, only CONTROL explicitly disables them
                (
                    "vocal_tics",
                    "Involuntary sounds",
                    lambda c: _get_enabled_desc(
                        c, "enable_vocal_tics", "Coughs, sneezes", "None", default=True
                    ),
                    lambda c: c.get("enable_vocal_tics", True),
                ),
                (
                    "non_directed",
                    "Non agent-directed speech",
                    lambda c: _get_enabled_desc(
                        c,
                        "enable_non_directed_phrases",
                        "``hold on'', ``one sec''",
                        "None",
                        default=True,
                    ),
                    lambda c: c.get("enable_non_directed_phrases", True),
                ),
                (
                    "interruptions",
                    "Interruptions",
                    lambda c: _get_enabled_desc(
                        c, "enable_interruptions", "LLM-based", "None"
                    ),
                    lambda c: c.get("enable_interruptions", False),
                ),
                (
                    "backchanneling",
                    "Backchanneling",
                    lambda c: _get_enabled_desc(
                        c, "use_llm_backchannel", "LLM-based", "None"
                    ),
                    lambda c: c.get("use_llm_backchannel", False),
                ),
            ],
        ),
    ]

    # =========================================================================
    # Table 1: Control vs Realistic with values (grouped by category)
    # =========================================================================
    main_conditions = ["control", "regular"]
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\caption{Speech complexity conditions: Clean vs Realistic.}")
    lines.append(r"\label{tab:conditions}")
    lines.append(r"\centering")
    lines.append(r"\begin{small}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{@{}llll@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Category} & \textbf{Setting} & \textbf{Clean} & \textbf{Realistic} \\"
    )
    lines.append(r"\midrule")

    for group_idx, (group_name, settings) in enumerate(setting_groups):
        n_rows = len(settings)
        for i, (setting_key, setting_label, extractor, _) in enumerate(settings):
            if i == 0:
                category_cell = rf"\multirow{{{n_rows}}}{{*}}{{{group_name}}}"
            else:
                category_cell = ""

            row_cells = [category_cell, setting_label]
            for condition in main_conditions:
                config = COMPLEXITY_CONFIGS.get(condition, {})
                value = extractor(config)
                row_cells.append(value)
            lines.append(" & ".join(row_cells) + r" \\")

        if group_idx < len(setting_groups) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{small}")
    lines.append(r"\end{table}")

    tex_path = output_dir / "conditions_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")

    # =========================================================================
    # Table 2: All conditions with checkmarks
    # =========================================================================
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(
        r"\caption{Speech complexity conditions by ablation. "
        r"Columns: Cln=Clean, +N=Noise, +A=Accents, +I=Interrupts, "
        r"+NA/NI/AI=pairwise combinations, Real=Realistic (all effects).}"
    )
    lines.append(r"\label{tab:conditions-ablation}")
    lines.append(r"\centering")
    lines.append(r"\begin{small}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")

    col_spec = "ll" + "c" * len(all_conditions)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header_cells = [r"\textbf{Category}", r"\textbf{Setting}"] + [
        rf"\textbf{{{condition_labels[c]}}}" for c in all_conditions
    ]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for group_idx, (group_name, settings) in enumerate(setting_groups):
        n_rows = len(settings)
        for i, (setting_key, setting_label, _, is_enabled) in enumerate(settings):
            if i == 0:
                category_cell = rf"\multirow{{{n_rows}}}{{*}}{{{group_name}}}"
            else:
                category_cell = ""

            row_cells = [category_cell, setting_label]
            for condition in all_conditions:
                config = COMPLEXITY_CONFIGS.get(condition, {})
                enabled = is_enabled(config)
                row_cells.append(r"\checkmark" if enabled else "")
            lines.append(" & ".join(row_cells) + r" \\")

        if group_idx < len(setting_groups) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{small}")
    lines.append(r"\end{table}")

    tex_path = output_dir / "conditions_ablation_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")

    # =========================================================================
    # Table 3: Single-factor ablations only (no pairwise)
    # =========================================================================
    single_conditions = [
        "control",
        "control_audio",
        "control_accents",
        "control_behavior",
        "regular",
    ]
    single_condition_labels = {
        "control": "Cln",
        "control_audio": "+N",
        "control_accents": "+A",
        "control_behavior": "+I",
        "regular": "Real",
    }

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(
        r"\caption{Speech complexity conditions by ablation (single factors). "
        r"Columns: Cln=Clean, +N=Noise, +A=Accents, +I=Interrupts, Real=Realistic (all effects).}"
    )
    lines.append(r"\label{tab:conditions-ablation-single}")
    lines.append(r"\centering")
    lines.append(r"\begin{small}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")

    col_spec = "ll" + "c" * len(single_conditions)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header_cells = [r"\textbf{Category}", r"\textbf{Setting}"] + [
        rf"\textbf{{{single_condition_labels[c]}}}" for c in single_conditions
    ]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for group_idx, (group_name, settings) in enumerate(setting_groups):
        n_rows = len(settings)
        for i, (setting_key, setting_label, _, is_enabled) in enumerate(settings):
            if i == 0:
                category_cell = rf"\multirow{{{n_rows}}}{{*}}{{{group_name}}}"
            else:
                category_cell = ""

            row_cells = [category_cell, setting_label]
            for condition in single_conditions:
                config = COMPLEXITY_CONFIGS.get(condition, {})
                enabled = is_enabled(config)
                row_cells.append(r"\checkmark" if enabled else "")
            lines.append(" & ".join(row_cells) + r" \\")

        if group_idx < len(setting_groups) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{small}")
    lines.append(r"\end{table}")

    tex_path = output_dir / "conditions_ablation_table_single.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")

    # =========================================================================
    # CSV (combined with values)
    # =========================================================================
    csv_rows = []
    for group_name, settings in setting_groups:
        for setting_key, setting_label, extractor, is_enabled in settings:
            row = {"category": group_name, "setting": setting_label}
            for condition in all_conditions:
                config = COMPLEXITY_CONFIGS.get(condition, {})
                value = extractor(config)
                value = (
                    value.replace("$\\mu$-law", "μ-law")
                    .replace("``", '"')
                    .replace("''", '"')
                )
                value = value.replace("$\\sim$", "~").replace("\\%", "%")
                row[condition] = value
                row[f"{condition}_enabled"] = is_enabled(config)
            csv_rows.append(row)

    df_csv = pd.DataFrame(csv_rows)
    csv_path = output_dir / "conditions_table.csv"
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")


def _get_control_personas():
    """Get control persona names for comparison."""
    from tau2.data_model.voice_personas import CONTROL_PERSONA_NAMES

    return CONTROL_PERSONA_NAMES


def _get_persona_desc(config: dict) -> str:
    """Get persona description from config."""
    from tau2.data_model.voice_personas import (
        CONTROL_PERSONA_NAMES,
        REGULAR_PERSONA_NAMES,
    )

    persona_names = config.get("persona_names", CONTROL_PERSONA_NAMES)
    if persona_names == CONTROL_PERSONA_NAMES:
        return "American"
    elif persona_names == REGULAR_PERSONA_NAMES:
        return "Diverse accents"
    else:
        return "Mixed"


def _get_enabled_desc(
    config: dict, key: str, enabled_text: str, disabled_text: str, default: bool = False
) -> str:
    """Get description based on boolean config value."""
    return enabled_text if config.get(key, default) else disabled_text


def _get_burst_desc(config: dict) -> str:
    """Get burst noise description."""
    if not config.get("enable_burst_noise", False):
        return "None"
    rate = config.get("burst_noise_events_per_minute", 0)
    if rate > 0:
        return f"$\\sim${rate:.0f}/min"
    return "None"


def _get_frame_drop_desc(config: dict) -> str:
    """Get frame drop description."""
    rate = config.get("frame_drop_rate", 0)
    if rate > 0:
        pct = rate * 100
        return f"$\\sim${pct:.1f}\\% (G-E model)"
    return "None"


def _generate_voice_vs_text_table(output_dir: Path, df_metrics: pd.DataFrame) -> None:
    """Generate LaTeX table comparing voice Pass^1 vs text baseline with deltas.

    Shows voice results (regular condition) alongside text-based results
    to highlight the gap between text and voice modalities.
    """
    df_metrics = df_metrics.copy()

    # Filter to regular complexity (the main voice condition)
    regular_data = df_metrics[df_metrics["speech_complexity"] == "regular"]
    if regular_data.empty:
        logger.warning("No regular complexity data for voice vs text table.")
        return

    # Get unique values
    domains = [d for d in DOMAINS if d in df_metrics["domain"].unique()]

    # Load text baseline (hardcoded reference scores)
    from experiments.tau_voice.exp.text_baselines import TEXT_SOTA

    text_model_name = TEXT_SOTA.model_name
    text_scores = TEXT_SOTA.get_scores_dict(domains, capitalize=False)
    logger.info(f"Text baseline for voice vs text table: {text_model_name}")

    # Aggregate voice data by domain and llm
    agg_df = regular_data.groupby(["domain", "llm"])["pass_hat_1"].mean().reset_index()

    # Build rows by model
    rows = []
    for domain in domains:
        domain_data = agg_df[agg_df["domain"] == domain]
        n_tasks = DOMAIN_TASK_COUNTS.get(domain, "?")
        text_val = text_scores.get(domain)

        for llm in domain_data["llm"].unique():
            model_data = domain_data[domain_data["llm"] == llm]
            voice_val = (
                model_data["pass_hat_1"].values[0] if len(model_data) > 0 else None
            )

            if voice_val is None:
                continue

            rows.append(
                {
                    "domain": domain,
                    "n_tasks": n_tasks,
                    "llm": llm,
                    "model": get_short_llm_name(llm),
                    "provider": get_provider_display(get_provider_key(llm)),
                    "text": text_val,
                    "voice": voice_val,
                }
            )

    # Build "All" aggregate rows (average across domains per model)
    all_rows = []
    total_tasks = sum(DOMAIN_TASK_COUNTS.get(d, 0) for d in domains)
    text_all = text_scores.get("all")

    models_in_data = sorted(
        set(r["llm"] for r in rows),
        key=lambda m: get_model_sort_key(m),
    )
    for llm in models_in_data:
        model_rows = [r for r in rows if r["llm"] == llm]
        voice_vals = [r["voice"] for r in model_rows if r["voice"] is not None]
        voice_avg = sum(voice_vals) / len(voice_vals) if voice_vals else None

        if voice_avg is not None:
            all_rows.append(
                {
                    "domain": "all",
                    "n_tasks": total_tasks,
                    "llm": llm,
                    "model": get_short_llm_name(llm),
                    "provider": get_provider_display(get_provider_key(llm)),
                    "text": text_all,
                    "voice": voice_avg,
                }
            )

    # Generate LaTeX
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(
        rf"\caption{{Voice vs Text comparison (pass@1). Text baseline: {text_model_name or 'GPT-4.1'}. Voice uses Realistic condition. $\Delta$ = Voice $-$ Text.}}"
    )
    lines.append(r"\label{tab:voice-vs-text}")
    lines.append(r"\centering")
    lines.append(r"\begin{small}")
    lines.append(r"\begin{tabular}{@{}llcccc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Domain} & \textbf{Model} & \textbf{Text} & \textbf{Voice} & \textbf{$\Delta$} & \textbf{$\Delta_{\%}$} \\"
    )
    lines.append(r"\midrule")

    # Helper function to format values
    def fmt_val(val, is_best=False):
        if val is None:
            return "--"
        pct = f"{round(val * 100)}\\%"
        return rf"\textbf{{{pct}}}" if is_best else pct

    def fmt_delta(text_val, voice_val):
        if text_val is None or voice_val is None:
            return "--", "--"
        delta = round((voice_val - text_val) * 100)
        delta_str = f"+{delta}\\%" if delta >= 0 else f"$-${abs(delta)}\\%"
        if text_val != 0:
            delta_rel = (voice_val - text_val) / text_val * 100
            delta_rel_str = (
                f"+{delta_rel:.1f}\\%"
                if delta_rel >= 0
                else f"$-${abs(delta_rel):.1f}\\%"
            )
        else:
            delta_rel_str = "--"
        return delta_str, delta_rel_str

    # Add "All" rows first
    if all_rows:
        all_rows_sorted = sorted(
            all_rows,
            key=lambda r: get_model_sort_key(r["llm"]),
        )
        voice_vals = [r["voice"] for r in all_rows_sorted if r["voice"] is not None]
        best_voice = max(voice_vals) if voice_vals else None
        n_models = len(all_rows_sorted)

        for i, row in enumerate(all_rows_sorted):
            if i == 0:
                domain_label = rf"\multirow{{{n_models}}}{{*}}{{All}}"
                text_str = rf"\multirow{{{n_models}}}{{*}}{{{fmt_val(row['text'])}}}"
            else:
                domain_label = ""
                text_str = ""

            voice_str = fmt_val(
                row["voice"],
                row["voice"] == best_voice and row["voice"] is not None,
            )
            delta_str, delta_rel_str = fmt_delta(row["text"], row["voice"])

            lines.append(
                f"{domain_label} & {row['model']} & {text_str} & {voice_str} & {delta_str} & {delta_rel_str} \\\\"
            )

        lines.append(r"\midrule")

    # Add domain-specific rows
    for domain in domains:
        domain_rows = [r for r in rows if r["domain"] == domain]
        domain_rows = sorted(
            domain_rows,
            key=lambda r: get_model_sort_key(r["llm"]),
        )
        n_models = len(domain_rows)

        voice_vals = [r["voice"] for r in domain_rows if r["voice"] is not None]
        best_voice = max(voice_vals) if voice_vals else None

        for i, row in enumerate(domain_rows):
            if i == 0:
                domain_label = rf"\multirow{{{n_models}}}{{*}}{{{domain.capitalize()}}}"
                text_str = rf"\multirow{{{n_models}}}{{*}}{{{fmt_val(row['text'])}}}"
            else:
                domain_label = ""
                text_str = ""

            voice_str = fmt_val(
                row["voice"],
                row["voice"] == best_voice and row["voice"] is not None,
            )
            delta_str, delta_rel_str = fmt_delta(row["text"], row["voice"])

            lines.append(
                f"{domain_label} & {row['model']} & {text_str} & {voice_str} & {delta_str} & {delta_rel_str} \\\\"
            )

        if domain != domains[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{small}")
    lines.append(r"\end{table}")

    tex_path = output_dir / "voice_vs_text_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")

    # Also save as CSV
    csv_rows = []
    for row in all_rows + rows:
        delta = None
        delta_rel = None
        if row["text"] is not None and row["voice"] is not None:
            delta = row["voice"] - row["text"]
            if row["text"] != 0:
                delta_rel = (row["voice"] - row["text"]) / row["text"]
        csv_rows.append(
            {
                "domain": "All" if row["domain"] == "all" else row["domain"],
                "n_tasks": row["n_tasks"],
                "provider": row["provider"],
                "model": row["model"],
                "text_sota": row["text"],
                "voice": row["voice"],
                "delta": delta,
                "delta_relative": delta_rel,
            }
        )
    df_csv = pd.DataFrame(csv_rows)
    csv_path = output_dir / "voice_vs_text_table.csv"
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")


def _generate_combined_comparison_table(
    output_dir: Path, df_metrics: pd.DataFrame
) -> None:
    """Generate LaTeX table comparing Text (reasoning), Text (non-reasoning), Control, and Realistic.

    Format: Text (GPT-5, reasoning) | Text (GPT-4.1) | Clean x% (-y) | Realistic x% (-y)
    where (-y) is the difference from the non-thinking text baseline.
    """
    df_metrics = df_metrics.copy()

    # Get unique values
    domains = [d for d in DOMAINS if d in df_metrics["domain"].unique()]

    # Load both text baselines (hardcoded reference scores)
    from experiments.tau_voice.exp.text_baselines import (
        TEXT_SOTA,
        TEXT_SOTA_NONTHINKING,
    )

    text_sota_name = TEXT_SOTA.model_name
    text_sota_scores = TEXT_SOTA.get_scores_dict(domains, capitalize=False)
    logger.info(f"Text reasoning for combined table: {text_sota_name}")

    text_nonthinking_name = TEXT_SOTA_NONTHINKING.model_name
    text_nonthinking_scores = TEXT_SOTA_NONTHINKING.get_scores_dict(
        domains, capitalize=False
    )
    logger.info(f"Text non-thinking for combined table: {text_nonthinking_name}")

    # Aggregate voice data by domain, llm, and complexity
    agg_df = (
        df_metrics.groupby(["domain", "llm", "speech_complexity"])["pass_hat_1"]
        .mean()
        .reset_index()
    )

    # Build rows by model (llm)
    rows = []
    for domain in domains:
        domain_data = agg_df[agg_df["domain"] == domain]
        text_sota_val = text_sota_scores.get(domain)
        text_nonthinking_val = text_nonthinking_scores.get(domain)

        for llm in domain_data["llm"].unique():
            llm_data = domain_data[domain_data["llm"] == llm]

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

            rows.append(
                {
                    "domain": domain,
                    "llm": llm,
                    "model": get_short_llm_name(llm),
                    "provider": get_provider_display(get_provider_key(llm)),
                    "text_sota": text_sota_val,
                    "text_nonthinking": text_nonthinking_val,
                    "control": control_val,
                    "regular": regular_val,
                }
            )

    # Build "All" aggregate rows (average across domains per model)
    all_rows = []
    text_sota_all = text_sota_scores.get("all")
    text_nonthinking_all = text_nonthinking_scores.get("all")

    models_in_data = sorted(
        set(r["llm"] for r in rows),
        key=get_model_sort_key,
    )
    for llm in models_in_data:
        model_rows = [r for r in rows if r["llm"] == llm]
        control_vals = [r["control"] for r in model_rows if r["control"] is not None]
        regular_vals = [r["regular"] for r in model_rows if r["regular"] is not None]

        control_avg = sum(control_vals) / len(control_vals) if control_vals else None
        regular_avg = sum(regular_vals) / len(regular_vals) if regular_vals else None

        if control_avg is not None or regular_avg is not None:
            all_rows.append(
                {
                    "domain": "all",
                    "llm": llm,
                    "model": get_short_llm_name(llm),
                    "provider": get_provider_display(get_provider_key(llm)),
                    "text_sota": text_sota_all,
                    "text_nonthinking": text_nonthinking_all,
                    "control": control_avg,
                    "regular": regular_avg,
                }
            )

    # Generate LaTeX (tabular content only - table wrapper added in paper)
    lines = []
    lines.append(r"\begin{tabular}{@{}llccc@{}}")
    lines.append(r"\toprule")
    lines.append(r" &  &  & \multicolumn{2}{c}{\textbf{Voice}} \\")
    lines.append(r"\cmidrule(l){4-5}")
    lines.append(
        r"\textbf{Domain} & \textbf{Model} & \textbf{Text} & \textbf{Clean} & \textbf{Realistic} \\"
    )
    lines.append(r"\midrule")

    def fmt_val_with_delta(val, text_val, is_best=False):
        if val is None:
            return "--"
        pct = round(val * 100)
        if text_val is not None:
            delta = round((val - text_val) * 100)
            if text_val != 0:
                delta_rel = (val - text_val) / text_val * 100
                cell = f"{pct}\\% ({delta:+d}, {delta_rel:+.1f}\\%)"
            else:
                cell = f"{pct}\\% ({delta:+d})"
        else:
            cell = f"{pct}\\%"
        return rf"\textbf{{{cell}}}" if is_best else cell

    def fmt_text_stacked(sota_val, nonthinking_val):
        """Format text column with GPT-5 and GPT-4.1 in parenthesis on same line."""
        if sota_val is None and nonthinking_val is None:
            return "--"
        sota_str = f"{round(sota_val * 100)}\\%" if sota_val is not None else "--"
        nonthinking_str = (
            f"({round(nonthinking_val * 100)}\\%)"
            if nonthinking_val is not None
            else ""
        )
        return f"{sota_str} {nonthinking_str}".strip()

    # Add "All" rows first
    if all_rows:
        all_rows_sorted = sorted(all_rows, key=lambda r: get_model_sort_key(r["llm"]))
        control_vals = [
            r["control"] for r in all_rows_sorted if r["control"] is not None
        ]
        regular_vals = [
            r["regular"] for r in all_rows_sorted if r["regular"] is not None
        ]
        best_control = max(control_vals) if control_vals else None
        best_regular = max(regular_vals) if regular_vals else None
        n_providers = len(all_rows_sorted)

        for i, row in enumerate(all_rows_sorted):
            if i == 0:
                domain_label = rf"\multirow{{{n_providers}}}{{*}}{{All}}"
                text_str = rf"\multirow{{{n_providers}}}{{*}}{{{fmt_text_stacked(row['text_sota'], row['text_nonthinking'])}}}"
            else:
                domain_label = ""
                text_str = ""

            # Compute deltas relative to GPT-5 text baseline
            control_str = fmt_val_with_delta(
                row["control"],
                row["text_sota"],
                row["control"] == best_control and row["control"] is not None,
            )
            regular_str = fmt_val_with_delta(
                row["regular"],
                row["text_sota"],
                row["regular"] == best_regular and row["regular"] is not None,
            )

            lines.append(
                f"{domain_label} & {row['model']} & {text_str} & {control_str} & {regular_str} \\\\"
            )

        lines.append(r"\midrule")

    # Add domain-specific rows
    for domain in domains:
        domain_rows = [r for r in rows if r["domain"] == domain]
        domain_rows = sorted(domain_rows, key=lambda r: get_model_sort_key(r["llm"]))
        n_providers = len(domain_rows)

        control_vals = [r["control"] for r in domain_rows if r["control"] is not None]
        regular_vals = [r["regular"] for r in domain_rows if r["regular"] is not None]
        best_control = max(control_vals) if control_vals else None
        best_regular = max(regular_vals) if regular_vals else None

        for i, row in enumerate(domain_rows):
            if i == 0:
                domain_label = (
                    rf"\multirow{{{n_providers}}}{{*}}{{{domain.capitalize()}}}"
                )
                text_str = rf"\multirow{{{n_providers}}}{{*}}{{{fmt_text_stacked(row['text_sota'], row['text_nonthinking'])}}}"
            else:
                domain_label = ""
                text_str = ""

            # Compute deltas relative to GPT-5 text baseline
            control_str = fmt_val_with_delta(
                row["control"],
                row["text_sota"],
                row["control"] == best_control and row["control"] is not None,
            )
            regular_str = fmt_val_with_delta(
                row["regular"],
                row["text_sota"],
                row["regular"] == best_regular and row["regular"] is not None,
            )

            lines.append(
                f"{domain_label} & {row['model']} & {text_str} & {control_str} & {regular_str} \\\\"
            )

        if domain != domains[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(
        r"\multicolumn{5}{l}{\footnotesize \textit{Text column: GPT-5, reasoning (GPT-4.1, best non-reasoning model). Deltas relative to GPT-5.}} \\"
    )
    lines.append(r"\end{tabular}")

    tex_path = output_dir / "combined_comparison_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")

    # Also save as CSV
    def _csv_row(row, domain_label):
        ctrl = row["control"]
        reg = row["regular"]
        sota = row["text_sota"]
        ctrl_delta = ctrl - sota if ctrl is not None and sota is not None else None
        reg_delta = reg - sota if reg is not None and sota is not None else None
        ctrl_delta_rel = (
            (ctrl - sota) / sota
            if ctrl is not None and sota is not None and sota != 0
            else None
        )
        reg_delta_rel = (
            (reg - sota) / sota
            if reg is not None and sota is not None and sota != 0
            else None
        )
        return {
            "domain": domain_label,
            "provider": row["provider"],
            "model": row["model"],
            "text_sota": sota,
            "text_nonthinking": row["text_nonthinking"],
            "control": ctrl,
            "control_delta": ctrl_delta,
            "control_delta_rel": ctrl_delta_rel,
            "regular": reg,
            "regular_delta": reg_delta,
            "regular_delta_rel": reg_delta_rel,
        }

    csv_rows = []
    for row in all_rows:
        csv_rows.append(_csv_row(row, "All"))
    for row in rows:
        csv_rows.append(_csv_row(row, row["domain"]))
    df_csv = pd.DataFrame(csv_rows)
    csv_path = output_dir / "combined_comparison_table.csv"
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")


# =============================================================================
# Performance Figures
# =============================================================================


def _plot_pass_1_headline(output_dir: Path, df_metrics: pd.DataFrame) -> None:
    """Generate headline Pass^1 figure with text baseline."""
    import matplotlib.pyplot as plt
    import numpy as np

    from experiments.tau_voice.exp.plot_style import DOMAIN_COLORS

    if "pass_hat_1" not in df_metrics.columns:
        logger.warning("pass_hat_1 not found. Skipping headline plot.")
        return

    # Filter to regular complexity
    regular_data = df_metrics[df_metrics["speech_complexity"] == "regular"]
    if regular_data.empty:
        logger.warning("No regular complexity data. Skipping headline plot.")
        return

    llms = sorted(regular_data["llm"].unique())
    tested_domains = regular_data["domain"].unique()
    domains_to_plot = [d for d in DOMAINS if d in tested_domains]

    # Compute scores
    llm_scores = {}
    for llm in llms:
        llm_data = regular_data[regular_data["llm"] == llm]
        scores = {"All": llm_data["pass_hat_1"].mean()}
        for domain in domains_to_plot:
            domain_data = llm_data[llm_data["domain"] == domain]
            if len(domain_data) > 0:
                scores[domain.capitalize()] = domain_data["pass_hat_1"].mean()
            else:
                scores[domain.capitalize()] = np.nan
        llm_scores[llm] = scores

    # Build models to plot
    categories = ["All"] + [d.capitalize() for d in domains_to_plot]
    n_categories = len(categories)

    # Load text baselines (hardcoded reference scores)
    from experiments.tau_voice.exp.text_baselines import DEFAULT_TEXT_BASELINES

    text_baselines_data = []
    for baseline in DEFAULT_TEXT_BASELINES:
        scores = baseline.get_scores_dict(domains_to_plot)
        text_baselines_data.append((baseline.display_name, scores))
        logger.info(f"Text baseline: {baseline.model_name}")

    models_to_plot = []
    n_text_baselines = len(text_baselines_data)
    for name, scores in text_baselines_data:
        models_to_plot.append((f"Text\n({name})", scores))

    for llm in llms:
        llm_short = llm.split(":")[-1] if ":" in llm else llm
        llm_short = llm_short.replace("gpt-realtime-", "gpt-rt-")
        llm_short = llm_short.replace("gemini-2.0-flash-live-", "gem-2-live-")
        llm_short = llm_short.replace("grok-2-realtime-", "grok-2-rt-")
        if len(llm_short) > 15:
            llm_short = llm_short[:15]
        models_to_plot.append((llm_short, llm_scores[llm]))

    n_models = len(models_to_plot)

    # Create figure
    fig_width = max(6, 1.2 * n_models)
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    group_width = 0.75
    bar_width = group_width / n_categories
    model_positions = np.arange(n_models) * 0.85

    # Category colors
    category_colors = {"All": "#778899"}
    for domain in domains_to_plot:
        category_colors[domain.capitalize()] = DOMAIN_COLORS.get(domain, "#888888")

    for m_idx, (model_name, scores) in enumerate(models_to_plot):
        for c_idx, cat in enumerate(categories):
            val = scores.get(cat, np.nan)
            if np.isnan(val):
                continue

            bar_offset = (c_idx - (n_categories - 1) / 2) * bar_width
            x = model_positions[m_idx] + bar_offset
            bar_color = category_colors.get(cat, "#888888")

            ax.bar(
                x,
                val,
                bar_width * 0.85,
                color=bar_color,
                edgecolor="white",
                linewidth=0.5,
            )
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

    ax.set_ylabel("pass@1", fontsize=11, fontweight="medium")
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)

    # Legend
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(
            facecolor=category_colors.get(cat, "#888888"), edgecolor="white", label=cat
        )
        for cat in categories
    ]
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


def _plot_pass_1_headline_simple(output_dir: Path, df_metrics: pd.DataFrame) -> None:
    """
    Generate simplified headline figure showing only 'All' aggregate
    with Text on the left and Voice (Control/Realistic) per model on the right.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if "pass_hat_1" not in df_metrics.columns:
        logger.warning("pass_hat_1 not found. Skipping simple headline plot.")
        return

    # Get control and regular data
    control_data = df_metrics[df_metrics["speech_complexity"] == "control"]
    regular_data = df_metrics[df_metrics["speech_complexity"] == "regular"]

    if control_data.empty and regular_data.empty:
        logger.warning("No control/regular data. Skipping simple headline plot.")
        return

    # Get models from either dataset
    all_data = df_metrics[df_metrics["speech_complexity"].isin(["control", "regular"])]
    llms = sorted(all_data["llm"].unique())

    # Compute "All" scores for each model and condition (key by llm to avoid
    # overwriting when multiple models share a provider)
    model_scores = {}
    for llm in llms:
        control_llm = control_data[control_data["llm"] == llm]
        regular_llm = regular_data[regular_data["llm"] == llm]

        control_all = (
            control_llm["pass_hat_1"].mean() if not control_llm.empty else np.nan
        )
        regular_all = (
            regular_llm["pass_hat_1"].mean() if not regular_llm.empty else np.nan
        )

        model_scores[llm] = {
            "Clean": control_all,  # "Clean" is display name for control condition
            "Realistic": regular_all,
        }

    # Load both text baselines (hardcoded reference scores)
    from experiments.tau_voice.exp.text_baselines import (
        TEXT_SOTA,
        TEXT_SOTA_NONTHINKING,
    )

    text_sota_name = TEXT_SOTA.model_name
    text_sota_score = TEXT_SOTA.overall
    logger.info(f"Text reasoning: {text_sota_name} = {text_sota_score:.1%}")

    text_nonthinking_name = TEXT_SOTA_NONTHINKING.model_name
    text_nonthinking_score = TEXT_SOTA_NONTHINKING.overall
    logger.info(
        f"Text non-thinking: {text_nonthinking_name} = {text_nonthinking_score:.1%}"
    )

    # Sort models by get_model_sort_key
    llms_sorted = sorted(model_scores.keys(), key=get_model_sort_key)

    # Create figure
    n_models = len(llms_sorted)
    fig_width = max(6, 1.5 + 1.2 * n_models)  # Extra space for two text bars
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    # Colors (using "Clean" label for control condition in display)
    colors = {
        "TextSOTA": "#1e40af",  # Dark blue for reasoning model
        "TextNonThinking": "#3b82f6",  # Lighter blue for non-thinking
        "Clean": "#10b981",  # Green for clean/control
        "Realistic": "#f59e0b",  # Orange for realistic
    }

    bar_width = 0.3
    group_width = 0.7  # Width for each provider group (Control + Realistic)

    # Positions for text bars (two bars on the left)
    text_sota_x = 0
    text_nonthinking_x = 0.5

    # Plot Text reasoning bar (GPT-5)
    if text_sota_score is not None:
        sota_short = text_sota_name or "GPT-5"
        if sota_short and len(sota_short) > 12:
            sota_short = sota_short[:12]

        ax.bar(
            text_sota_x,
            text_sota_score,
            bar_width * 1.1,
            color=colors["TextSOTA"],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.text(
            text_sota_x,
            text_sota_score + 0.02,
            f"{text_sota_score:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="medium",
        )

    # Plot Text Non-Thinking bar (GPT-4.1)
    if text_nonthinking_score is not None:
        nonthinking_short = text_nonthinking_name or "GPT-4.1"
        if nonthinking_short and len(nonthinking_short) > 12:
            nonthinking_short = nonthinking_short[:12]

        ax.bar(
            text_nonthinking_x,
            text_nonthinking_score,
            bar_width * 1.1,
            color=colors["TextNonThinking"],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.text(
            text_nonthinking_x,
            text_nonthinking_score + 0.02,
            f"{text_nonthinking_score:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="medium",
        )

    # Positions for voice models start after a gap
    voice_start_x = 1.3
    model_positions = voice_start_x + np.arange(n_models) * (group_width + 0.3)

    # Add vertical separator between text and voice (midway between last text bar and first voice bar)
    sep_x = (text_nonthinking_x + voice_start_x) / 2
    ax.axvline(x=sep_x, color="#cccccc", linestyle="-", linewidth=1, zorder=0)

    # Plot voice bars (one per model)
    for p_idx, llm in enumerate(llms_sorted):
        scores = model_scores[llm]
        base_x = model_positions[p_idx]

        # Clean (control) bar
        clean_val = scores.get("Clean", np.nan)
        if not np.isnan(clean_val):
            x_clean = base_x - bar_width / 2 - 0.02
            ax.bar(
                x_clean,
                clean_val,
                bar_width * 0.9,
                color=colors["Clean"],
                edgecolor="white",
                linewidth=0.5,
            )
            ax.text(
                x_clean,
                clean_val + 0.02,
                f"{clean_val:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="medium",
            )

        # Realistic bar
        regular_val = scores.get("Realistic", np.nan)
        if not np.isnan(regular_val):
            x_reg = base_x + bar_width / 2 + 0.02
            ax.bar(
                x_reg,
                regular_val,
                bar_width * 0.9,
                color=colors["Realistic"],
                edgecolor="white",
                linewidth=0.5,
            )
            ax.text(
                x_reg,
                regular_val + 0.02,
                f"{regular_val:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="medium",
            )

    # X-axis labels - include both text models and voice model short names
    all_xticks = [text_sota_x, text_nonthinking_x] + list(model_positions)
    sota_label = (
        f"Text\n({sota_short})" if text_sota_score and sota_short else "Text\n(GPT-5)"
    )
    nonthinking_label = (
        f"Text\n({nonthinking_short})"
        if text_nonthinking_score and nonthinking_short
        else "Text\n(GPT-4.1)"
    )
    all_xlabels = [sota_label, nonthinking_label] + [
        get_short_llm_name(llm) for llm in llms_sorted
    ]
    ax.set_xticks(all_xticks)
    ax.set_xticklabels(all_xlabels, fontsize=9, fontweight="medium")

    # Styling
    ax.set_ylabel("pass@1", fontsize=11, fontweight="medium")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-0.4, model_positions[-1] + 0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)

    # Legend
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(
            facecolor=colors["TextSOTA"],
            edgecolor="white",
            label="Text (GPT-5, reasoning)",
        ),
        Patch(
            facecolor=colors["TextNonThinking"],
            edgecolor="white",
            label="Text (GPT-4.1)",
        ),
        Patch(facecolor=colors["Clean"], edgecolor="white", label="Voice (Clean)"),
        Patch(
            facecolor=colors["Realistic"], edgecolor="white", label="Voice (Realistic)"
        ),
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=8,
        frameon=False,
    )

    # Add subtitle noting GPT-5 (reasoning) and GPT-4.1 (non-reasoning)
    ax.set_title(
        "Task Completion (pass@1): Text vs Voice\n"
        "(GPT-5 reasoning, GPT-4.1 best non-reasoning)",
        fontsize=10,
        fontweight="medium",
        pad=10,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "pass_1_headline_simple.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    logger.info(f"Saved: {output_dir / 'pass_1_headline_simple.pdf'}")


def _plot_ablation_figure(output_dir: Path, df_metrics: pd.DataFrame) -> None:
    """Generate ablation figure for retail domain."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Filter to retail and realtime LLMs
    retail_data = df_metrics[df_metrics["domain"] == "retail"]
    all_llms = sorted(retail_data["llm"].unique())
    llms = [
        llm
        for llm in all_llms
        if any(p in llm.lower() for p in ["gemini", "gpt-realtime", "grok", "xai"])
    ]

    if not llms:
        logger.warning("No realtime LLMs found for ablation figure.")
        return

    tested_complexities = retail_data["speech_complexity"].unique()
    complexities_to_plot = [c for c in SPEECH_COMPLEXITIES if c in tested_complexities]

    n_llms = len(llms)
    n_complexities = len(complexities_to_plot)

    # Create figure
    fig_width = max(3.5, 1.2 * n_llms)
    fig, ax = plt.subplots(figsize=(fig_width, 3.5))

    x = np.arange(n_llms) * 0.85
    bar_width = 0.7 / n_complexities

    for c_idx, complexity in enumerate(complexities_to_plot):
        complexity_color = SPEECH_COMPLEXITY_COLORS.get(complexity, "#888888")
        values = []

        for llm in llms:
            subset = retail_data[
                (retail_data["llm"] == llm)
                & (retail_data["speech_complexity"] == complexity)
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
                label = f"{round(val * 100)}"
                ax.text(xp, val + 0.02, label, ha="center", va="bottom", fontsize=5)

    # X-axis labels
    llm_short_names = [get_short_llm_name(llm) for llm in llms]

    ax.set_ylabel("pass@1", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(llm_short_names, fontsize=8, rotation=35, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.4, x[-1] + 0.4 if len(x) > 0 else 0.5)
    ax.set_title("Retail - Ablation", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(
            facecolor=SPEECH_COMPLEXITY_COLORS.get(c, "#888888"),
            edgecolor="white",
            label=get_complexity_display_name(c),
        )
        for c in complexities_to_plot
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=6,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "ablation_retail.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved: {output_dir / 'ablation_retail.pdf'}")


# =============================================================================
# Voice Tables
# =============================================================================


def _generate_voice_quality_table(
    output_dir: Path,
    df_latency: pd.DataFrame,
    df_interruption: pd.DataFrame,
) -> None:
    """Generate LaTeX table for voice quality metrics."""
    import numpy as np

    # Filter to regular complexity only
    df_latency_reg = df_latency[df_latency["speech_complexity"] == "regular"].copy()
    df_int_reg = df_interruption[
        df_interruption["speech_complexity"] == "regular"
    ].copy()

    if df_latency_reg.empty and df_int_reg.empty:
        logger.warning("No regular complexity data for voice quality table.")
        return

    # Map provider names
    df_latency_reg["provider_display"] = df_latency_reg["provider"].apply(
        get_provider_display
    )
    df_int_reg["provider_display"] = df_int_reg["provider"].apply(get_provider_display)

    # Merge
    merge_cols = ["llm", "domain", "speech_complexity", "provider", "provider_display"]
    if not df_latency_reg.empty and not df_int_reg.empty:
        df = pd.merge(df_latency_reg, df_int_reg, on=merge_cols, how="outer")
    elif not df_latency_reg.empty:
        df = df_latency_reg
    else:
        df = df_int_reg

    # Use all domains, not just those with data
    domains = DOMAINS

    # Get all models (llm) that appear in any domain
    all_llms_in_data = set()
    if not df.empty:
        all_llms_in_data = set(df["llm"].unique())
    # Fall back to empty if no data - will skip "All" section
    llms_sorted = (
        sorted(
            list(all_llms_in_data),
            key=get_model_sort_key,
        )
        if all_llms_in_data
        else []
    )

    # Build "All" aggregate data by model (average across domains)
    all_agg_data = {}
    for llm in llms_sorted:
        model_data = df[df["llm"] == llm]
        if not model_data.empty:
            all_agg_data[llm] = {
                "response_rate": model_data["response_rate"].mean()
                if "response_rate" in model_data
                else np.nan,
                "latency_mean": model_data["latency_mean"].mean()
                if "latency_mean" in model_data
                else np.nan,
                "user_interrupts_yield_rate": model_data[
                    "user_interrupts_yield_rate"
                ].mean()
                if "user_interrupts_yield_rate" in model_data
                else np.nan,
                "user_interrupts_yield_time_mean": model_data[
                    "user_interrupts_yield_time_mean"
                ].mean()
                if "user_interrupts_yield_time_mean" in model_data
                else np.nan,
                "backchannel_correct_rate": model_data[
                    "backchannel_correct_rate"
                ].mean()
                if "backchannel_correct_rate" in model_data
                else np.nan,
                "vocal_tic_correct_rate": model_data["vocal_tic_correct_rate"].mean()
                if "vocal_tic_correct_rate" in model_data
                else np.nan,
                "vocal_tic_silent_correct_rate": model_data[
                    "vocal_tic_silent_correct_rate"
                ].mean()
                if "vocal_tic_silent_correct_rate" in model_data
                else np.nan,
                "non_directed_silent_correct_rate": model_data[
                    "non_directed_silent_correct_rate"
                ].mean()
                if "non_directed_silent_correct_rate" in model_data
                else np.nan,
            }

    # Generate LaTeX (tabular content only - table wrapper added in paper)
    lines = []
    lines.append(r"\begin{tabular}{llcccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Domain} & \textbf{Model} & \makecell{\textbf{Resp.}\\\textbf{Rate}$\uparrow$} & \makecell{\textbf{Resp.}\\\textbf{Latency (s)}$\downarrow$} & \makecell{\textbf{Yield}\\\textbf{Rate}$\uparrow$} & \makecell{\textbf{Yield}\\\textbf{Time (s)}$\downarrow$} & \makecell{\textbf{Backchannel}\\\textbf{Correct}$\uparrow$} & \makecell{\textbf{Vocal Tic}\\\textbf{Correct}$\uparrow$} & \makecell{\textbf{Tic Ignore}\\\textbf{(Silent)}$\uparrow$} & \makecell{\textbf{Non-Agent}\\\textbf{Ignore}$\uparrow$} \\"
    )
    lines.append(r"\midrule")

    def fmt_pct(val, best_val):
        if pd.isna(val):
            return "--"
        pct = f"{round(val * 100)}\\%"
        return rf"\textbf{{{pct}}}" if val == best_val else pct

    def fmt_sec(val, best_val):
        if pd.isna(val):
            return "--"
        sec = f"{val:.2f}"
        return rf"\textbf{{{sec}}}" if val == best_val else sec

    # Add "All" rows first
    if all_agg_data:
        all_llms = sorted(all_agg_data.keys(), key=get_model_sort_key)
        # Find best values for "All" (higher is better for rates, lower for latencies)
        valid_resp_rates = [
            d["response_rate"]
            for d in all_agg_data.values()
            if not pd.isna(d["response_rate"])
        ]
        valid_latencies = [
            d["latency_mean"]
            for d in all_agg_data.values()
            if not pd.isna(d["latency_mean"])
        ]
        valid_yield_rates = [
            d["user_interrupts_yield_rate"]
            for d in all_agg_data.values()
            if not pd.isna(d["user_interrupts_yield_rate"])
        ]
        valid_yield_times = [
            d["user_interrupts_yield_time_mean"]
            for d in all_agg_data.values()
            if not pd.isna(d["user_interrupts_yield_time_mean"])
        ]
        valid_backchannel = [
            d["backchannel_correct_rate"]
            for d in all_agg_data.values()
            if not pd.isna(d["backchannel_correct_rate"])
        ]
        valid_vocal_tic = [
            d["vocal_tic_correct_rate"]
            for d in all_agg_data.values()
            if not pd.isna(d["vocal_tic_correct_rate"])
        ]
        valid_tic_silent = [
            d["vocal_tic_silent_correct_rate"]
            for d in all_agg_data.values()
            if not pd.isna(d["vocal_tic_silent_correct_rate"])
        ]
        valid_non_dir = [
            d["non_directed_silent_correct_rate"]
            for d in all_agg_data.values()
            if not pd.isna(d["non_directed_silent_correct_rate"])
        ]

        best_resp_rate = max(valid_resp_rates) if valid_resp_rates else None
        best_resp_latency = min(valid_latencies) if valid_latencies else None
        best_yield_rate = max(valid_yield_rates) if valid_yield_rates else None
        best_yield_time = min(valid_yield_times) if valid_yield_times else None
        best_backchannel = max(valid_backchannel) if valid_backchannel else None
        best_vocal_tic = max(valid_vocal_tic) if valid_vocal_tic else None
        best_tic_silent = max(valid_tic_silent) if valid_tic_silent else None
        best_non_dir = max(valid_non_dir) if valid_non_dir else None

        for i, llm in enumerate(all_llms):
            agg = all_agg_data[llm]
            model_name = get_short_llm_name(llm, max_len=25)
            if i == 0:
                domain_label = rf"\multirow{{{len(all_llms)}}}{{*}}{{All}}"
            else:
                domain_label = ""

            lines.append(
                f"{domain_label} & {model_name} & {fmt_pct(agg['response_rate'], best_resp_rate)} & "
                f"{fmt_sec(agg['latency_mean'], best_resp_latency)} & {fmt_pct(agg['user_interrupts_yield_rate'], best_yield_rate)} & "
                f"{fmt_sec(agg['user_interrupts_yield_time_mean'], best_yield_time)} & {fmt_pct(agg['backchannel_correct_rate'], best_backchannel)} & "
                f"{fmt_pct(agg['vocal_tic_correct_rate'], best_vocal_tic)} & {fmt_pct(agg['vocal_tic_silent_correct_rate'], best_tic_silent)} & "
                f"{fmt_pct(agg['non_directed_silent_correct_rate'], best_non_dir)} \\\\"
            )

        lines.append(r"\midrule")

    # Add domain-specific rows
    for domain in domains:
        domain_data = df[df["domain"] == domain] if not df.empty else pd.DataFrame()

        # Determine models for this domain
        if not domain_data.empty:
            domain_llms = sorted(
                domain_data["llm"].unique().tolist(),
                key=get_model_sort_key,
            )
            best_resp_rate = (
                domain_data["response_rate"].max()
                if "response_rate" in domain_data
                else None
            )
            best_resp_latency = (
                domain_data["latency_mean"].min()
                if "latency_mean" in domain_data
                else None
            )
            best_yield_rate = (
                domain_data["user_interrupts_yield_rate"].max()
                if "user_interrupts_yield_rate" in domain_data
                else None
            )
            best_yield_time = (
                domain_data["user_interrupts_yield_time_mean"].min()
                if "user_interrupts_yield_time_mean" in domain_data
                else None
            )
            best_backchannel = (
                domain_data["backchannel_correct_rate"].max()
                if "backchannel_correct_rate" in domain_data
                else None
            )
            best_vocal_tic = (
                domain_data["vocal_tic_correct_rate"].max()
                if "vocal_tic_correct_rate" in domain_data
                else None
            )
            best_tic_silent = (
                domain_data["vocal_tic_silent_correct_rate"].max()
                if "vocal_tic_silent_correct_rate" in domain_data
                else None
            )
            best_non_dir = (
                domain_data["non_directed_silent_correct_rate"].max()
                if "non_directed_silent_correct_rate" in domain_data
                else None
            )
        else:
            # No data for this domain - use models from other domains
            domain_llms = llms_sorted
            best_resp_rate = None
            best_resp_latency = None
            best_yield_rate = None
            best_yield_time = None
            best_backchannel = None
            best_vocal_tic = None
            best_tic_silent = None
            best_non_dir = None

        for i, llm in enumerate(domain_llms):
            model_name = get_short_llm_name(llm, max_len=25)
            if not domain_data.empty:
                model_rows = domain_data[domain_data["llm"] == llm]
                if len(model_rows) > 0:
                    row = model_rows.iloc[0]
                    resp_rate = row.get("response_rate", np.nan)
                    resp_latency = row.get("latency_mean", np.nan)
                    yield_rate = row.get("user_interrupts_yield_rate", np.nan)
                    yield_time = row.get("user_interrupts_yield_time_mean", np.nan)
                    backchannel = row.get("backchannel_correct_rate", np.nan)
                    vocal_tic = row.get("vocal_tic_correct_rate", np.nan)
                    tic_silent = row.get("vocal_tic_silent_correct_rate", np.nan)
                    non_dir = row.get("non_directed_silent_correct_rate", np.nan)
                else:
                    resp_rate = resp_latency = yield_rate = yield_time = backchannel = (
                        np.nan
                    )
                    vocal_tic = tic_silent = non_dir = np.nan
            else:
                resp_rate = resp_latency = yield_rate = yield_time = backchannel = (
                    np.nan
                )
                vocal_tic = tic_silent = non_dir = np.nan

            if i == 0:
                domain_label = (
                    rf"\multirow{{{len(domain_llms)}}}{{*}}{{{domain.capitalize()}}}"
                )
            else:
                domain_label = ""

            lines.append(
                f"{domain_label} & {model_name} & {fmt_pct(resp_rate, best_resp_rate)} & "
                f"{fmt_sec(resp_latency, best_resp_latency)} & {fmt_pct(yield_rate, best_yield_rate)} & "
                f"{fmt_sec(yield_time, best_yield_time)} & {fmt_pct(backchannel, best_backchannel)} & "
                f"{fmt_pct(vocal_tic, best_vocal_tic)} & {fmt_pct(tic_silent, best_tic_silent)} & "
                f"{fmt_pct(non_dir, best_non_dir)} \\\\"
            )

        if domain != domains[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = output_dir / "voice_quality_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")

    # Also save as CSV (include "All" rows first, then domain rows)
    csv_rows = []

    # Add "All" aggregate rows first
    for llm, agg in all_agg_data.items():
        model_name = get_short_llm_name(llm, max_len=25)
        provider_display = get_provider_display(get_provider_key(llm))
        csv_rows.append(
            {
                "domain": "All",
                "provider": provider_display,
                "model": model_name,
                "response_rate": agg["response_rate"],
                "response_latency": agg["latency_mean"],
                "yield_rate": agg["user_interrupts_yield_rate"],
                "yield_time": agg["user_interrupts_yield_time_mean"],
                "backchannel_correct": agg["backchannel_correct_rate"],
                "vocal_tic_correct": agg["vocal_tic_correct_rate"],
                "vocal_tic_silent_correct": agg["vocal_tic_silent_correct_rate"],
                "non_directed_silent_correct": agg["non_directed_silent_correct_rate"],
            }
        )

    # Add domain-specific rows
    for domain in domains:
        domain_data = df[df["domain"] == domain] if not df.empty else pd.DataFrame()

        if not domain_data.empty:
            domain_llms = sorted(
                domain_data["llm"].unique().tolist(),
                key=get_model_sort_key,
            )
        else:
            domain_llms = llms_sorted

        for llm in domain_llms:
            model_name = get_short_llm_name(llm, max_len=25)
            provider_display = get_provider_display(get_provider_key(llm))
            if not domain_data.empty:
                model_rows = domain_data[domain_data["llm"] == llm]
                if len(model_rows) > 0:
                    row = model_rows.iloc[0]
                    csv_rows.append(
                        {
                            "domain": domain,
                            "provider": provider_display,
                            "model": model_name,
                            "response_rate": row.get("response_rate"),
                            "response_latency": row.get("latency_mean"),
                            "yield_rate": row.get("user_interrupts_yield_rate"),
                            "yield_time": row.get("user_interrupts_yield_time_mean"),
                            "backchannel_correct": row.get("backchannel_correct_rate"),
                            "vocal_tic_correct": row.get("vocal_tic_correct_rate"),
                            "vocal_tic_silent_correct": row.get(
                                "vocal_tic_silent_correct_rate"
                            ),
                            "non_directed_silent_correct": row.get(
                                "non_directed_silent_correct_rate"
                            ),
                        }
                    )
                else:
                    csv_rows.append(
                        {
                            "domain": domain,
                            "provider": provider_display,
                            "model": model_name,
                            "response_rate": None,
                            "response_latency": None,
                            "yield_rate": None,
                            "yield_time": None,
                            "backchannel_correct": None,
                            "vocal_tic_correct": None,
                            "vocal_tic_silent_correct": None,
                            "non_directed_silent_correct": None,
                        }
                    )
            else:
                csv_rows.append(
                    {
                        "domain": domain,
                        "provider": provider_display,
                        "model": model_name,
                        "response_rate": None,
                        "response_latency": None,
                        "yield_rate": None,
                        "yield_time": None,
                        "backchannel_correct": None,
                        "vocal_tic_correct": None,
                        "vocal_tic_silent_correct": None,
                        "non_directed_silent_correct": None,
                    }
                )

    df_csv = pd.DataFrame(csv_rows)
    csv_path = output_dir / "voice_quality_table.csv"
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")


def _generate_voice_quality_table_unified(
    output_dir: Path,
    df_voice: pd.DataFrame,
    df_interruption: pd.DataFrame = None,
) -> None:
    """
    Generate LaTeX table for voice quality metrics from unified voice_quality_analysis.csv.

    This uses the same event extraction as the speech timeline visualization,
    ensuring perfect consistency between visualizations and tables.

    Metrics:
    - Response Rate (higher is better)
    - Response Latency (lower is better)
    - Yield Rate (higher is better)
    - Yield Latency (lower is better)
    - Agent Interruption Rate (lower is better) - how often agent interrupts user
    - Backchannel Error Rate (lower is better)
    - Vocal Tic Error Rate (lower is better)
    - Non Agent-Directed Error Rate (lower is better)
    """
    import numpy as np

    # Filter to regular complexity only
    df = df_voice[df_voice["speech_complexity"] == "regular"].copy()

    if df.empty:
        logger.warning("No regular complexity data for voice quality table.")
        return

    # Add provider display column
    df["provider_key"] = df["llm"].apply(get_provider_key)
    df["provider_display"] = df["provider_key"].apply(get_provider_display)

    # Merge interruption data if available
    if df_interruption is not None:
        df_int = df_interruption[
            df_interruption["speech_complexity"] == "regular"
        ].copy()
        df_int["provider_key"] = df_int["llm"].apply(get_provider_key)
        df_int["provider_display"] = df_int["provider_key"].apply(get_provider_display)
        # Merge on llm, domain, speech_complexity
        df = df.merge(
            df_int[["llm", "domain", "speech_complexity", "agent_interrupts_count"]],
            on=["llm", "domain", "speech_complexity"],
            how="left",
        )
        # Compute agent interruption rate: agent_interrupts_count / response_total (user turns)
        df["agent_interruption_rate"] = (
            df["agent_interrupts_count"] / df["response_total"]
        )
    else:
        df["agent_interrupts_count"] = np.nan
        df["agent_interruption_rate"] = np.nan

    # Use all domains
    domains = DOMAINS

    # Get all models (llm) that appear in any domain
    all_llms_in_data = set(df["llm"].unique()) if not df.empty else set()
    llms_sorted = (
        sorted(list(all_llms_in_data), key=get_model_sort_key)
        if all_llms_in_data
        else []
    )

    # Build "All" aggregate data by model (average across domains)
    all_agg_data = {}
    for llm in llms_sorted:
        model_data = df[df["llm"] == llm]
        if not model_data.empty:
            all_agg_data[llm] = {
                "response_rate": model_data["response_rate"].mean(),
                "response_latency_mean": model_data["response_latency_mean"].mean(),
                "yield_rate": model_data["yield_rate"].mean(),
                "yield_latency_mean": model_data["yield_latency_mean"].mean(),
                "agent_interruption_rate": model_data["agent_interruption_rate"].mean(),
                "backchannel_error_rate": model_data["backchannel_error_rate"].mean(),
                "vocal_tic_error_rate": model_data["vocal_tic_error_rate"].mean(),
                "non_directed_error_rate": model_data["non_directed_error_rate"].mean(),
            }

    # Generate LaTeX (tabular content only - table wrapper added in paper)
    lines = []
    lines.append(r"\begin{tabular}{@{}ll|cc|cc|c|ccc@{}}")
    lines.append(r"\toprule")
    # Grouped header row
    lines.append(
        r" & & \multicolumn{2}{c|}{\textbf{Latency}$\downarrow$} & "
        r"\multicolumn{2}{c|}{\textbf{Responsiveness}$\uparrow$} & "
        r"\textbf{Interrupt}$\downarrow$ & "
        r"\multicolumn{3}{c}{\textbf{Selectivity}$\uparrow$} \\"
    )
    # Sub-header row with individual metrics
    lines.append(
        r"\textbf{Domain} & \textbf{Model} & "
        r"$L_R$ & $L_Y$ & $R_R$ & $R_Y$ & $I_A$ & $S_{BC}$ & $S_{VT}$ & $S_{ND}$ \\"
    )
    lines.append(r"\midrule")

    def fmt_pct_higher_better(val, best_val):
        """Format percentage where higher is better."""
        if pd.isna(val):
            return "--"
        pct = f"{round(val * 100)}\\%"
        return rf"\textbf{{{pct}}}" if val == best_val else pct

    def fmt_pct_lower_better(val, best_val):
        """Format percentage where lower is better (error rates)."""
        if pd.isna(val):
            return "--"
        pct = f"{round(val * 100)}\\%"
        return rf"\textbf{{{pct}}}" if val == best_val else pct

    def fmt_sec(val, best_val):
        """Format seconds where lower is better."""
        if pd.isna(val):
            return "--"
        sec = f"{val:.2f}s"
        return rf"\textbf{{{sec}}}" if val == best_val else sec

    # Add "All" rows first
    if all_agg_data:
        all_llms = sorted(all_agg_data.keys(), key=get_model_sort_key)
        # Find best values for "All" (higher is better for rates, lower for latencies/errors)
        best_resp_rate = max(
            d["response_rate"]
            for d in all_agg_data.values()
            if not pd.isna(d["response_rate"])
        )
        best_resp_latency = min(
            d["response_latency_mean"]
            for d in all_agg_data.values()
            if not pd.isna(d["response_latency_mean"])
        )
        best_yield_rate = max(
            d["yield_rate"]
            for d in all_agg_data.values()
            if not pd.isna(d["yield_rate"])
        )
        best_yield_latency = min(
            d["yield_latency_mean"]
            for d in all_agg_data.values()
            if not pd.isna(d["yield_latency_mean"])
        )
        # Best agent interruption rate (lower is better)
        agent_int_values = [
            d["agent_interruption_rate"]
            for d in all_agg_data.values()
            if not pd.isna(d["agent_interruption_rate"])
        ]
        best_agent_interruption = min(agent_int_values) if agent_int_values else np.nan
        # Compute best correct rates for selectivity (higher is better)
        best_backchannel_correct = max(
            1 - d["backchannel_error_rate"]
            for d in all_agg_data.values()
            if not pd.isna(d["backchannel_error_rate"])
        )
        best_vocal_tic_correct = max(
            1 - d["vocal_tic_error_rate"]
            for d in all_agg_data.values()
            if not pd.isna(d["vocal_tic_error_rate"])
        )
        best_non_dir_correct = max(
            1 - d["non_directed_error_rate"]
            for d in all_agg_data.values()
            if not pd.isna(d["non_directed_error_rate"])
        )

        for i, llm in enumerate(all_llms):
            agg = all_agg_data[llm]
            model_name = get_short_llm_name(llm, max_len=25)
            if i == 0:
                domain_label = rf"\multirow{{{len(all_llms)}}}{{*}}{{All}}"
            else:
                domain_label = ""

            # Compute correct rates for selectivity
            bc_correct = (
                1 - agg["backchannel_error_rate"]
                if not pd.isna(agg["backchannel_error_rate"])
                else np.nan
            )
            vt_correct = (
                1 - agg["vocal_tic_error_rate"]
                if not pd.isna(agg["vocal_tic_error_rate"])
                else np.nan
            )
            nd_correct = (
                1 - agg["non_directed_error_rate"]
                if not pd.isna(agg["non_directed_error_rate"])
                else np.nan
            )

            # Order: Latency (L_R, L_Y), Responsiveness (R_R, R_Y), Turn (I_A), Selectivity (S_BC, S_VT, S_ND)
            lines.append(
                f"{domain_label} & {model_name} & "
                f"{fmt_sec(agg['response_latency_mean'], best_resp_latency)} & "
                f"{fmt_sec(agg['yield_latency_mean'], best_yield_latency)} & "
                f"{fmt_pct_higher_better(agg['response_rate'], best_resp_rate)} & "
                f"{fmt_pct_higher_better(agg['yield_rate'], best_yield_rate)} & "
                f"{fmt_pct_lower_better(agg['agent_interruption_rate'], best_agent_interruption)} & "
                f"{fmt_pct_higher_better(bc_correct, best_backchannel_correct)} & "
                f"{fmt_pct_higher_better(vt_correct, best_vocal_tic_correct)} & "
                f"{fmt_pct_higher_better(nd_correct, best_non_dir_correct)} \\\\"
            )

        lines.append(r"\midrule")

    # Add domain-specific rows
    for domain in domains:
        domain_data = df[df["domain"] == domain] if not df.empty else pd.DataFrame()

        if not domain_data.empty:
            domain_llms = sorted(
                domain_data["llm"].unique().tolist(),
                key=get_model_sort_key,
            )
            # Find best values (lower is better for latencies, higher for rates and correct rates)
            best_resp_latency = domain_data["response_latency_mean"].min()
            best_yield_latency = domain_data["yield_latency_mean"].min()
            best_resp_rate = domain_data["response_rate"].max()
            best_yield_rate = domain_data["yield_rate"].max()
            # Best agent interruption rate (lower is better)
            best_agent_interruption = domain_data["agent_interruption_rate"].min()
            # Best correct rates for selectivity (higher is better)
            best_backchannel_correct = (1 - domain_data["backchannel_error_rate"]).max()
            best_vocal_tic_correct = (1 - domain_data["vocal_tic_error_rate"]).max()
            best_non_dir_correct = (1 - domain_data["non_directed_error_rate"]).max()
        else:
            domain_llms = llms_sorted
            best_resp_latency = None
            best_yield_latency = None
            best_resp_rate = None
            best_yield_rate = None
            best_agent_interruption = None
            best_backchannel_correct = None
            best_vocal_tic_correct = None
            best_non_dir_correct = None

        for i, llm in enumerate(domain_llms):
            model_name = get_short_llm_name(llm, max_len=25)
            if not domain_data.empty:
                model_rows = domain_data[domain_data["llm"] == llm]
                if len(model_rows) > 0:
                    row = model_rows.iloc[0]
                    resp_latency = row.get("response_latency_mean", np.nan)
                    yield_latency = row.get("yield_latency_mean", np.nan)
                    resp_rate = row.get("response_rate", np.nan)
                    yield_rate = row.get("yield_rate", np.nan)
                    agent_int_rate = row.get("agent_interruption_rate", np.nan)
                    backchannel_err = row.get("backchannel_error_rate", np.nan)
                    vocal_tic_err = row.get("vocal_tic_error_rate", np.nan)
                    non_dir_err = row.get("non_directed_error_rate", np.nan)
                else:
                    resp_latency = yield_latency = resp_rate = yield_rate = np.nan
                    agent_int_rate = np.nan
                    backchannel_err = vocal_tic_err = non_dir_err = np.nan
            else:
                resp_latency = yield_latency = resp_rate = yield_rate = np.nan
                agent_int_rate = np.nan
                backchannel_err = vocal_tic_err = non_dir_err = np.nan

            # Compute correct rates for selectivity
            bc_correct = 1 - backchannel_err if not pd.isna(backchannel_err) else np.nan
            vt_correct = 1 - vocal_tic_err if not pd.isna(vocal_tic_err) else np.nan
            nd_correct = 1 - non_dir_err if not pd.isna(non_dir_err) else np.nan

            if i == 0:
                domain_label = (
                    rf"\multirow{{{len(domain_llms)}}}{{*}}{{{domain.capitalize()}}}"
                )
            else:
                domain_label = ""

            # Order: Latency (L_R, L_Y), Responsiveness (R_R, R_Y), Turn (I_A), Selectivity (S_BC, S_VT, S_ND)
            lines.append(
                f"{domain_label} & {model_name} & "
                f"{fmt_sec(resp_latency, best_resp_latency)} & "
                f"{fmt_sec(yield_latency, best_yield_latency)} & "
                f"{fmt_pct_higher_better(resp_rate, best_resp_rate)} & "
                f"{fmt_pct_higher_better(yield_rate, best_yield_rate)} & "
                f"{fmt_pct_lower_better(agent_int_rate, best_agent_interruption)} & "
                f"{fmt_pct_higher_better(bc_correct, best_backchannel_correct)} & "
                f"{fmt_pct_higher_better(vt_correct, best_vocal_tic_correct)} & "
                f"{fmt_pct_higher_better(nd_correct, best_non_dir_correct)} \\\\"
            )

        if domain != domains[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = output_dir / "voice_quality_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")

    # Also save as CSV (include "All" rows first, then domain rows)
    csv_rows = []

    # Add "All" aggregate rows first
    for llm, agg in all_agg_data.items():
        model_name = get_short_llm_name(llm, max_len=25)
        provider_display = get_provider_display(get_provider_key(llm))
        csv_rows.append(
            {
                "domain": "All",
                "provider": provider_display,
                "model": model_name,
                "response_rate": agg["response_rate"],
                "response_latency": agg["response_latency_mean"],
                "yield_rate": agg["yield_rate"],
                "yield_latency": agg["yield_latency_mean"],
                "backchannel_error_rate": agg["backchannel_error_rate"],
                "vocal_tic_error_rate": agg["vocal_tic_error_rate"],
                "non_directed_error_rate": agg["non_directed_error_rate"],
            }
        )

    # Add domain-specific rows
    for domain in domains:
        domain_data = df[df["domain"] == domain] if not df.empty else pd.DataFrame()

        if not domain_data.empty:
            domain_llms = sorted(
                domain_data["llm"].unique().tolist(),
                key=get_model_sort_key,
            )
        else:
            domain_llms = llms_sorted

        for llm in domain_llms:
            model_name = get_short_llm_name(llm, max_len=25)
            provider_display = get_provider_display(get_provider_key(llm))
            if not domain_data.empty:
                model_rows = domain_data[domain_data["llm"] == llm]
                if len(model_rows) > 0:
                    row = model_rows.iloc[0]
                    csv_rows.append(
                        {
                            "domain": domain,
                            "provider": provider_display,
                            "model": model_name,
                            "response_rate": row.get("response_rate"),
                            "response_latency": row.get("response_latency_mean"),
                            "yield_rate": row.get("yield_rate"),
                            "yield_latency": row.get("yield_latency_mean"),
                            "backchannel_error_rate": row.get("backchannel_error_rate"),
                            "vocal_tic_error_rate": row.get("vocal_tic_error_rate"),
                            "non_directed_error_rate": row.get(
                                "non_directed_error_rate"
                            ),
                        }
                    )
                else:
                    csv_rows.append(
                        {
                            "domain": domain,
                            "provider": provider_display,
                            "model": model_name,
                            "response_rate": None,
                            "response_latency": None,
                            "yield_rate": None,
                            "yield_latency": None,
                            "backchannel_error_rate": None,
                            "vocal_tic_error_rate": None,
                            "non_directed_error_rate": None,
                        }
                    )
            else:
                csv_rows.append(
                    {
                        "domain": domain,
                        "provider": provider_display,
                        "model": model_name,
                        "response_rate": None,
                        "response_latency": None,
                        "yield_rate": None,
                        "yield_latency": None,
                        "backchannel_error_rate": None,
                        "vocal_tic_error_rate": None,
                        "non_directed_error_rate": None,
                    }
                )

    df_csv = pd.DataFrame(csv_rows)
    csv_path = output_dir / "voice_quality_table.csv"
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")


def _generate_voice_quality_aggregated_table(
    output_dir: Path,
    df_voice: pd.DataFrame,
    df_interruption: pd.DataFrame = None,
) -> None:
    """
    Generate compact LaTeX table with aggregated voice quality metrics.

    Reports four aggregate scores:
    - Latency = avg(Response Latency, Yield Latency)
    - Responsiveness = avg(Response Rate, Yield Rate)
    - Interrupt = Agent Interruption Rate (lower is better, can exceed 100%)
    - Selectivity = avg(Backchannel Correct, Vocal Tic Correct, Non-Directed Correct)
    """
    import numpy as np

    # Filter to regular complexity only
    df = df_voice[df_voice["speech_complexity"] == "regular"].copy()

    if df.empty:
        logger.warning("No regular complexity data for aggregated voice quality table.")
        return

    # Add provider display column
    df["provider_key"] = df["llm"].apply(get_provider_key)
    df["provider_display"] = df["provider_key"].apply(get_provider_display)

    # Merge interruption data if available
    if df_interruption is not None:
        df_int = df_interruption[
            df_interruption["speech_complexity"] == "regular"
        ].copy()
        df_int["provider_key"] = df_int["llm"].apply(get_provider_key)
        df_int["provider_display"] = df_int["provider_key"].apply(get_provider_display)
        # Merge on llm, domain, speech_complexity
        df = df.merge(
            df_int[["llm", "domain", "speech_complexity", "agent_interrupts_count"]],
            on=["llm", "domain", "speech_complexity"],
            how="left",
        )
        # Compute agent interruption rate: agent_interrupts_count / response_total (user turns)
        df["agent_interruption_rate"] = (
            df["agent_interrupts_count"] / df["response_total"]
        )
    else:
        df["agent_interrupts_count"] = np.nan
        df["agent_interruption_rate"] = np.nan

    # Use all domains
    domains = DOMAINS

    # Get all models (llm)
    all_llms_in_data = set(df["llm"].unique()) if not df.empty else set()
    llms_sorted = (
        sorted(list(all_llms_in_data), key=get_model_sort_key)
        if all_llms_in_data
        else []
    )

    def compute_aggregates(row_or_agg):
        """Compute the 4 aggregate metrics from individual metrics."""
        resp_latency = row_or_agg.get("response_latency_mean", np.nan)
        yield_latency = row_or_agg.get("yield_latency_mean", np.nan)
        resp_rate = row_or_agg.get("response_rate", np.nan)
        yield_rate = row_or_agg.get("yield_rate", np.nan)
        agent_int_rate = row_or_agg.get("agent_interruption_rate", np.nan)
        bc_err = row_or_agg.get("backchannel_error_rate", np.nan)
        vt_err = row_or_agg.get("vocal_tic_error_rate", np.nan)
        nd_err = row_or_agg.get("non_directed_error_rate", np.nan)

        # Latency = avg(L_R, L_Y)
        latency_vals = [v for v in [resp_latency, yield_latency] if not pd.isna(v)]
        latency = np.mean(latency_vals) if latency_vals else np.nan

        # Responsiveness = avg(R_R, R_Y)
        resp_vals = [v for v in [resp_rate, yield_rate] if not pd.isna(v)]
        responsiveness = np.mean(resp_vals) if resp_vals else np.nan

        # Interrupt = Agent Interruption Rate (lower is better, can exceed 100%)
        interrupt = agent_int_rate

        # Selectivity = avg(1 - BC_err, 1 - VT_err, 1 - ND_err)
        sel_vals = []
        for err in [bc_err, vt_err, nd_err]:
            if not pd.isna(err):
                sel_vals.append(1 - err)
        selectivity = np.mean(sel_vals) if sel_vals else np.nan

        return latency, responsiveness, interrupt, selectivity

    # Build "All" aggregate data by model
    all_agg_data = {}
    for llm in llms_sorted:
        model_data = df[df["llm"] == llm]
        if not model_data.empty:
            agg = {
                "response_latency_mean": model_data["response_latency_mean"].mean(),
                "yield_latency_mean": model_data["yield_latency_mean"].mean(),
                "response_rate": model_data["response_rate"].mean(),
                "yield_rate": model_data["yield_rate"].mean(),
                "agent_interruption_rate": model_data["agent_interruption_rate"].mean(),
                "backchannel_error_rate": model_data["backchannel_error_rate"].mean(),
                "vocal_tic_error_rate": model_data["vocal_tic_error_rate"].mean(),
                "non_directed_error_rate": model_data["non_directed_error_rate"].mean(),
            }
            latency, responsiveness, interrupt, selectivity = compute_aggregates(agg)
            all_agg_data[llm] = {
                "latency": latency,
                "responsiveness": responsiveness,
                "interrupt": interrupt,
                "selectivity": selectivity,
            }

    # Generate LaTeX - simplified table with only "All" domain (no Domain column)
    lines = []
    lines.append(r"\begin{tabular}{@{}lcccc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & "
        r"\textbf{Latency}$\downarrow$ & \textbf{Responsiveness}$\uparrow$ & "
        r"\textbf{Interrupt}$\downarrow$ & \textbf{Selectivity}$\uparrow$ \\"
    )
    lines.append(r"\midrule")

    def fmt_sec(val, best_val):
        if pd.isna(val):
            return "--"
        sec = f"{val:.2f}s"
        return rf"\textbf{{{sec}}}" if abs(val - best_val) < 0.001 else sec

    def fmt_pct(val, best_val):
        if pd.isna(val):
            return "--"
        pct = f"{round(val * 100)}\\%"
        return rf"\textbf{{{pct}}}" if abs(val - best_val) < 0.001 else pct

    # Only show "All" rows (aggregated across domains)
    if all_agg_data:
        all_llms = sorted(all_agg_data.keys(), key=get_model_sort_key)
        best_latency = min(
            d["latency"] for d in all_agg_data.values() if not pd.isna(d["latency"])
        )
        best_resp = max(
            d["responsiveness"]
            for d in all_agg_data.values()
            if not pd.isna(d["responsiveness"])
        )
        interrupt_vals = [
            d["interrupt"] for d in all_agg_data.values() if not pd.isna(d["interrupt"])
        ]
        best_interrupt = (
            min(interrupt_vals) if interrupt_vals else np.nan
        )  # lower is better
        best_sel = max(
            d["selectivity"]
            for d in all_agg_data.values()
            if not pd.isna(d["selectivity"])
        )

        for llm in all_llms:
            agg = all_agg_data[llm]
            model_name = get_short_llm_name(llm, max_len=25)
            lines.append(
                f"{model_name} & "
                f"{fmt_sec(agg['latency'], best_latency)} & "
                f"{fmt_pct(agg['responsiveness'], best_resp)} & "
                f"{fmt_pct(agg['interrupt'], best_interrupt)} & "
                f"{fmt_pct(agg['selectivity'], best_sel)} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = output_dir / "voice_quality_aggregated_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")

    # Also save as CSV
    csv_rows = []
    for llm, agg in all_agg_data.items():
        model_name = get_short_llm_name(llm, max_len=25)
        provider_display = get_provider_display(get_provider_key(llm))
        csv_rows.append(
            {
                "domain": "All",
                "provider": provider_display,
                "model": model_name,
                "latency": agg["latency"],
                "responsiveness": agg["responsiveness"],
                "interrupt": agg["interrupt"],
                "selectivity": agg["selectivity"],
            }
        )
    for domain in domains:
        domain_data = df[df["domain"] == domain] if not df.empty else pd.DataFrame()
        if not domain_data.empty:
            for llm in domain_data["llm"].unique():
                model_rows = domain_data[domain_data["llm"] == llm]
                if len(model_rows) > 0:
                    row = model_rows.iloc[0]
                    latency, responsiveness, interrupt, selectivity = (
                        compute_aggregates(row)
                    )
                    model_name = get_short_llm_name(llm, max_len=25)
                    provider_display = get_provider_display(get_provider_key(llm))
                    csv_rows.append(
                        {
                            "domain": domain,
                            "provider": provider_display,
                            "model": model_name,
                            "latency": latency,
                            "responsiveness": responsiveness,
                            "interrupt": interrupt,
                            "selectivity": selectivity,
                        }
                    )

    df_csv = pd.DataFrame(csv_rows)
    csv_path = output_dir / "voice_quality_aggregated_table.csv"
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")


def _generate_vertical_voice_quality_table(
    output_dir: Path,
    df_voice: pd.DataFrame,
    df_interruption: pd.DataFrame = None,
) -> None:
    """
    Generate vertical LaTeX table with metrics as rows and providers as columns.

    Shows "All" domain data only, with separate rows for Clean (C) and Regular (R)
    conditions where applicable. Designed to fit in a single column.
    """
    import numpy as np

    # Filter to control and regular only
    df = df_voice[df_voice["speech_complexity"].isin(["control", "regular"])].copy()

    if df.empty:
        logger.warning("No control/regular data for vertical voice quality table.")
        return

    # Merge interruption data if available
    if df_interruption is not None:
        df_int = df_interruption[
            df_interruption["speech_complexity"].isin(["control", "regular"])
        ].copy()
        df = df.merge(
            df_int[["llm", "domain", "speech_complexity", "agent_interrupts_count"]],
            on=["llm", "domain", "speech_complexity"],
            how="left",
        )
        df["agent_interruption_rate"] = (
            df["agent_interrupts_count"] / df["response_total"]
        )
    else:
        df["agent_interruption_rate"] = np.nan

    # Get all models (by llm, not provider)
    all_llms_in_data = set(df["llm"].unique())
    llms_sorted = sorted(list(all_llms_in_data), key=get_model_sort_key)

    def get_agg_data(condition):
        """Get aggregated data for a condition across all domains."""
        cond_df = df[df["speech_complexity"] == condition]
        result = {}
        for llm in llms_sorted:
            pdata = cond_df[cond_df["llm"] == llm]
            if not pdata.empty:
                result[llm] = {
                    "response_latency": pdata["response_latency_mean"].mean(),
                    "yield_latency": pdata["yield_latency_mean"].mean(),
                    "response_rate": pdata["response_rate"].mean(),
                    "yield_rate": pdata["yield_rate"].mean(),
                    "agent_interruption_rate": pdata["agent_interruption_rate"].mean(),
                    "backchannel_error_rate": pdata["backchannel_error_rate"].mean(),
                    "vocal_tic_error_rate": pdata["vocal_tic_error_rate"].mean(),
                    "non_directed_error_rate": pdata["non_directed_error_rate"].mean(),
                }
        return result

    ctrl_data = get_agg_data("control")
    reg_data = get_agg_data("regular")

    # Helper functions
    def fmt_sec(val, best_val):
        if pd.isna(val):
            return "--"
        sec = f"{val:.2f}s"
        if (
            best_val is not None
            and not pd.isna(best_val)
            and abs(val - best_val) < 0.001
        ):
            return rf"\textbf{{{sec}}}"
        return sec

    def fmt_pct_higher(val, best_val):
        if pd.isna(val):
            return "--"
        pct = f"{round(val * 100)}\\%"
        if (
            best_val is not None
            and not pd.isna(best_val)
            and abs(val - best_val) < 0.001
        ):
            return rf"\textbf{{{pct}}}"
        return pct

    def fmt_pct_lower(val, best_val):
        if pd.isna(val):
            return "--"
        pct = f"{round(val * 100)}\\%"
        if (
            best_val is not None
            and not pd.isna(best_val)
            and abs(val - best_val) < 0.001
        ):
            return rf"\textbf{{{pct}}}"
        return pct

    def get_best(data_dict, key, minimize=False):
        vals = [
            d.get(key)
            for d in data_dict.values()
            if d.get(key) is not None and not pd.isna(d.get(key))
        ]
        if not vals:
            return None
        return min(vals) if minimize else max(vals)

    # Build the table
    num_models = len(llms_sorted)
    col_spec = "l" + "c" * num_models

    lines = []
    lines.append(rf"\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}")
    lines.append(r"\toprule")

    # Header row with model names
    model_names = [get_short_llm_name(llm, max_len=25) for llm in llms_sorted]
    header = (
        r"\textbf{Metric} & "
        + " & ".join([rf"\textbf{{{m}}}" for m in model_names])
        + r" \\"
    )
    lines.append(header)
    lines.append(r"\midrule")

    # Define metric rows: (label, data_source, key, format_func, minimize, category)
    # Group by category with separators
    metrics = [
        # Latency section
        (r"$L_R$ (C)", "ctrl", "response_latency", fmt_sec, True),
        (r"$L_R$ (R)", "reg", "response_latency", fmt_sec, True),
        (r"$L_Y$", "reg", "yield_latency", fmt_sec, True),
        # Responsiveness section
        (r"$R_R$ (C)", "ctrl", "response_rate", fmt_pct_higher, False),
        (r"$R_R$ (R)", "reg", "response_rate", fmt_pct_higher, False),
        (r"$R_Y$", "reg", "yield_rate", fmt_pct_higher, False),
        # Interrupt section
        (r"$I_A$ (C)", "ctrl", "agent_interruption_rate", fmt_pct_lower, True),
        (r"$I_A$ (R)", "reg", "agent_interruption_rate", fmt_pct_lower, True),
        # Selectivity section (converted to correct rate = 1 - error)
        (r"$S_{BC}$", "reg", "backchannel_correct", fmt_pct_higher, False),
        (r"$S_{VT}$", "reg", "vocal_tic_correct", fmt_pct_higher, False),
        (r"$S_{ND}$", "reg", "non_directed_correct", fmt_pct_higher, False),
    ]

    # Add correct rates to reg_data
    for llm in llms_sorted:
        if llm in reg_data:
            reg_data[llm]["backchannel_correct"] = (
                1 - reg_data[llm].get("backchannel_error_rate", 1)
                if not pd.isna(reg_data[llm].get("backchannel_error_rate"))
                else np.nan
            )
            reg_data[llm]["vocal_tic_correct"] = (
                1 - reg_data[llm].get("vocal_tic_error_rate", 1)
                if not pd.isna(reg_data[llm].get("vocal_tic_error_rate"))
                else np.nan
            )
            reg_data[llm]["non_directed_correct"] = (
                1 - reg_data[llm].get("non_directed_error_rate", 1)
                if not pd.isna(reg_data[llm].get("non_directed_error_rate"))
                else np.nan
            )

    # Section headers for grouping
    section_starts = {
        0: "Latency",
        3: "Responsiveness",
        6: "Interrupt",
        8: "Selectivity",
    }

    for i, (label, source, key, fmt_func, minimize) in enumerate(metrics):
        # Add section header if needed
        if i in section_starts:
            if i > 0:
                lines.append(r"\midrule")
            # Add section header spanning all columns
            lines.append(
                rf"\multicolumn{{{num_models + 1}}}{{l}}{{\textit{{{section_starts[i]}}}}} \\"
            )

        data = ctrl_data if source == "ctrl" else reg_data
        best_val = get_best(data, key, minimize=minimize)

        row_vals = []
        for llm in llms_sorted:
            if llm in data:
                val = data[llm].get(key)
                row_vals.append(fmt_func(val, best_val))
            else:
                row_vals.append("--")

        lines.append(f"{label} & " + " & ".join(row_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = output_dir / "voice_quality_vertical_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")


def _generate_core_metrics_table(
    output_dir: Path,
    df_voice: pd.DataFrame,
    df_interruption: pd.DataFrame = None,
) -> None:
    """
    Generate LaTeX table comparing core metrics between control and regular conditions.

    Core metrics (applicable to both control and regular):
    - Response Rate (higher is better)
    - Response Latency (lower is better)
    - Agent Interruption Rate (lower is better)

    These metrics don't depend on speech complexity features like backchannels,
    vocal tics, or non-directed speech.
    """
    import numpy as np

    # Filter to control and regular only
    df = df_voice[df_voice["speech_complexity"].isin(["control", "regular"])].copy()

    if df.empty:
        logger.warning("No control/regular data for core metrics table.")
        return

    # Merge interruption data if available
    if df_interruption is not None:
        df_int = df_interruption[
            df_interruption["speech_complexity"].isin(["control", "regular"])
        ].copy()
        # Merge on llm, domain, speech_complexity
        df = df.merge(
            df_int[["llm", "domain", "speech_complexity", "agent_interrupts_count"]],
            on=["llm", "domain", "speech_complexity"],
            how="left",
        )
        # Compute agent interruption rate
        df["agent_interruption_rate"] = (
            df["agent_interrupts_count"] / df["response_total"]
        )
    else:
        df["agent_interrupts_count"] = np.nan
        df["agent_interruption_rate"] = np.nan

    # Get all models (by llm, not provider)
    all_llms_in_data = set(df["llm"].unique())
    llms_sorted = sorted(list(all_llms_in_data), key=get_model_sort_key)

    # Generate LaTeX table
    lines = []
    lines.append(r"\begin{tabular}{@{}ll|ccc|ccc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r" & & \multicolumn{3}{c|}{\textbf{Control}} & "
        r"\multicolumn{3}{c}{\textbf{Regular}} \\"
    )
    lines.append(
        r"\textbf{Domain} & \textbf{Model} & "
        r"$L_R\downarrow$ & $R_R\uparrow$ & $I_A\downarrow$ & "
        r"$L_R\downarrow$ & $R_R\uparrow$ & $I_A\downarrow$ \\"
    )
    lines.append(r"\midrule")

    def fmt_sec(val, best_val):
        if pd.isna(val):
            return "--"
        sec = f"{val:.2f}s"
        if (
            best_val is not None
            and not pd.isna(best_val)
            and abs(val - best_val) < 0.001
        ):
            return rf"\textbf{{{sec}}}"
        return sec

    def fmt_pct_higher(val, best_val):
        if pd.isna(val):
            return "--"
        pct = f"{round(val * 100)}\\%"
        if (
            best_val is not None
            and not pd.isna(best_val)
            and abs(val - best_val) < 0.001
        ):
            return rf"\textbf{{{pct}}}"
        return pct

    def fmt_pct_lower(val, best_val):
        if pd.isna(val):
            return "--"
        pct = f"{round(val * 100)}\\%"
        if (
            best_val is not None
            and not pd.isna(best_val)
            and abs(val - best_val) < 0.001
        ):
            return rf"\textbf{{{pct}}}"
        return pct

    # Build "All" aggregate data
    all_agg = {"control": {}, "regular": {}}
    for complexity in ["control", "regular"]:
        for llm in llms_sorted:
            pdata = df[(df["llm"] == llm) & (df["speech_complexity"] == complexity)]
            if not pdata.empty:
                all_agg[complexity][llm] = {
                    "response_latency": pdata["response_latency_mean"].mean(),
                    "response_rate": pdata["response_rate"].mean(),
                    "agent_interruption_rate": pdata["agent_interruption_rate"].mean(),
                }

    # Find best values for "All"
    def get_best(agg_dict, metric, minimize=False):
        vals = [d[metric] for d in agg_dict.values() if not pd.isna(d.get(metric))]
        if not vals:
            return None
        return min(vals) if minimize else max(vals)

    best_ctrl_latency = get_best(all_agg["control"], "response_latency", minimize=True)
    best_ctrl_resp = get_best(all_agg["control"], "response_rate", minimize=False)
    best_ctrl_int = get_best(
        all_agg["control"], "agent_interruption_rate", minimize=True
    )
    best_reg_latency = get_best(all_agg["regular"], "response_latency", minimize=True)
    best_reg_resp = get_best(all_agg["regular"], "response_rate", minimize=False)
    best_reg_int = get_best(
        all_agg["regular"], "agent_interruption_rate", minimize=True
    )

    # Add "All" rows
    for i, llm in enumerate(llms_sorted):
        model_name = get_short_llm_name(llm, max_len=25)
        if i == 0:
            domain_label = rf"\multirow{{{len(llms_sorted)}}}{{*}}{{All}}"
        else:
            domain_label = ""

        ctrl = all_agg["control"].get(llm, {})
        reg = all_agg["regular"].get(llm, {})

        lines.append(
            f"{domain_label} & {model_name} & "
            f"{fmt_sec(ctrl.get('response_latency'), best_ctrl_latency)} & "
            f"{fmt_pct_higher(ctrl.get('response_rate'), best_ctrl_resp)} & "
            f"{fmt_pct_lower(ctrl.get('agent_interruption_rate'), best_ctrl_int)} & "
            f"{fmt_sec(reg.get('response_latency'), best_reg_latency)} & "
            f"{fmt_pct_higher(reg.get('response_rate'), best_reg_resp)} & "
            f"{fmt_pct_lower(reg.get('agent_interruption_rate'), best_reg_int)} \\\\"
        )

    lines.append(r"\midrule")

    # Add domain-specific rows
    for domain in DOMAINS:
        domain_data = df[df["domain"] == domain]

        domain_agg = {"control": {}, "regular": {}}
        for complexity in ["control", "regular"]:
            for llm in llms_sorted:
                pdata = domain_data[
                    (domain_data["llm"] == llm)
                    & (domain_data["speech_complexity"] == complexity)
                ]
                if not pdata.empty:
                    row = pdata.iloc[0]
                    domain_agg[complexity][llm] = {
                        "response_latency": row.get("response_latency_mean"),
                        "response_rate": row.get("response_rate"),
                        "agent_interruption_rate": row.get("agent_interruption_rate"),
                    }

        # Find best values for this domain
        best_ctrl_latency = get_best(
            domain_agg["control"], "response_latency", minimize=True
        )
        best_ctrl_resp = get_best(
            domain_agg["control"], "response_rate", minimize=False
        )
        best_ctrl_int = get_best(
            domain_agg["control"], "agent_interruption_rate", minimize=True
        )
        best_reg_latency = get_best(
            domain_agg["regular"], "response_latency", minimize=True
        )
        best_reg_resp = get_best(domain_agg["regular"], "response_rate", minimize=False)
        best_reg_int = get_best(
            domain_agg["regular"], "agent_interruption_rate", minimize=True
        )

        domain_llms = [
            llm
            for llm in llms_sorted
            if llm in domain_agg["control"] or llm in domain_agg["regular"]
        ]

        for i, llm in enumerate(domain_llms):
            model_name = get_short_llm_name(llm, max_len=25)
            if i == 0:
                domain_label = (
                    rf"\multirow{{{len(domain_llms)}}}{{*}}{{{domain.capitalize()}}}"
                )
            else:
                domain_label = ""

            ctrl = domain_agg["control"].get(llm, {})
            reg = domain_agg["regular"].get(llm, {})

            lines.append(
                f"{domain_label} & {model_name} & "
                f"{fmt_sec(ctrl.get('response_latency'), best_ctrl_latency)} & "
                f"{fmt_pct_higher(ctrl.get('response_rate'), best_ctrl_resp)} & "
                f"{fmt_pct_lower(ctrl.get('agent_interruption_rate'), best_ctrl_int)} & "
                f"{fmt_sec(reg.get('response_latency'), best_reg_latency)} & "
                f"{fmt_pct_higher(reg.get('response_rate'), best_reg_resp)} & "
                f"{fmt_pct_lower(reg.get('agent_interruption_rate'), best_reg_int)} \\\\"
            )

        if domain != DOMAINS[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = output_dir / "core_metrics_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")

    # Also save as CSV
    csv_rows = []
    for domain in ["All"] + DOMAINS:
        for llm in llms_sorted:
            model_name = get_short_llm_name(llm, max_len=25)
            provider = get_provider_display(get_provider_key(llm))
            for complexity in ["control", "regular"]:
                if domain == "All":
                    pdata = df[
                        (df["llm"] == llm) & (df["speech_complexity"] == complexity)
                    ]
                    if not pdata.empty:
                        csv_rows.append(
                            {
                                "domain": domain,
                                "model": model_name,
                                "provider": provider,
                                "complexity": complexity,
                                "response_latency": pdata[
                                    "response_latency_mean"
                                ].mean(),
                                "response_rate": pdata["response_rate"].mean(),
                                "agent_interruption_rate": pdata[
                                    "agent_interruption_rate"
                                ].mean(),
                            }
                        )
                else:
                    pdata = df[
                        (df["domain"] == domain)
                        & (df["llm"] == llm)
                        & (df["speech_complexity"] == complexity)
                    ]
                    if not pdata.empty:
                        row = pdata.iloc[0]
                        csv_rows.append(
                            {
                                "domain": domain,
                                "model": model_name,
                                "provider": provider,
                                "complexity": complexity,
                                "response_latency": row.get("response_latency_mean"),
                                "response_rate": row.get("response_rate"),
                                "agent_interruption_rate": row.get(
                                    "agent_interruption_rate"
                                ),
                            }
                        )

    df_csv = pd.DataFrame(csv_rows)
    csv_path = output_dir / "core_metrics_table.csv"
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")


def _generate_full_voice_quality_table(
    output_dir: Path,
    df_voice: pd.DataFrame,
    df_interruption: pd.DataFrame = None,
) -> None:
    """
    Generate LaTeX table with separate Clean/Real columns for L_R, R_R, I_A.

    Three-row header structure:
    - Row 1: Category headers (Latency, Responsiveness, Interrupt, Selectivity)
    - Row 2: Metric names ($L_R$, $L_Y$, $R_R$, $R_Y$, $I_A$, $S_{BC}$, $S_{VT}$, $S_{ND}$)
    - Row 3: Condition labels (C/R) for metrics with both conditions
    """
    import numpy as np

    # Filter to control and regular only
    df = df_voice[df_voice["speech_complexity"].isin(["control", "regular"])].copy()

    if df.empty:
        logger.warning("No control/regular data for full voice quality table.")
        return

    # Merge interruption data if available
    if df_interruption is not None:
        df_int = df_interruption[
            df_interruption["speech_complexity"].isin(["control", "regular"])
        ].copy()
        df = df.merge(
            df_int[["llm", "domain", "speech_complexity", "agent_interrupts_count"]],
            on=["llm", "domain", "speech_complexity"],
            how="left",
        )
        df["agent_interruption_rate"] = (
            df["agent_interrupts_count"] / df["response_total"]
        )
    else:
        df["agent_interruption_rate"] = np.nan

    # Get all models (by llm, not provider)
    all_llms_in_data = set(df["llm"].unique())
    llms_sorted = sorted(list(all_llms_in_data), key=get_model_sort_key)

    # Helper functions
    def fmt_sec(val, best_val):
        if pd.isna(val):
            return "--"
        sec = f"{val:.2f}s"
        if (
            best_val is not None
            and not pd.isna(best_val)
            and abs(val - best_val) < 0.001
        ):
            return rf"\textbf{{{sec}}}"
        return sec

    def fmt_pct_higher(val, best_val):
        if pd.isna(val):
            return "--"
        pct = f"{round(val * 100)}\\%"
        if (
            best_val is not None
            and not pd.isna(best_val)
            and abs(val - best_val) < 0.001
        ):
            return rf"\textbf{{{pct}}}"
        return pct

    def fmt_pct_lower(val, best_val):
        if pd.isna(val):
            return "--"
        pct = f"{round(val * 100)}\\%"
        if (
            best_val is not None
            and not pd.isna(best_val)
            and abs(val - best_val) < 0.001
        ):
            return rf"\textbf{{{pct}}}"
        return pct

    def get_best(data_list, metric, minimize=False):
        vals = [
            d.get(metric)
            for d in data_list
            if d.get(metric) is not None and not pd.isna(d.get(metric))
        ]
        if not vals:
            return None
        return min(vals) if minimize else max(vals)

    # Generate LaTeX table with 3-row header
    # Column structure: Domain, Model, L_R(C), L_R(R), L_Y, R_R(C), R_R(R), R_Y, I_A(C), I_A(R), S_BC, S_VT, S_ND
    lines = []
    lines.append(r"\begin{tabular}{@{}ll|cc|c|cc|c|cc|ccc@{}}")
    lines.append(r"\toprule")
    # Row 1: Category headers
    lines.append(
        r" & & \multicolumn{3}{c|}{\textbf{Latency}$\downarrow$} & "
        r"\multicolumn{3}{c|}{\textbf{Responsiveness}$\uparrow$} & "
        r"\multicolumn{2}{c|}{\textbf{Interrupt}$\downarrow$} & "
        r"\multicolumn{3}{c}{\textbf{Selectivity}$\uparrow$} \\"
    )
    # Row 2: Metric names (with multirow for single-condition metrics)
    lines.append(
        r" & & \multicolumn{2}{c|}{$L_R$} & \multirow{2}{*}{$L_Y$} & "
        r"\multicolumn{2}{c|}{$R_R$} & \multirow{2}{*}{$R_Y$} & "
        r"\multicolumn{2}{c|}{$I_A$} & "
        r"\multirow{2}{*}{$S_{BC}$} & \multirow{2}{*}{$S_{VT}$} & \multirow{2}{*}{$S_{ND}$} \\"
    )
    # Row 3: Condition labels
    lines.append(
        r"\textbf{Domain} & \textbf{Model} & C & R & & C & R & & C & R & & & \\"
    )
    lines.append(r"\midrule")

    def get_row_data(pdata):
        """Extract metrics from a row or aggregated data."""
        if pdata.empty:
            return {}
        if len(pdata) == 1:
            row = pdata.iloc[0]
            data = {
                "response_latency": row.get("response_latency_mean"),
                "yield_latency": row.get("yield_latency_mean"),
                "response_rate": row.get("response_rate"),
                "yield_rate": row.get("yield_rate"),
                "agent_interruption_rate": row.get("agent_interruption_rate"),
                "backchannel_error_rate": row.get("backchannel_error_rate"),
                "vocal_tic_error_rate": row.get("vocal_tic_error_rate"),
                "non_directed_error_rate": row.get("non_directed_error_rate"),
            }
        else:
            data = {
                "response_latency": pdata["response_latency_mean"].mean(),
                "yield_latency": pdata["yield_latency_mean"].mean(),
                "response_rate": pdata["response_rate"].mean(),
                "yield_rate": pdata["yield_rate"].mean(),
                "agent_interruption_rate": pdata["agent_interruption_rate"].mean(),
                "backchannel_error_rate": pdata["backchannel_error_rate"].mean(),
                "vocal_tic_error_rate": pdata["vocal_tic_error_rate"].mean(),
                "non_directed_error_rate": pdata["non_directed_error_rate"].mean(),
            }
        return data

    # Process each domain (All first, then individual domains)
    for domain in ["All"] + DOMAINS:
        if domain == "All":
            domain_df = df
        else:
            domain_df = df[df["domain"] == domain]

        if domain_df.empty:
            continue

        # Collect data for all models
        ctrl_data = {}
        reg_data = {}
        for llm in llms_sorted:
            ctrl_pdata = domain_df[
                (domain_df["llm"] == llm)
                & (domain_df["speech_complexity"] == "control")
            ]
            reg_pdata = domain_df[
                (domain_df["llm"] == llm)
                & (domain_df["speech_complexity"] == "regular")
            ]
            ctrl_data[llm] = get_row_data(ctrl_pdata)
            reg_data[llm] = get_row_data(reg_pdata)

        # Find best values for this domain
        ctrl_list = [v for v in ctrl_data.values() if v]
        reg_list = [v for v in reg_data.values() if v]

        # Best values for Regular
        best_reg_lr = get_best(reg_list, "response_latency", minimize=True)
        best_reg_ly = get_best(reg_list, "yield_latency", minimize=True)
        best_reg_rr = get_best(reg_list, "response_rate", minimize=False)
        best_reg_ry = get_best(reg_list, "yield_rate", minimize=False)
        best_reg_ia = get_best(reg_list, "agent_interruption_rate", minimize=True)
        best_reg_bc = get_best(
            [
                {"v": 1 - d.get("backchannel_error_rate", 1)}
                for d in reg_list
                if d.get("backchannel_error_rate") is not None
                and not pd.isna(d.get("backchannel_error_rate"))
            ],
            "v",
            minimize=False,
        )
        best_reg_vt = get_best(
            [
                {"v": 1 - d.get("vocal_tic_error_rate", 1)}
                for d in reg_list
                if d.get("vocal_tic_error_rate") is not None
                and not pd.isna(d.get("vocal_tic_error_rate"))
            ],
            "v",
            minimize=False,
        )
        best_reg_nd = get_best(
            [
                {"v": 1 - d.get("non_directed_error_rate", 1)}
                for d in reg_list
                if d.get("non_directed_error_rate") is not None
                and not pd.isna(d.get("non_directed_error_rate"))
            ],
            "v",
            minimize=False,
        )

        # Best values for Control (only L_R, R_R, I_A)
        best_ctrl_lr = get_best(ctrl_list, "response_latency", minimize=True)
        best_ctrl_rr = get_best(ctrl_list, "response_rate", minimize=False)
        best_ctrl_ia = get_best(ctrl_list, "agent_interruption_rate", minimize=True)

        domain_llms = [
            llm for llm in llms_sorted if ctrl_data.get(llm) or reg_data.get(llm)
        ]

        for i, llm in enumerate(domain_llms):
            model_name = get_short_llm_name(llm, max_len=25)
            if i == 0:
                domain_label = rf"\multirow{{{len(domain_llms)}}}{{*}}{{{domain.capitalize() if domain != 'All' else 'All'}}}"
            else:
                domain_label = ""

            ctrl = ctrl_data.get(llm, {})
            reg = reg_data.get(llm, {})

            # Compute selectivity correct rates
            bc_correct = (
                1 - reg.get("backchannel_error_rate", 1)
                if reg.get("backchannel_error_rate") is not None
                and not pd.isna(reg.get("backchannel_error_rate"))
                else np.nan
            )
            vt_correct = (
                1 - reg.get("vocal_tic_error_rate", 1)
                if reg.get("vocal_tic_error_rate") is not None
                and not pd.isna(reg.get("vocal_tic_error_rate"))
                else np.nan
            )
            nd_correct = (
                1 - reg.get("non_directed_error_rate", 1)
                if reg.get("non_directed_error_rate") is not None
                and not pd.isna(reg.get("non_directed_error_rate"))
                else np.nan
            )

            # Build row with separate C/R columns
            lines.append(
                f"{domain_label} & {model_name} & "
                # Latency: L_R(C), L_R(R), L_Y
                f"{fmt_sec(ctrl.get('response_latency'), best_ctrl_lr)} & "
                f"{fmt_sec(reg.get('response_latency'), best_reg_lr)} & "
                f"{fmt_sec(reg.get('yield_latency'), best_reg_ly)} & "
                # Responsiveness: R_R(C), R_R(R), R_Y
                f"{fmt_pct_higher(ctrl.get('response_rate'), best_ctrl_rr)} & "
                f"{fmt_pct_higher(reg.get('response_rate'), best_reg_rr)} & "
                f"{fmt_pct_higher(reg.get('yield_rate'), best_reg_ry)} & "
                # Interrupt: I_A(C), I_A(R)
                f"{fmt_pct_lower(ctrl.get('agent_interruption_rate'), best_ctrl_ia)} & "
                f"{fmt_pct_lower(reg.get('agent_interruption_rate'), best_reg_ia)} & "
                # Selectivity: regular only
                f"{fmt_pct_higher(bc_correct, best_reg_bc)} & "
                f"{fmt_pct_higher(vt_correct, best_reg_vt)} & "
                f"{fmt_pct_higher(nd_correct, best_reg_nd)} \\\\"
            )

        if domain != DOMAINS[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = output_dir / "full_voice_quality_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")


# =============================================================================
# CLI
# =============================================================================


def get_cli_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate paper-ready outputs from existing analysis CSVs."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root analysis directory (containing performance_analysis/ and voice_analysis/).",
    )
    parser.add_argument(
        "--performance-only",
        action="store_true",
        help="Only generate performance analysis outputs.",
    )
    parser.add_argument(
        "--voice-only",
        action="store_true",
        help="Only generate voice analysis outputs.",
    )
    parser.add_argument(
        "--copy-to-paper",
        type=str,
        default=None,
        help="Copy outputs to paper results directory (e.g., papers/tau-voice/results/).",
    )
    return parser


def main():
    parser = get_cli_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return

    copy_to_paper_dir = Path(args.copy_to_paper) if args.copy_to_paper else None

    generate_all_paper_outputs(
        output_dir,
        performance_only=args.performance_only,
        voice_only=args.voice_only,
        copy_to_paper_dir=copy_to_paper_dir,
    )


if __name__ == "__main__":
    main()
