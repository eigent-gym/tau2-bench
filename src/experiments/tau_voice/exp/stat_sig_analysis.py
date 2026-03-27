#!/usr/bin/env python3
"""
Statistical significance analysis for tau-voice experiments.

Loads voice and text trial results, creates a unified CSV with per-simulation
data, and computes statistical significance (Wilson CIs, pairwise comparisons)
per domain and condition.

Usage:
    # Retail analysis with 2 voice trials and text baselines
    python -m experiments.tau_voice.exp.stat_sig_analysis \
        --voice-dirs data/exp/final_voice_results_lite/voice_trial1/base \
                     data/exp/final_voice_results_lite/voice_trial2/base \
        --text-dir data/exp/final_text_results_lite \
        --domain retail \
        --output-dir data/exp/stat_sig_output

    # All domains, voice only
    python -m experiments.tau_voice.exp.stat_sig_analysis \
        --voice-dirs data/exp/final_voice_results_lite/voice_trial1/base \
        --output-dir data/exp/stat_sig_output

    # Voice only, no text baselines
    python -m experiments.tau_voice.exp.stat_sig_analysis \
        --voice-dirs data/exp/final_voice_results_lite/voice_trial1/base \
                     data/exp/final_voice_results_lite/voice_trial2/base \
        --domain retail \
        --output-dir data/exp/stat_sig_output
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Optional

import pandas as pd

CONDITION_ORDER = ["text", "control", "regular"]
CONDITION_DISPLAY = {
    "text": "Text",
    "control": "Voice (Clean)",
    "regular": "Voice (Realistic)",
}

PROVIDER_DISPLAY = {
    "gemini": "Google",
    "openai": "OpenAI",
    "xai": "xAI",
}


# =============================================================================
# Wilson CI
# =============================================================================


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


# =============================================================================
# Data Loading
# =============================================================================


def _parse_provider(llm: str) -> str:
    """Extract provider from LLM string like 'gemini:model-name'."""
    if ":" in llm:
        return llm.split(":")[0]
    if llm.startswith("gpt"):
        return "openai"
    return "unknown"


def _short_model_name(llm: str) -> str:
    """Derive a short display name from the raw LLM string."""
    name = llm.split(":")[-1] if ":" in llm else llm
    name = re.sub(r"-\d{4}-\d{2}-\d{2}.*$", "", name)
    return name


def load_voice_trials(
    voice_dirs: list[Path],
    domain_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Load voice results from multiple trial directories.

    Each directory represents one independent trial. Within each directory,
    subdirectories contain results.json files for each (domain, complexity, model)
    combination with a single trial (trial=0).
    """
    rows = []
    for trial_idx, voice_dir in enumerate(voice_dirs):
        if not voice_dir.exists():
            print(f"  Warning: voice dir not found, skipping: {voice_dir}")
            continue

        n_loaded = 0
        for sim_subdir in sorted(voice_dir.iterdir()):
            if not sim_subdir.is_dir():
                continue
            results_file = sim_subdir / "results.json"
            if not results_file.exists():
                continue

            with open(results_file) as f:
                data = json.load(f)

            info = data["info"]
            domain = info["environment_info"]["domain_name"]
            if domain_filter and domain != domain_filter:
                continue

            complexity = info.get("speech_complexity") or "unknown"
            llm = info["agent_info"]["llm"]
            provider = _parse_provider(llm)

            for sim in data.get("simulations", []):
                reward = sim["reward_info"]["reward"] if sim.get("reward_info") else 0.0
                rows.append(
                    {
                        "domain": domain,
                        "task_id": sim["task_id"],
                        "trial": trial_idx,
                        "modality": "voice",
                        "model": llm,
                        "provider": provider,
                        "condition": complexity,
                        "reward": reward,
                        "success": int(reward == 1.0),
                    }
                )
                n_loaded += 1

        print(
            f"  Trial {trial_idx} ({voice_dir.parent.name}/{voice_dir.name}): {n_loaded} simulations"
        )

    return pd.DataFrame(rows)


def load_text_trials(
    text_dir: Path,
    domain_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Load text results (multiple internal trials per results.json)."""
    rows = []
    if not text_dir.exists():
        print(f"  Warning: text dir not found, skipping: {text_dir}")
        return pd.DataFrame()

    for sim_subdir in sorted(text_dir.iterdir()):
        if not sim_subdir.is_dir():
            continue
        results_file = sim_subdir / "results.json"
        if not results_file.exists():
            continue

        with open(results_file) as f:
            data = json.load(f)

        info = data["info"]
        domain = info["environment_info"]["domain_name"]
        if domain_filter and domain != domain_filter:
            continue

        llm = info["agent_info"]["llm"]
        provider = _parse_provider(llm)
        n_loaded = 0

        for sim in data.get("simulations", []):
            reward = sim["reward_info"]["reward"] if sim.get("reward_info") else 0.0
            rows.append(
                {
                    "domain": domain,
                    "task_id": sim["task_id"],
                    "trial": sim["trial"],
                    "modality": "text",
                    "model": llm,
                    "provider": provider,
                    "condition": "text",
                    "reward": reward,
                    "success": int(reward == 1.0),
                }
            )
            n_loaded += 1

        print(
            f"  {sim_subdir.name}: {n_loaded} simulations ({info.get('num_trials', '?')} trials)"
        )

    return pd.DataFrame(rows)


def build_unified_dataframe(
    voice_dirs: list[Path],
    text_dir: Optional[Path],
    domain_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Load all data and return a unified DataFrame."""
    dfs = []

    if voice_dirs:
        print(f"Loading voice trials from {len(voice_dirs)} directories...")
        df_voice = load_voice_trials(voice_dirs, domain_filter)
        if not df_voice.empty:
            dfs.append(df_voice)
            print(f"  → {len(df_voice)} voice simulations total")

    if text_dir:
        print(f"Loading text trials from {text_dir}...")
        df_text = load_text_trials(text_dir, domain_filter)
        if not df_text.empty:
            dfs.append(df_text)
            print(f"  → {len(df_text)} text simulations total")

    if not dfs:
        print("No data loaded!")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df["model_short"] = df["model"].apply(_short_model_name)
    df["provider_display"] = df["provider"].map(PROVIDER_DISPLAY).fillna(df["provider"])
    df["condition_display"] = (
        df["condition"].map(CONDITION_DISPLAY).fillna(df["condition"])
    )

    print(f"\nUnified dataset: {len(df)} simulations")
    print(f"  Domains: {sorted(df['domain'].unique())}")
    print(f"  Conditions: {sorted(df['condition'].unique())}")
    print(f"  Models: {sorted(df['model_short'].unique())}")
    return df


# =============================================================================
# Statistical Analysis
# =============================================================================


def compute_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(domain, condition, model) statistics.

    For each group:
    - Per-trial success counts and rates
    - Mean, std, SEM across trials (trial-level variability)
    - Mean ± 1.96*SEM (95% CI assuming normal approximation)
    - Pooled Wilson CI (treating each sim as independent Bernoulli)
    """
    groups = df.groupby(["domain", "condition", "model"])

    stats_rows = []
    for (domain, condition, model), group_df in groups:
        trial_stats = (
            group_df.groupby("trial")
            .agg(successes=("success", "sum"), n=("success", "count"))
            .reset_index()
        )
        trial_stats["rate"] = trial_stats["successes"] / trial_stats["n"]

        n_trials = len(trial_stats)
        n_per_trial = int(trial_stats["n"].median())
        rates = trial_stats["rate"].tolist()

        mean_rate = sum(rates) / len(rates)
        if n_trials >= 2:
            std_rate = math.sqrt(
                sum((r - mean_rate) ** 2 for r in rates) / (n_trials - 1)
            )
            sem = std_rate / math.sqrt(n_trials)
        else:
            std_rate = 0.0
            sem = 0.0
        ci95_half = 1.96 * sem

        total_success = int(trial_stats["successes"].sum())
        total_n = int(trial_stats["n"].sum())
        w_lo, w_hi = wilson_ci(total_success, total_n)

        provider = group_df["provider"].iloc[0]
        trial_successes_str = ";".join(
            str(int(s)) for s in trial_stats["successes"].tolist()
        )

        stats_rows.append(
            {
                "domain": domain,
                "condition": condition,
                "condition_display": CONDITION_DISPLAY.get(condition, condition),
                "model": model,
                "model_short": _short_model_name(model),
                "provider": provider,
                "provider_display": PROVIDER_DISPLAY.get(provider, provider),
                "n_trials": n_trials,
                "n_per_trial": n_per_trial,
                "trial_successes": trial_successes_str,
                "mean": mean_rate,
                "std": std_rate,
                "sem": sem,
                "ci95_lo": mean_rate - ci95_half,
                "ci95_hi": mean_rate + ci95_half,
                "wilson_lo": w_lo,
                "wilson_hi": w_hi,
                "total_successes": total_success,
                "total_n": total_n,
            }
        )

    stats_df = pd.DataFrame(stats_rows)

    cond_rank = {c: i for i, c in enumerate(CONDITION_ORDER)}
    stats_df["_cond_rank"] = stats_df["condition"].map(cond_rank).fillna(99)
    stats_df = stats_df.sort_values(["domain", "_cond_rank", "model"]).drop(
        columns="_cond_rank"
    )

    return stats_df.reset_index(drop=True)


def compute_pairwise_comparisons(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Check Wilson CI overlap for key pairwise comparisons.

    Comparisons:
    1. Text vs Voice-Clean (each text model vs each voice model)
    2. Voice-Clean vs Voice-Realistic (same voice model)
    3. Provider vs Provider within the same condition
    """
    comparisons = []

    for domain in sorted(stats_df["domain"].unique()):
        ds = stats_df[stats_df["domain"] == domain]

        text_rows = ds[ds["condition"] == "text"]
        clean_rows = ds[ds["condition"] == "control"]
        realistic_rows = ds[ds["condition"] == "regular"]

        # Text vs Voice Clean
        for _, tr in text_rows.iterrows():
            for _, cr in clean_rows.iterrows():
                sig = (
                    tr["wilson_lo"] > cr["wilson_hi"]
                    or cr["wilson_lo"] > tr["wilson_hi"]
                )
                comparisons.append(
                    _comparison_row(domain, "Text vs Clean", tr, cr, sig)
                )

        # Voice Clean vs Realistic (same model)
        for _, cr in clean_rows.iterrows():
            matching = realistic_rows[realistic_rows["model"] == cr["model"]]
            for _, rr in matching.iterrows():
                sig = (
                    cr["wilson_lo"] > rr["wilson_hi"]
                    or rr["wilson_lo"] > cr["wilson_hi"]
                )
                comparisons.append(
                    _comparison_row(domain, "Clean vs Realistic", cr, rr, sig)
                )

        # Provider vs Provider within condition
        for condition in ["control", "regular"]:
            cond_rows = ds[ds["condition"] == condition].reset_index(drop=True)
            cond_display = CONDITION_DISPLAY.get(condition, condition)
            for i in range(len(cond_rows)):
                for j in range(i + 1, len(cond_rows)):
                    ra, rb = cond_rows.iloc[i], cond_rows.iloc[j]
                    sig = (
                        ra["wilson_lo"] > rb["wilson_hi"]
                        or rb["wilson_lo"] > ra["wilson_hi"]
                    )
                    comparisons.append(
                        _comparison_row(
                            domain,
                            f"Provider vs Provider ({cond_display})",
                            ra,
                            rb,
                            sig,
                        )
                    )

    return pd.DataFrame(comparisons) if comparisons else pd.DataFrame()


def _comparison_row(
    domain: str, comparison: str, row_a: pd.Series, row_b: pd.Series, significant: bool
) -> dict:
    return {
        "domain": domain,
        "comparison": comparison,
        "model_a": row_a["model_short"],
        "model_b": row_b["model_short"],
        "mean_a": row_a["mean"],
        "mean_b": row_b["mean"],
        "wilson_ci_a": f"[{row_a['wilson_lo']:.1%}, {row_a['wilson_hi']:.1%}]",
        "wilson_ci_b": f"[{row_b['wilson_lo']:.1%}, {row_b['wilson_hi']:.1%}]",
        "delta_pp": (row_b["mean"] - row_a["mean"]) * 100,
        "significant": significant,
    }


# =============================================================================
# Output
# =============================================================================


def print_report(stats_df: pd.DataFrame, comparisons_df: pd.DataFrame) -> None:
    """Print formatted report to console."""
    print("\n" + "=" * 90)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 90)

    for domain in sorted(stats_df["domain"].unique()):
        domain_stats = stats_df[stats_df["domain"] == domain]
        n_tasks = domain_stats["n_per_trial"].iloc[0]

        print(f"\n{'─' * 90}")
        print(f"  Domain: {domain.upper()} (n={n_tasks} tasks per trial)")
        print(f"{'─' * 90}")

        for condition in CONDITION_ORDER:
            cond_rows = domain_stats[domain_stats["condition"] == condition]
            if cond_rows.empty:
                continue

            cond_display = CONDITION_DISPLAY.get(condition, condition)
            print(f"\n  {cond_display}:")
            hdr = f"    {'Model':<35} {'Trials':>6}  {'Mean':>7}  {'± 95% CI':>12}  {'Wilson 95% CI':>22}  {'Per-trial':>15}"
            print(hdr)
            print(f"    {'─' * (len(hdr) - 4)}")

            for _, row in cond_rows.iterrows():
                trial_str = f"n={row['n_trials']}"
                successes = row["trial_successes"].split(";")
                per_trial_str = "/".join(f"{s}/{row['n_per_trial']}" for s in successes)
                if row["n_trials"] >= 2:
                    ci_str = f"± {row['sem'] * 1.96 * 100:.1f}pp"
                else:
                    ci_str = "(1 trial)"
                wilson_str = (
                    f"[{row['wilson_lo'] * 100:.1f}%, {row['wilson_hi'] * 100:.1f}%]"
                )

                print(
                    f"    {row['model_short']:<35} {trial_str:>6}  {row['mean'] * 100:>6.1f}%  {ci_str:>12}  {wilson_str:>22}  {per_trial_str:>15}"
                )

    if comparisons_df.empty:
        return

    print(f"\n{'=' * 90}")
    print("PAIRWISE COMPARISONS (non-overlapping Wilson CIs)")
    print(f"{'=' * 90}")

    for domain in sorted(comparisons_df["domain"].unique()):
        dc = comparisons_df[comparisons_df["domain"] == domain]
        print(f"\n{'─' * 90}")
        print(f"  Domain: {domain.upper()}")

        for comp_type in dc["comparison"].unique():
            tc = dc[dc["comparison"] == comp_type]
            print(f"\n  {comp_type}:")

            for _, row in tc.iterrows():
                marker = (
                    "*** SIGNIFICANT" if row["significant"] else "    not significant"
                )
                print(
                    f"    {row['model_a']:<30}  vs  {row['model_b']:<30}  "
                    f"Δ={row['delta_pp']:+6.1f}pp  {marker}"
                )
                print(
                    f"      {row['mean_a'] * 100:5.1f}% {row['wilson_ci_a']}     "
                    f"{row['mean_b'] * 100:5.1f}% {row['wilson_ci_b']}"
                )


def generate_stat_sig_table_tex(
    stats_df: pd.DataFrame,
    output_dir: Path,
    domain_filter: Optional[str] = None,
) -> None:
    """Generate LaTeX table matching the paper's stat_sig_table.tex format."""
    df = stats_df
    if domain_filter:
        df = df[df["domain"] == domain_filter]

    lines = [
        r"\begin{tabular}{llc}",
        r"\toprule",
        r"\textbf{Condition} & \textbf{Model} & \textbf{Mean $\pm$ 95\% CI} \\",
        r"\midrule",
    ]

    domains = sorted(df["domain"].unique())
    for domain in domains:
        domain_df = df[df["domain"] == domain]

        for condition in CONDITION_ORDER:
            cond_rows = domain_df[domain_df["condition"] == condition]
            if cond_rows.empty:
                continue

            cond_display = CONDITION_DISPLAY.get(condition, condition)
            n_models = len(cond_rows)

            for i, (_, row) in enumerate(cond_rows.iterrows()):
                mean_pct = row["mean"] * 100
                ci_pct = row["sem"] * 1.96 * 100

                if row["n_trials"] >= 2:
                    val_str = f"{mean_pct:.1f}\\% $\\pm$ {ci_pct:.1f}\\%"
                else:
                    val_str = f"{mean_pct:.1f}\\%"

                if i == 0:
                    label = rf"\multirow{{{n_models}}}{{*}}{{{cond_display}}}"
                else:
                    label = ""

                lines.append(f"{label} & {row['model_short']} & {val_str} \\\\")

            lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append("")

    tex_path = output_dir / "stat_sig_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved LaTeX table: {tex_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Statistical significance analysis for tau-voice experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Retail with 2 voice trials + text baselines
  python -m experiments.tau_voice.exp.stat_sig_analysis \\
      --voice-dirs data/exp/final_voice_results_lite/voice_trial1/base \\
                   data/exp/final_voice_results_lite/voice_trial2/base \\
      --text-dir data/exp/final_text_results_lite \\
      --domain retail \\
      --output-dir data/exp/stat_sig_output

  # All domains, voice only
  python -m experiments.tau_voice.exp.stat_sig_analysis \\
      --voice-dirs data/exp/final_voice_results_lite/voice_trial1/base \\
      --output-dir data/exp/stat_sig_output
        """,
    )
    parser.add_argument(
        "--voice-dirs",
        type=str,
        nargs="+",
        default=[],
        help="Directories containing voice trial results (each dir = one independent trial).",
    )
    parser.add_argument(
        "--text-dir",
        type=str,
        default=None,
        help="Directory containing text model results (multi-trial per results.json).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter to a specific domain (e.g., retail, airline, telecom).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSV and LaTeX files. Defaults to current directory.",
    )

    args = parser.parse_args()

    if not args.voice_dirs and not args.text_dir:
        parser.error("At least one of --voice-dirs or --text-dir is required.")

    voice_dirs = [Path(d) for d in args.voice_dirs]
    text_dir = Path(args.text_dir) if args.text_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and unify data
    df = build_unified_dataframe(voice_dirs, text_dir, args.domain)
    if df.empty:
        return

    csv_path = output_dir / "all_trials.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved unified CSV: {csv_path}")

    # 2. Compute per-group stats
    stats_df = compute_group_stats(df)
    stats_csv = output_dir / "stat_sig_summary.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"Saved stats CSV: {stats_csv}")

    # 3. Pairwise comparisons
    comparisons_df = compute_pairwise_comparisons(stats_df)
    if not comparisons_df.empty:
        comp_csv = output_dir / "pairwise_comparisons.csv"
        comparisons_df.to_csv(comp_csv, index=False)
        print(f"Saved comparisons CSV: {comp_csv}")

    # 4. Console report
    print_report(stats_df, comparisons_df)

    # 5. LaTeX table
    generate_stat_sig_table_tex(stats_df, output_dir, domain_filter=args.domain)


if __name__ == "__main__":
    main()
