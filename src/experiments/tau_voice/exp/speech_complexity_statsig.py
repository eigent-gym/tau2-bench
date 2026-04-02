#!/usr/bin/env python3
"""
Pairwise statistical significance between speech complexity conditions.

For each (domain, provider), runs pairwise comparisons between all speech
complexity conditions using:
  - Paired permutation test (paired by task_id, on per-task mean success rates)
  - Fisher's exact test (unpaired, on raw simulation counts)
  - Wilson CI overlap check
  - Holm-Bonferroni correction for multiple comparisons

Optionally loads text baselines from results.json files (--text-dir) and
compares them against voice control conditions per domain/provider.

Usage:
    # From pre-processed CSV
    python -m experiments.tau_voice.exp.speech_complexity_statsig \
        --input data/exp/analysis/pass_k_by_domain/pass_k_by_domain_raw.csv

    # From raw voice result directories
    python -m experiments.tau_voice.exp.speech_complexity_statsig \
        --voice-dirs data/exp/run1 data/exp/run2 \
        --output-dir data/exp/statsig_output

    # With text baselines
    python -m experiments.tau_voice.exp.speech_complexity_statsig \
        --voice-dirs data/exp/run1 \
        --text-dir data/exp/text_results \
        --output-dir data/exp/statsig_output \
        --domain retail
"""

import argparse
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from tau2.data_model.simulation import Results


def _clear_line():
    """Clear the current terminal line (overwrite progress indicator)."""
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


COMPLEXITY_ORDER = [
    "text_reasoning",
    "text_non_reasoning",
    "control",
    "control_audio",
    "control_accents",
    "control_behavior",
    "regular",
]

COMPLEXITY_DISPLAY = {
    "text_reasoning": "Text (Reasoning)",
    "text_non_reasoning": "Text (Non-Reasoning)",
    "control": "Control (Clean)",
    "control_audio": "Control (Audio)",
    "control_accents": "Control (Accents)",
    "control_behavior": "Control (Behavior)",
    "regular": "Regular (Realistic)",
}

REASONING_MODELS = {"gpt-5.2", "gpt52-high"}
NON_REASONING_MODELS = {"gpt-4.1-2025-04-14", "gpt-4.1", "gpt41"}


def _classify_text_model(llm: str) -> str:
    """Classify a text model LLM string as reasoning or non-reasoning."""
    name = llm.lower()
    for tag in REASONING_MODELS:
        if tag in name:
            return "text_reasoning"
    for tag in NON_REASONING_MODELS:
        if tag in name:
            return "text_non_reasoning"
    return f"text_{name}"


def _parse_provider(llm: str) -> str:
    """Extract provider from LLM string like 'gemini:model-name'."""
    if ":" in llm:
        return llm.split(":")[0]
    if llm.startswith("gpt"):
        return "openai"
    return "unknown"


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def paired_permutation_test(
    paired_a: np.ndarray,
    paired_b: np.ndarray,
    n_permutations: int = 100_000,
    seed: int = 42,
) -> float:
    """Two-sided permutation test on paired differences.

    Tests H0: mean(A) == mean(B) by permuting the sign of per-task
    differences. Works on continuous per-task success rates (not just binary),
    so the test statistic matches the reported raw rate difference.
    """
    diffs = paired_a - paired_b
    observed = np.abs(np.mean(diffs))

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_permutations, len(diffs)))
    perm_means = np.abs((signs * diffs).mean(axis=1))

    return float(np.mean(perm_means >= observed))


def bootstrap_delta_ci(
    paired_a: np.ndarray,
    paired_b: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """BCa bootstrap confidence interval on the paired mean difference (A - B).

    Uses the bias-corrected and accelerated (BCa) method for more accurate
    coverage than the percentile method, especially with skewed distributions.
    Returns (ci_lo, ci_hi) in raw proportion units (multiply by 100 for pp).
    """
    diffs = paired_a - paired_b
    n = len(diffs)
    observed = np.mean(diffs)

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = diffs[boot_indices].mean(axis=1)

    # Bias correction factor
    z0 = _norm_ppf(np.mean(boot_means < observed))

    # Acceleration factor (jackknife)
    jack_means = np.array([np.mean(np.delete(diffs, i)) for i in range(n)])
    jack_mean = jack_means.mean()
    num = np.sum((jack_mean - jack_means) ** 3)
    denom = 6.0 * np.sum((jack_mean - jack_means) ** 2) ** 1.5
    a = num / denom if denom != 0 else 0.0

    # Adjusted percentiles
    z_lo = _norm_ppf(alpha / 2)
    z_hi = _norm_ppf(1 - alpha / 2)

    alpha_lo = _norm_cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    alpha_hi = _norm_cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))

    # Clamp to valid percentile range
    alpha_lo = np.clip(alpha_lo, 1 / n_bootstrap, 1 - 1 / n_bootstrap)
    alpha_hi = np.clip(alpha_hi, 1 / n_bootstrap, 1 - 1 / n_bootstrap)

    sorted_boots = np.sort(boot_means)
    ci_lo = float(np.percentile(sorted_boots, alpha_lo * 100))
    ci_hi = float(np.percentile(sorted_boots, alpha_hi * 100))
    return ci_lo, ci_hi


def _norm_ppf(p: float) -> float:
    """Normal distribution percent point function (inverse CDF)."""
    from scipy.stats import norm

    return float(norm.ppf(np.clip(p, 1e-10, 1 - 1e-10)))


def _norm_cdf(z: float) -> float:
    """Normal distribution CDF."""
    from scipy.stats import norm

    return float(norm.cdf(z))


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction for multiple comparisons.

    Returns adjusted p-values (capped at 1.0).
    """
    m = len(p_values)
    if m <= 1:
        return list(p_values)

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (m - rank)
        cummax = max(cummax, adj)
        adjusted[orig_idx] = min(cummax, 1.0)
    return adjusted


# =============================================================================
# Data Loading
# =============================================================================


def load_text_results(
    text_dir: Path,
    domain_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Load text model results from results.json files.

    Uses Results.load_metadata() for metadata and either simulation_index
    or iter_simulations() for per-simulation data.

    Returns a DataFrame in the same schema as the voice CSV:
    simulation_id, task_id, trial, domain, speech_complexity, llm, provider, reward, success
    """
    rows = []
    subdirs = sorted(d for d in text_dir.iterdir() if d.is_dir())
    n_dirs = len(subdirs)
    n_loaded = 0
    for dir_idx, subdir in enumerate(subdirs, 1):
        results_file = subdir / "results.json"
        if not results_file.exists():
            continue

        print(
            f"\r  {text_dir.name}: {dir_idx}/{n_dirs} dirs, {n_loaded} sims loaded",
            end="",
            flush=True,
        )

        results = Results.load_metadata(results_file)
        info = results.info
        domain = info.environment_info.domain_name
        if domain_filter and domain != domain_filter:
            continue

        llm = info.agent_info.llm or "unknown"
        speech_complexity = _classify_text_model(llm)

        if results.simulation_index:
            for entry in results.simulation_index:
                reward = entry.reward if entry.reward is not None else 0.0
                rows.append(
                    {
                        "simulation_id": entry.id,
                        "task_id": entry.task_id,
                        "trial": entry.trial,
                        "domain": domain,
                        "speech_complexity": speech_complexity,
                        "llm": llm,
                        "provider": "text",
                        "reward": reward,
                        "success": reward == 1.0,
                    }
                )
                n_loaded += 1
        else:
            for sim in Results.iter_simulations(results_file):
                reward = sim.reward_info.reward if sim.reward_info else 0.0
                rows.append(
                    {
                        "simulation_id": sim.id,
                        "task_id": sim.task_id,
                        "trial": sim.trial,
                        "domain": domain,
                        "speech_complexity": speech_complexity,
                        "llm": llm,
                        "provider": "text",
                        "reward": reward,
                        "success": reward == 1.0,
                    }
                )
                n_loaded += 1

    _clear_line()

    df = pd.DataFrame(rows)
    if not df.empty:
        for sc in sorted(df["speech_complexity"].unique()):
            sc_df = df[df["speech_complexity"] == sc]
            domains = sorted(sc_df["domain"].unique())
            print(f"  {sc}: {len(sc_df)} sims across {domains}")
    return df


def load_voice_results(
    voice_dirs: list[Path],
    domain_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Load voice results from directories containing results.json files.

    Each directory is searched recursively for results.json files.
    Uses Results.load_metadata() which handles both monolithic JSON and
    directory-based formats automatically.

    Returns a DataFrame with the same schema as the CSV input:
    simulation_id, task_id, trial, domain, speech_complexity, llm, provider,
    reward, success
    """
    rows = []
    for voice_dir in voice_dirs:
        if not voice_dir.exists():
            print(f"  Warning: voice dir not found, skipping: {voice_dir}")
            continue

        results_files = sorted(voice_dir.rglob("results.json"))
        n_files = len(results_files)
        n_loaded = 0
        for file_idx, results_file in enumerate(results_files, 1):
            print(
                f"\r  {voice_dir.name}: {file_idx}/{n_files} files, "
                f"{n_loaded} sims loaded",
                end="",
                flush=True,
            )
            results = Results.load_metadata(results_file)
            info = results.info
            domain = info.environment_info.domain_name
            if domain_filter and domain != domain_filter:
                continue

            llm = info.agent_info.llm or "unknown"
            provider = _parse_provider(llm)
            speech_complexity = info.speech_complexity or "unknown"

            if results.simulation_index:
                for entry in results.simulation_index:
                    reward = entry.reward if entry.reward is not None else 0.0
                    rows.append(
                        {
                            "simulation_id": entry.id,
                            "task_id": entry.task_id,
                            "trial": entry.trial,
                            "domain": domain,
                            "speech_complexity": speech_complexity,
                            "llm": llm,
                            "provider": provider,
                            "reward": reward,
                            "success": reward == 1.0,
                        }
                    )
                    n_loaded += 1
            else:
                for sim in Results.iter_simulations(results_file):
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
                    n_loaded += 1

        _clear_line()
        print(f"  {voice_dir}: {n_loaded} simulations loaded from {n_files} files")

    df = pd.DataFrame(rows)
    if not df.empty:
        for sc in sorted(df["speech_complexity"].unique()):
            sc_df = df[df["speech_complexity"] == sc]
            providers = sorted(sc_df["provider"].unique())
            domains = sorted(sc_df["domain"].unique())
            print(
                f"  {sc}: {len(sc_df)} sims, providers={providers}, domains={domains}"
            )
    return df


def aggregate_per_task(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple simulations per task to a per-task mean success rate."""
    return (
        df.groupby(["domain", "provider", "speech_complexity", "task_id"])
        .agg(
            n_sims=("success", "count"),
            mean_success=("success", "mean"),
        )
        .reset_index()
    )


# =============================================================================
# Statistical Comparisons
# =============================================================================


def _compute_one_comparison(
    cond_a: str,
    cond_b: str,
    raw_a: pd.DataFrame,
    raw_b: pd.DataFrame,
    task_a_series: pd.Series,
    task_b_series: pd.Series,
    domain: str,
    provider: str,
    comparison_type: str,
) -> dict:
    """Compute a single pairwise comparison between two conditions."""
    n_a, s_a = len(raw_a), int(raw_a["success"].sum())
    n_b, s_b = len(raw_b), int(raw_b["success"].sum())
    rate_a, rate_b = s_a / n_a, s_b / n_b

    w_lo_a, w_hi_a = wilson_ci(s_a, n_a)
    w_lo_b, w_hi_b = wilson_ci(s_b, n_b)
    ci_overlap = not (w_lo_a > w_hi_b or w_lo_b > w_hi_a)

    table = [[s_a, n_a - s_a], [s_b, n_b - s_b]]
    _, fisher_p = fisher_exact(table, alternative="two-sided")

    shared_tasks = sorted(set(task_a_series.index) & set(task_b_series.index))
    arr_a = np.array([task_a_series.loc[t] for t in shared_tasks])
    arr_b = np.array([task_b_series.loc[t] for t in shared_tasks])
    perm_p = paired_permutation_test(arr_a, arr_b)

    # Bootstrap CI on the paired delta (B - A), matching delta_pp sign convention
    ci_lo_ab, ci_hi_ab = bootstrap_delta_ci(arr_a, arr_b)
    delta_ci_lo, delta_ci_hi = -ci_hi_ab, -ci_lo_ab

    return {
        "domain": domain,
        "provider": provider,
        "comparison_type": comparison_type,
        "condition_a": cond_a,
        "condition_b": cond_b,
        "condition_a_display": COMPLEXITY_DISPLAY.get(cond_a, cond_a),
        "condition_b_display": COMPLEXITY_DISPLAY.get(cond_b, cond_b),
        "n_sims_a": n_a,
        "successes_a": s_a,
        "rate_a": rate_a,
        "wilson_ci_a": f"[{w_lo_a:.1%}, {w_hi_a:.1%}]",
        "n_sims_b": n_b,
        "successes_b": s_b,
        "rate_b": rate_b,
        "wilson_ci_b": f"[{w_lo_b:.1%}, {w_hi_b:.1%}]",
        "delta_pp": (rate_b - rate_a) * 100,
        "delta_ci_lo_pp": delta_ci_lo * 100,
        "delta_ci_hi_pp": delta_ci_hi * 100,
        "wilson_ci_overlap": ci_overlap,
        "fisher_p": fisher_p,
        "n_paired_tasks": len(shared_tasks),
        "permutation_p": perm_p,
    }


def compute_pairwise_statsig(
    df_raw: pd.DataFrame,
    df_task: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute pairwise statistical significance between speech complexity conditions.

    Two types of comparisons:
    1. Voice-only: pairwise voice conditions within each (domain, voice_provider)
    2. Text vs Voice: each text condition vs each voice provider's control,
       paired by task_id
    """
    rows = []
    voice_providers = sorted(p for p in df_raw["provider"].unique() if p != "text")
    text_conditions = sorted(
        c for c in df_raw["speech_complexity"].unique() if c.startswith("text_")
    )
    voice_conditions = [c for c in COMPLEXITY_ORDER if not c.startswith("text_")]

    for domain in sorted(df_raw["domain"].unique()):
        domain_raw = df_raw[df_raw["domain"] == domain]
        domain_task = df_task[df_task["domain"] == domain]

        # 1) Voice-only: pairwise within each (domain, voice_provider)
        for provider in voice_providers:
            sub_raw = domain_raw[domain_raw["provider"] == provider]
            sub_task = domain_task[domain_task["provider"] == provider]

            complexities = sorted(
                sub_raw["speech_complexity"].unique(),
                key=lambda c: COMPLEXITY_ORDER.index(c)
                if c in COMPLEXITY_ORDER
                else 99,
            )
            if len(complexities) < 2:
                continue

            for cond_a, cond_b in combinations(complexities, 2):
                raw_a = sub_raw[sub_raw["speech_complexity"] == cond_a]
                raw_b = sub_raw[sub_raw["speech_complexity"] == cond_b]
                task_a = sub_task[sub_task["speech_complexity"] == cond_a].set_index(
                    "task_id"
                )["mean_success"]
                task_b = sub_task[sub_task["speech_complexity"] == cond_b].set_index(
                    "task_id"
                )["mean_success"]

                rows.append(
                    _compute_one_comparison(
                        cond_a,
                        cond_b,
                        raw_a,
                        raw_b,
                        task_a,
                        task_b,
                        domain,
                        provider,
                        "voice_pairwise",
                    )
                )

        # 2) Text vs Voice: each text condition vs each voice provider's control
        text_raw = domain_raw[domain_raw["provider"] == "text"]
        text_task = domain_task[domain_task["provider"] == "text"]
        if text_raw.empty:
            continue

        for text_cond in text_conditions:
            t_raw = text_raw[text_raw["speech_complexity"] == text_cond]
            if t_raw.empty:
                continue
            t_task = text_task[text_task["speech_complexity"] == text_cond].set_index(
                "task_id"
            )["mean_success"]

            for provider in voice_providers:
                sub_raw = domain_raw[domain_raw["provider"] == provider]
                sub_task = domain_task[domain_task["provider"] == provider]

                for voice_cond in voice_conditions:
                    v_raw = sub_raw[sub_raw["speech_complexity"] == voice_cond]
                    if v_raw.empty:
                        continue
                    v_task = sub_task[
                        sub_task["speech_complexity"] == voice_cond
                    ].set_index("task_id")["mean_success"]

                    rows.append(
                        _compute_one_comparison(
                            text_cond,
                            voice_cond,
                            t_raw,
                            v_raw,
                            t_task,
                            v_task,
                            domain,
                            provider,
                            "text_vs_voice",
                        )
                    )

    # 3) All-domains-combined voice pairwise: pool across domains,
    #    but only for (provider, complexity) combos present in every domain.
    all_domains = sorted(df_raw["domain"].unique())
    if len(all_domains) > 1:
        for provider in voice_providers:
            prov_raw = df_raw[df_raw["provider"] == provider]
            prov_task = df_task[df_task["provider"] == provider]
            domains_for_prov = set(prov_raw["domain"].unique())

            # Only keep complexities that appear in every domain for this provider
            complexities_per_domain = (
                prov_raw.groupby("domain")["speech_complexity"].apply(set).to_dict()
            )
            shared_complexities = set.intersection(*complexities_per_domain.values())
            shared_complexities = sorted(
                shared_complexities,
                key=lambda c: COMPLEXITY_ORDER.index(c)
                if c in COMPLEXITY_ORDER
                else 99,
            )

            if len(shared_complexities) < 2 or len(domains_for_prov) < len(all_domains):
                continue

            # Use (domain, task_id) as composite pairing key
            prov_task = prov_task.copy()
            prov_task["domain_task"] = (
                prov_task["domain"].astype(str) + ":" + prov_task["task_id"].astype(str)
            )

            for cond_a, cond_b in combinations(shared_complexities, 2):
                raw_a = prov_raw[prov_raw["speech_complexity"] == cond_a]
                raw_b = prov_raw[prov_raw["speech_complexity"] == cond_b]
                task_a = prov_task[prov_task["speech_complexity"] == cond_a].set_index(
                    "domain_task"
                )["mean_success"]
                task_b = prov_task[prov_task["speech_complexity"] == cond_b].set_index(
                    "domain_task"
                )["mean_success"]

                rows.append(
                    _compute_one_comparison(
                        cond_a,
                        cond_b,
                        raw_a,
                        raw_b,
                        task_a,
                        task_b,
                        "all",
                        provider,
                        "voice_pairwise",
                    )
                )

        # 4) All-domains-combined text vs voice
        text_raw_all = df_raw[df_raw["provider"] == "text"]
        text_task_all = df_task[df_task["provider"] == "text"]
        if not text_raw_all.empty:
            text_task_all = text_task_all.copy()
            text_task_all["domain_task"] = (
                text_task_all["domain"].astype(str)
                + ":"
                + text_task_all["task_id"].astype(str)
            )

            for text_cond in text_conditions:
                t_raw = text_raw_all[text_raw_all["speech_complexity"] == text_cond]
                if t_raw.empty or set(t_raw["domain"].unique()) != set(all_domains):
                    continue
                t_task = text_task_all[
                    text_task_all["speech_complexity"] == text_cond
                ].set_index("domain_task")["mean_success"]

                for provider in voice_providers:
                    prov_raw = df_raw[df_raw["provider"] == provider]
                    prov_task = df_task[df_task["provider"] == provider].copy()
                    prov_task["domain_task"] = (
                        prov_task["domain"].astype(str)
                        + ":"
                        + prov_task["task_id"].astype(str)
                    )

                    for voice_cond in voice_conditions:
                        v_raw = prov_raw[prov_raw["speech_complexity"] == voice_cond]
                        if v_raw.empty or set(v_raw["domain"].unique()) != set(
                            all_domains
                        ):
                            continue
                        v_task = prov_task[
                            prov_task["speech_complexity"] == voice_cond
                        ].set_index("domain_task")["mean_success"]

                        rows.append(
                            _compute_one_comparison(
                                text_cond,
                                voice_cond,
                                t_raw,
                                v_raw,
                                t_task,
                                v_task,
                                "all",
                                provider,
                                "text_vs_voice",
                            )
                        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Holm-Bonferroni correction within each (domain, provider, comparison_type)
    result["fisher_p_adj"] = np.nan
    result["permutation_p_adj"] = np.nan
    for _, group in result.groupby(["domain", "provider", "comparison_type"]):
        idx = group.index
        result.loc[idx, "fisher_p_adj"] = holm_bonferroni(group["fisher_p"].tolist())
        result.loc[idx, "permutation_p_adj"] = holm_bonferroni(
            group["permutation_p"].tolist()
        )

    result["fisher_sig"] = result["fisher_p_adj"] < alpha
    result["permutation_sig"] = result["permutation_p_adj"] < alpha

    return result


# =============================================================================
# Reporting
# =============================================================================


def print_report(comparisons: pd.DataFrame) -> None:
    """Print formatted console report."""

    for comp_type, type_label in [
        ("voice_pairwise", "VOICE PAIRWISE COMPARISONS"),
        ("text_vs_voice", "TEXT vs VOICE COMPARISONS"),
    ]:
        section = comparisons[comparisons["comparison_type"] == comp_type]
        if section.empty:
            continue

        print("\n" + "=" * 100)
        print(f"  {type_label}")
        print("=" * 100)

        for domain in sorted(section["domain"].unique()):
            dc = section[section["domain"] == domain]
            print(f"\n{'━' * 100}")
            print(f"  DOMAIN: {domain.upper()}")
            print(f"{'━' * 100}")

            for provider in sorted(dc["provider"].unique()):
                pc = dc[dc["provider"] == provider]
                label = (
                    f"Voice provider: {provider.upper()}"
                    if comp_type == "text_vs_voice"
                    else f"Provider: {provider.upper()}"
                )
                print(f"\n  {label}")
                print(f"  {'─' * 94}")

                hdr = (
                    f"  {'Condition A':<22} {'Condition B':<22} "
                    f"{'Rate A':>7} {'Rate B':>7} "
                    f"{'Δ(pp)':>7} {'95% CI Δ':>16}  "
                    f"{'Perm':>10} {'Fisher':>10} {'Sig?':>5}"
                )
                print(hdr)
                print(f"  {'─' * 110}")

                for _, row in pc.iterrows():
                    disp_a = COMPLEXITY_DISPLAY.get(
                        row["condition_a"], row["condition_a"]
                    )
                    disp_b = COMPLEXITY_DISPLAY.get(
                        row["condition_b"], row["condition_b"]
                    )

                    perm_str = _fmt_pval(row["permutation_p_adj"])
                    fish_str = _fmt_pval(row["fisher_p_adj"])
                    sig = "***" if row["permutation_sig"] else ""
                    direction = _delta_direction(
                        row["delta_ci_lo_pp"], row["delta_ci_hi_pp"]
                    )
                    delta_ci = (
                        f"[{row['delta_ci_lo_pp']:+.1f}, "
                        f"{row['delta_ci_hi_pp']:+.1f}] {direction}"
                    )

                    print(
                        f"  {disp_a:<22} {disp_b:<22} "
                        f"{row['rate_a'] * 100:>6.1f}% {row['rate_b'] * 100:>6.1f}% "
                        f"{row['delta_pp']:>+6.1f} {delta_ci:>18}  "
                        f"{perm_str:>10} {fish_str:>10} {sig:>5}"
                    )

                print()


def _fmt_pval(p: float) -> str:
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _fmt_pval_md(p: float) -> str:
    if p < 0.001:
        return f"{p:.1e}"
    if p < 0.01:
        return f"{p:.4f}"
    return f"{p:.3f}"


def _delta_direction(ci_lo: float, ci_hi: float) -> str:
    """Return a directional indicator based on whether the CI excludes zero.

    ↓ = CI entirely negative (B worse than A)
    ↑ = CI entirely positive (B better than A)
    – = CI spans zero (no significant directional effect)
    """
    if ci_hi < 0:
        return "↓"
    if ci_lo > 0:
        return "↑"
    return "–"


def write_markdown_report(comparisons: pd.DataFrame, output_dir: Path) -> None:
    """Write pairwise comparison results as markdown tables."""
    lines: list[str] = []
    lines.append("# Speech Complexity — Pairwise Statistical Significance\n")
    lines.append(
        "Paired permutation test (100k permutations) on per-task mean success "
        "rates, paired by `task_id`. P-values are Holm-Bonferroni corrected "
        "within each (domain, provider) group. "
        "Fisher's exact test (unpaired) shown for reference.\n"
    )

    for comp_type, type_label in [
        ("voice_pairwise", "Voice Pairwise Comparisons"),
        ("text_vs_voice", "Text vs Voice Comparisons"),
    ]:
        section = comparisons[comparisons["comparison_type"] == comp_type]
        if section.empty:
            continue

        lines.append(f"## {type_label}\n")

        for domain in sorted(section["domain"].unique()):
            dc = section[section["domain"] == domain]

            for provider in sorted(dc["provider"].unique()):
                pc = dc[dc["provider"] == provider]
                prov_label = (
                    f"voice provider: {provider.upper()}"
                    if comp_type == "text_vs_voice"
                    else provider.upper()
                )
                lines.append(f"### {domain.capitalize()} — {prov_label}\n")
                lines.append(
                    "| Condition A | Condition B | Rate A | Rate B "
                    "| Δ (pp) | 95% CI Δ (pp) "
                    "| Perm p (adj) | Fisher p (adj) | Sig |"
                )
                lines.append("|---|---|---:|---:|---:|---:|---:|---:|:---:|")

                for _, row in pc.iterrows():
                    disp_a = COMPLEXITY_DISPLAY.get(
                        row["condition_a"], row["condition_a"]
                    )
                    disp_b = COMPLEXITY_DISPLAY.get(
                        row["condition_b"], row["condition_b"]
                    )
                    sig = "\\*\\*\\*" if row["permutation_sig"] else ""
                    direction = _delta_direction(
                        row["delta_ci_lo_pp"], row["delta_ci_hi_pp"]
                    )
                    delta_ci = (
                        f"[{row['delta_ci_lo_pp']:+.1f}, "
                        f"{row['delta_ci_hi_pp']:+.1f}] {direction}"
                    )
                    lines.append(
                        f"| {disp_a} | {disp_b} "
                        f"| {row['rate_a'] * 100:.1f}% "
                        f"| {row['rate_b'] * 100:.1f}% "
                        f"| {row['delta_pp']:+.1f} "
                        f"| {delta_ci} "
                        f"| {_fmt_pval_md(row['permutation_p_adj'])} "
                        f"| {_fmt_pval_md(row['fisher_p_adj'])} "
                        f"| {sig} |"
                    )

                lines.append("")

    md_path = output_dir / "speech_complexity_pairwise_statsig.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved markdown report: {md_path}")


# -- Headline comparisons: text vs voice + clean vs realistic ----------------

HEADLINE_COMPARISONS = [
    ("Text (NR) → Clean", "text_non_reasoning", "control"),
    ("Text (NR) → Realistic", "text_non_reasoning", "regular"),
    ("Text (R) → Clean", "text_reasoning", "control"),
    ("Text (R) → Realistic", "text_reasoning", "regular"),
    ("Clean → Realistic", "control", "regular"),
]


def _find_comparison_row(
    comparisons: pd.DataFrame,
    domain: str,
    provider: str,
    cond_a: str,
    cond_b: str,
) -> Optional[pd.Series]:
    """Look up a single comparison row, checking both comparison_type columns."""
    for comp_type in ("text_vs_voice", "voice_pairwise"):
        mask = (
            (comparisons["domain"] == domain)
            & (comparisons["provider"] == provider)
            & (comparisons["comparison_type"] == comp_type)
            & (comparisons["condition_a"] == cond_a)
            & (comparisons["condition_b"] == cond_b)
        )
        rows = comparisons[mask]
        if not rows.empty:
            return rows.iloc[0]
    return None


def write_text_vs_voice_report(
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write a focused text-vs-voice markdown report.

    Format: per-domain tables with rows grouped by headline comparison
    (Text→Clean, Clean→Realistic), provider breakdown within each group.
    Includes an "All Domains" section when domain="all" rows exist.
    """
    lines: list[str] = []
    lines.append("# Pairwise Statistical Significance — All Domains\n")

    # Collect domains in display order: real domains first, then "all"
    all_domain_vals = sorted(comparisons["domain"].unique())
    real_domains = [d for d in all_domain_vals if d != "all"]
    domains_ordered = real_domains + (["all"] if "all" in all_domain_vals else [])

    voice_providers = sorted(p for p in comparisons["provider"].unique() if p != "text")

    for domain in domains_ordered:
        domain_label = (
            "All Domains (Combined)" if domain == "all" else domain.capitalize()
        )
        lines.append(f"## {domain_label}\n")
        lines.append(
            "| Comparison | Provider | Rate A | Rate B | ∆ (pp) | p (adj) | Sig |"
        )
        lines.append("|---|---|---:|---:|---:|---:|:---:|")

        for comp_label, cond_a, cond_b in HEADLINE_COMPARISONS:
            first_in_group = True
            for provider in voice_providers:
                row = _find_comparison_row(
                    comparisons, domain, provider, cond_a, cond_b
                )
                if row is None:
                    continue

                label_cell = comp_label if first_in_group else ""
                first_in_group = False
                prov_display = provider.capitalize()
                sig = "\\*" if row["permutation_sig"] else ""
                p_str = (
                    "<0.001"
                    if row["permutation_p_adj"] < 0.001
                    else f"{row['permutation_p_adj']:.3f}"
                )

                lines.append(
                    f"| {label_cell} | {prov_display} "
                    f"| {row['rate_a'] * 100:.1f}% "
                    f"| {row['rate_b'] * 100:.1f}% "
                    f"| {row['delta_pp']:+.1f} "
                    f"| {p_str} "
                    f"| {sig} |"
                )

        lines.append("")

    md_path = output_dir / "pairwise_statsig_all_domains.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved text-vs-voice report: {md_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Pairwise statistical significance between speech complexity conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Path to pass_k_by_domain_raw.csv (voice results).",
    )
    input_group.add_argument(
        "--voice-dirs",
        type=str,
        nargs="+",
        help="One or more directories containing voice results.json files "
        "(searched recursively).",
    )
    parser.add_argument(
        "--text-dir",
        type=str,
        default=None,
        help="Directory containing text model results.json files "
        "(e.g., data/exp/final_text_results_lite).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. Required when using --voice-dirs.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter to a specific domain.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Filter to a specific voice provider (does not filter text).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold (default: 0.05).",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.input:
        output_dir = Path(args.input).parent
    else:
        parser.error("--output-dir is required when using --voice-dirs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load voice results
    if args.input:
        input_path = Path(args.input)
        print(f"Loading voice data from {input_path}...")
        df = pd.read_csv(input_path)
        print(f"  {len(df)} voice simulations loaded")
    else:
        voice_dirs = [Path(d) for d in args.voice_dirs]
        print(f"Loading voice data from {len(voice_dirs)} directories...")
        df = load_voice_results(voice_dirs, domain_filter=args.domain)
        print(f"  {len(df)} voice simulations loaded")

    if args.domain:
        df = df[df["domain"] == args.domain]
        print(f"  Filtered to domain={args.domain}: {len(df)} simulations")
    if args.provider:
        df = df[df["provider"] == args.provider]
        print(f"  Filtered to provider={args.provider}: {len(df)} simulations")

    # Load text results
    if args.text_dir:
        text_dir = Path(args.text_dir)
        print(f"\nLoading text baselines from {text_dir}...")
        df_text = load_text_results(text_dir, domain_filter=args.domain)
        if not df_text.empty:
            df = pd.concat([df, df_text], ignore_index=True)
            print(f"  Combined: {len(df)} total simulations")

            unified_csv = output_dir / "unified_voice_text_raw.csv"
            df.to_csv(unified_csv, index=False)
            print(f"  Saved unified CSV: {unified_csv}")

    if df.empty:
        print("No simulations loaded. Check your input paths.")
        return

    print(f"\n  Domains: {sorted(df['domain'].unique())}")
    print(f"  Providers: {sorted(df['provider'].unique())}")
    print(f"  Speech complexities: {sorted(df['speech_complexity'].unique())}")

    df_task = aggregate_per_task(df)

    comparisons = compute_pairwise_statsig(df, df_task, alpha=args.alpha)
    if comparisons.empty:
        print("No comparisons to make.")
        return

    csv_path = output_dir / "speech_complexity_pairwise_statsig.csv"
    comparisons.to_csv(csv_path, index=False)
    print(f"\nSaved pairwise comparisons: {csv_path}")

    print_report(comparisons)
    write_markdown_report(comparisons, output_dir)
    write_text_vs_voice_report(comparisons, output_dir)

    # Summary
    sig_perm = comparisons[comparisons["permutation_sig"]]
    sig_fish = comparisons[comparisons["fisher_sig"]]
    total = len(comparisons)

    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    print(f"  Total comparisons: {total}")
    print(
        f"  Significant (Paired permutation, Holm-corrected α={args.alpha}): "
        f"{len(sig_perm)}/{total}"
    )
    print(
        f"  Significant (Fisher, Holm-corrected α={args.alpha}): "
        f"{len(sig_fish)}/{total}"
    )

    for comp_type, label in [
        ("voice_pairwise", "Voice pairwise"),
        ("text_vs_voice", "Text vs Voice"),
    ]:
        sig_section = sig_perm[sig_perm["comparison_type"] == comp_type]
        if sig_section.empty:
            continue
        print(f"\n  {label} — significant results:")
        for _, row in sig_section.iterrows():
            print(
                f"    {row['domain']}/{row['provider']}: "
                f"{row['condition_a_display']} vs {row['condition_b_display']}  "
                f"Δ={row['delta_pp']:+.1f}pp  "
                f"p_adj={_fmt_pval(row['permutation_p_adj'])}"
            )


if __name__ == "__main__":
    main()
