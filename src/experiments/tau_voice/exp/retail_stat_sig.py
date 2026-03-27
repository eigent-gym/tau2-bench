#!/usr/bin/env python3
"""Compute statistical significance metrics (mean, std, SEM, 95% CI) for experiment results."""

import argparse
import math


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for proportions (95% CI with z=1.96)."""
    n = total
    p = successes / n

    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - spread, center + spread


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def report_stats(name: str, values: list[int], n_per_trial: int) -> dict:
    """Compute and report mean, std, SEM, 95% CI for a set of trial results."""
    rates = [v / n_per_trial for v in values]
    m = mean(rates)
    s = std(rates)
    sem = s / math.sqrt(len(rates)) if len(rates) > 1 else 0
    ci_95 = 1.96 * sem

    # Pooled Wilson CI (aggregate all trials)
    total_success = sum(values)
    total_n = n_per_trial * len(values)
    wilson_lo, wilson_hi = wilson_ci(total_success, total_n)

    result = {
        "name": name,
        "trials": values,
        "n_per_trial": n_per_trial,
        "mean": m,
        "std": s,
        "sem": sem,
        "ci_95": ci_95,
        "wilson_lo": wilson_lo,
        "wilson_hi": wilson_hi,
    }

    print(f"{name}")
    print(f"  Trials: {[f'{v}/{n_per_trial}' for v in values]}")
    print(f"  Mean: {m * 100:.1f}%  Std: {s * 100:.1f}%  SEM: {sem * 100:.1f}%")
    print(
        f"  Mean ± 95% CI: {m * 100:.1f}% ± {ci_95 * 100:.1f}%  →  [{(m - ci_95) * 100:.1f}%, {(m + ci_95) * 100:.1f}%]"
    )
    print(f"  Pooled Wilson 95% CI: [{wilson_lo * 100:.1f}%, {wilson_hi * 100:.1f}%]")
    print()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistics for experiment results"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all predefined experiments"
    )
    _args = parser.parse_args()

    print("=" * 70)
    print("TEXT MODELS (from leaderboard, 3 trials each)")
    print("=" * 70)

    # RETAIL TEXT
    print("\n>> RETAIL (n=114 per trial)")
    report_stats("GPT-4.1 (text)", [84, 81, 86], 114)
    report_stats("GPT-5 (text)", [93, 93, 94], 114)

    # TELECOM TEXT
    print("\n>> TELECOM (n=114 per trial)")
    report_stats("GPT-4.1 (text)", [42, 37, 40], 114)
    report_stats("GPT-5 (text)", [110, 112, 108], 114)

    # AIRLINE TEXT
    print("\n>> AIRLINE (n=50 per trial)")
    report_stats("GPT-4.1 (text)", [29, 27, 30], 50)
    report_stats("GPT-5 (text)", [31, 31, 31], 50)

    print("\n" + "=" * 70)
    print("AUDIO MODELS - RETAIL (3 providers × 2 settings, 3 runs each)")
    print("=" * 70)

    # RETAIL AUDIO - CONTROL
    print("\n>> RETAIL CONTROL (n=114 per trial)")
    report_stats("Gemini (ctrl)", [45, 41, 45], 114)
    report_stats("OpenAI (ctrl)", [45, 42, 47], 114)
    report_stats("XAI (ctrl)", [48, 37, 39], 114)

    # RETAIL AUDIO - REGULAR
    print("\n>> RETAIL REGULAR (n=114 per trial)")
    report_stats("Gemini (reg)", [32, 26, 31], 114)
    report_stats("OpenAI (reg)", [18, 12, 11], 114)
    # Note: XAI Run3 only has 4 samples, so excluding it
    report_stats("XAI (reg, 2 runs)", [23, 25], 114)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    data = [
        # Text models
        ("Text", "GPT-4.1", "Retail", [84, 81, 86], 114),
        ("Text", "GPT-5", "Retail", [93, 93, 94], 114),
        ("Text", "GPT-4.1", "Telecom", [42, 37, 40], 114),
        ("Text", "GPT-5", "Telecom", [110, 112, 108], 114),
        ("Text", "GPT-4.1", "Airline", [29, 27, 30], 50),
        ("Text", "GPT-5", "Airline", [31, 31, 31], 50),
        # Audio models - Retail
        ("Audio ctrl", "Gemini", "Retail", [45, 41, 45], 114),
        ("Audio ctrl", "OpenAI", "Retail", [45, 42, 47], 114),
        ("Audio ctrl", "XAI", "Retail", [48, 37, 39], 114),
        ("Audio reg", "Gemini", "Retail", [32, 26, 31], 114),
        ("Audio reg", "OpenAI", "Retail", [18, 12, 11], 114),
        ("Audio reg", "XAI", "Retail", [23, 25, 22], 114),
    ]

    print(
        f"\n{'Type':<12} {'Model':<10} {'Domain':<10} {'Mean ± Std':<18} {'Mean ± 95% CI':<18}"
    )
    print("-" * 70)

    for typ, model, domain, values, n in data:
        rates = [v / n for v in values]
        m = mean(rates)
        s = std(rates)
        sem = s / math.sqrt(len(rates)) if len(rates) > 1 else 0
        ci_95 = 1.96 * sem
        print(
            f"{typ:<12} {model:<10} {domain:<10} {m * 100:.1f}% ± {s * 100:.1f}%{'':<6} {m * 100:.1f}% ± {ci_95 * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
