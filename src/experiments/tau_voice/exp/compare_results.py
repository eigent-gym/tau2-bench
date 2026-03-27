#!/usr/bin/env python3
"""
Compare two sets of paper results from tau-voice experiments.

Reads the CSV tables produced by paper_outputs.py from two directories (labeled
"old" and "new") and prints side-by-side comparison tables highlighting the
differences.

Usage:
    python -m experiments.tau_voice.exp.compare_results \
        --old papers/tau-voice/results \
        --new data/exp/new_results_analysis/paper

    # Only compare specific tables:
    python -m experiments.tau_voice.exp.compare_results \
        --old papers/tau-voice/results \
        --new data/exp/new_results_analysis/paper \
        --tables main_results ablation

    # Save output to file:
    python -m experiments.tau_voice.exp.compare_results \
        --old papers/tau-voice/results \
        --new data/exp/new_results_analysis/paper \
        --output comparison_report.txt
"""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Table registry ────────────────────────────────────────────────────────────
# Each entry maps a short name to (csv_filename, key_columns, value_columns,
# display_name). key_columns are used for joining old/new; value_columns are
# the numeric columns to compare.

TABLE_REGISTRY: dict[str, dict] = {
    "main_results": {
        "file": "main_results_table.csv",
        "keys": ["domain", "provider"],
        "values": ["control", "regular", "delta"],
        "title": "Main Results (Pass@1)",
        "pct": True,
    },
    "core_metrics": {
        "file": "core_metrics_table.csv",
        "keys": ["domain", "provider", "complexity"],
        "values": ["response_latency", "response_rate", "agent_interruption_rate"],
        "title": "Core Voice Metrics",
        "pct": False,
    },
    "voice_quality": {
        "file": "voice_quality_table.csv",
        "keys": ["domain", "provider"],
        "values": [
            "response_rate",
            "response_latency",
            "yield_rate",
            "yield_latency",
            "backchannel_error_rate",
            "vocal_tic_error_rate",
            "non_directed_error_rate",
        ],
        "title": "Voice Quality (Regular condition)",
        "pct": False,
    },
    "voice_quality_aggregated": {
        "file": "voice_quality_aggregated_table.csv",
        "keys": ["domain", "provider"],
        "values": ["latency", "responsiveness", "interrupt", "selectivity"],
        "title": "Voice Quality Aggregated",
        "pct": False,
    },
    "voice_vs_text": {
        "file": "voice_vs_text_table.csv",
        "keys": ["domain", "provider"],
        "values": ["text_sota", "voice", "delta"],
        "title": "Voice vs Text",
        "pct": True,
    },
    "combined_comparison": {
        "file": "combined_comparison_table.csv",
        "keys": ["domain", "provider"],
        "values": [
            "text_sota",
            "text_nonthinking",
            "control",
            "control_delta",
            "regular",
            "regular_delta",
        ],
        "title": "Combined Comparison (Text / Control / Realistic)",
        "pct": True,
    },
    "ablation": {
        "file": "ablation_table.csv",
        "keys": ["condition"],
        "values": [],  # dynamic — all columns except "condition"
        "title": "Ablation (Retail, all conditions)",
        "pct": True,
    },
    "ablation_single": {
        "file": "ablation_table_single.csv",
        "keys": ["condition"],
        "values": [],  # dynamic
        "title": "Ablation (Retail, single-factor only)",
        "pct": True,
    },
}


# ── Formatting helpers ────────────────────────────────────────────────────────

RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"


def _color(text: str, code: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{code}{text}{RESET}"


def _fmt_val(v: float | None, pct: bool) -> str:
    if v is None or pd.isna(v):
        return "—"
    if pct:
        return f"{v * 100:.1f}%"
    return f"{v:.3f}"


def _fmt_diff(
    old_v: float | None, new_v: float | None, pct: bool, use_color: bool
) -> str:
    if old_v is None or new_v is None or pd.isna(old_v) or pd.isna(new_v):
        return ""
    diff = new_v - old_v
    if abs(diff) < 1e-9:
        return _color("=", DIM, use_color)
    sign = "+" if diff > 0 else ""
    if pct:
        text = f"{sign}{diff * 100:.1f}pp"
    else:
        text = f"{sign}{diff:.3f}"
    color = GREEN if diff > 0 else RED
    return _color(text, color, use_color)


def _section_header(title: str, use_color: bool) -> str:
    width = 80
    line = "═" * width
    header = f"\n{line}\n  {title}\n{line}"
    return _color(header, BOLD + CYAN, use_color)


# ── CSV loading ───────────────────────────────────────────────────────────────


def _load_csv(directory: Path, filename: str) -> Optional[pd.DataFrame]:
    path = directory / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


# ── Comparison logic ──────────────────────────────────────────────────────────


def _compare_table(
    old_df: Optional[pd.DataFrame],
    new_df: Optional[pd.DataFrame],
    spec: dict,
    use_color: bool,
) -> str:
    """Compare a single table and return formatted output."""
    buf = StringIO()
    title = spec["title"]
    keys = spec["keys"]
    pct = spec["pct"]

    buf.write(_section_header(title, use_color) + "\n")

    if old_df is None and new_df is None:
        buf.write("  (not found in either directory)\n")
        return buf.getvalue()
    if old_df is None:
        buf.write("  ⚠  Only in NEW results\n")
        buf.write(new_df.to_string(index=False) + "\n")
        return buf.getvalue()
    if new_df is None:
        buf.write("  ⚠  Only in OLD results\n")
        buf.write(old_df.to_string(index=False) + "\n")
        return buf.getvalue()

    # Determine value columns
    value_cols = spec["values"]
    if not value_cols:
        # Dynamic: everything except key columns
        all_cols = set(old_df.columns) | set(new_df.columns)
        value_cols = sorted(all_cols - set(keys))

    # Outer-join on keys
    merged = pd.merge(old_df, new_df, on=keys, how="outer", suffixes=("_old", "_new"))

    # ── Structural differences ────────────────────────────────────────────
    old_only_cols = set(old_df.columns) - set(new_df.columns) - set(keys)
    new_only_cols = set(new_df.columns) - set(old_df.columns) - set(keys)
    if old_only_cols:
        buf.write(
            _color(
                f"  Columns only in OLD: {sorted(old_only_cols)}\n", YELLOW, use_color
            )
        )
    if new_only_cols:
        buf.write(
            _color(
                f"  Columns only in NEW: {sorted(new_only_cols)}\n", YELLOW, use_color
            )
        )

    # Identify rows only in old / only in new
    old_key_set = set(old_df[keys].apply(tuple, axis=1))
    new_key_set = set(new_df[keys].apply(tuple, axis=1))
    old_only_rows = old_key_set - new_key_set
    new_only_rows = new_key_set - old_key_set
    if old_only_rows:
        buf.write(
            _color(f"  Rows only in OLD: {sorted(old_only_rows)}\n", YELLOW, use_color)
        )
    if new_only_rows:
        buf.write(
            _color(f"  Rows only in NEW: {sorted(new_only_rows)}\n", YELLOW, use_color)
        )

    buf.write("\n")

    # ── Build comparison table ────────────────────────────────────────────
    # Determine which value columns exist in each dataframe.
    # pd.merge only adds suffixes to columns present in BOTH frames;
    # columns unique to one side keep their original name.
    old_value_cols = set(old_df.columns) - set(keys)
    new_value_cols = set(new_df.columns) - set(keys)

    common_value_cols = [
        c for c in value_cols if c in old_value_cols or c in new_value_cols
    ]

    # Header
    key_headers = [k.upper() for k in keys]
    col_headers = []
    for c in common_value_cols:
        col_headers.extend([f"{c}(old)", f"{c}(new)", "Δ"])

    # Compute column widths
    header_row = key_headers + col_headers
    col_widths = [max(len(h), 8) for h in header_row]

    # Gather rows
    data_rows = []
    for _, row in merged.iterrows():
        key_vals = [str(row.get(k, "—")) for k in keys]
        metric_vals = []
        for c in common_value_cols:
            in_old = c in old_value_cols
            in_new = c in new_value_cols
            if in_old and in_new:
                # Column in both → suffixed by merge
                old_v = row.get(f"{c}_old")
                new_v = row.get(f"{c}_new")
            elif in_old:
                # Only in old → unsuffixed
                old_v = row.get(c)
                new_v = None
            else:
                # Only in new → unsuffixed
                old_v = None
                new_v = row.get(c)

            old_v = old_v if old_v is not None and not pd.isna(old_v) else None
            new_v = new_v if new_v is not None and not pd.isna(new_v) else None
            metric_vals.append(_fmt_val(old_v, pct))
            metric_vals.append(_fmt_val(new_v, pct))
            metric_vals.append(_fmt_diff(old_v, new_v, pct, use_color))
        data_rows.append(key_vals + metric_vals)

    # Recompute widths accounting for data (strip ANSI for width calc)
    import re

    ansi_re = re.compile(r"\033\[[0-9;]*m")
    for dr in data_rows:
        for i, cell in enumerate(dr):
            visible_len = len(ansi_re.sub("", cell))
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], visible_len)
            else:
                col_widths.append(visible_len)

    # Ensure col_widths matches header
    while len(col_widths) < len(header_row):
        col_widths.append(8)

    def _fmt_row(cells: list[str], widths: list[int]) -> str:
        parts = []
        for cell, w in zip(cells, widths):
            visible_len = len(ansi_re.sub("", cell))
            padding = w - visible_len
            parts.append(cell + " " * max(padding, 0))
        return "  " + "  ".join(parts)

    # Print header
    buf.write(_fmt_row(header_row, col_widths) + "\n")
    buf.write("  " + "  ".join("─" * w for w in col_widths) + "\n")

    # Print data
    for dr in data_rows:
        buf.write(_fmt_row(dr, col_widths) + "\n")

    return buf.getvalue()


def _generate_summary(
    old_dir: Path,
    new_dir: Path,
    use_color: bool,
) -> str:
    """Generate a structural summary of the two directories."""
    buf = StringIO()
    buf.write(_section_header("Directory Summary", use_color) + "\n")

    old_csvs = sorted(f.name for f in old_dir.glob("*.csv"))
    new_csvs = sorted(f.name for f in new_dir.glob("*.csv"))

    buf.write(f"  OLD: {old_dir}\n")
    buf.write(f"       {len(old_csvs)} CSV files: {', '.join(old_csvs)}\n\n")
    buf.write(f"  NEW: {new_dir}\n")
    buf.write(f"       {len(new_csvs)} CSV files: {', '.join(new_csvs)}\n\n")

    only_old = sorted(set(old_csvs) - set(new_csvs))
    only_new = sorted(set(new_csvs) - set(old_csvs))
    common = sorted(set(old_csvs) & set(new_csvs))

    if only_old:
        buf.write(_color(f"  CSVs only in OLD: {only_old}\n", YELLOW, use_color))
    if only_new:
        buf.write(_color(f"  CSVs only in NEW: {only_new}\n", YELLOW, use_color))
    buf.write(f"  Common CSVs: {common}\n")

    # Task counts
    for label, d in [("OLD", old_dir), ("NEW", new_dir)]:
        main_csv = d / "main_results_table.csv"
        if main_csv.exists():
            df = pd.read_csv(main_csv)
            providers = sorted(df["provider"].unique())
            domains = sorted(df["domain"].unique())
            n = (
                df.loc[df["domain"] == "All", "n_tasks"].iloc[0]
                if "All" in df["domain"].values
                else "?"
            )
            buf.write(
                f"  {label} coverage: providers={providers}, domains={domains}, n_tasks(All)={n}\n"
            )

    return buf.getvalue()


def compare_results(
    old_dir: Path,
    new_dir: Path,
    tables: Optional[list[str]] = None,
    use_color: bool = True,
) -> str:
    """
    Compare two result directories and return a formatted report.

    Args:
        old_dir: Path to the old results directory (paper/ CSVs).
        new_dir: Path to the new results directory (paper/ CSVs).
        tables: Optional list of table names to compare (keys in TABLE_REGISTRY).
                If None, compares all tables found.
        use_color: Whether to use ANSI color codes.

    Returns:
        Formatted comparison report string.
    """
    parts: list[str] = []

    parts.append(_generate_summary(old_dir, new_dir, use_color))

    table_names = tables if tables else list(TABLE_REGISTRY.keys())
    for name in table_names:
        if name not in TABLE_REGISTRY:
            parts.append(f"\n⚠  Unknown table: {name}\n")
            continue

        spec = TABLE_REGISTRY[name]
        old_df = _load_csv(old_dir, spec["file"])
        new_df = _load_csv(new_dir, spec["file"])

        # Skip if neither exists (unless explicitly requested)
        if old_df is None and new_df is None and tables is None:
            continue

        parts.append(_compare_table(old_df, new_df, spec, use_color))

    return "\n".join(parts)


# ── CLI ───────────────────────────────────────────────────────────────────────


def get_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two sets of tau-voice paper results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available tables:\n"
            + "\n".join(f"  {k:25s} {v['title']}" for k, v in TABLE_REGISTRY.items())
        ),
    )
    parser.add_argument(
        "--old",
        type=str,
        required=True,
        help="Path to OLD results directory containing CSV files.",
    )
    parser.add_argument(
        "--new",
        type=str,
        required=True,
        help="Path to NEW results directory containing CSV files.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=None,
        metavar="TABLE",
        help="Specific tables to compare (default: all).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write output to file instead of stdout.",
    )
    return parser


def main():
    parser = get_cli_parser()
    args = parser.parse_args()

    old_dir = Path(args.old)
    new_dir = Path(args.new)

    if not old_dir.is_dir():
        print(f"Error: OLD directory does not exist: {old_dir}", file=sys.stderr)
        sys.exit(1)
    if not new_dir.is_dir():
        print(f"Error: NEW directory does not exist: {new_dir}", file=sys.stderr)
        sys.exit(1)

    # Disable color when writing to file
    use_color = not args.no_color and args.output is None

    report = compare_results(old_dir, new_dir, args.tables, use_color=use_color)

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
