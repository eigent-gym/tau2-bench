#!/usr/bin/env python3
"""
Qualitative Analysis for Tau Voice Paper.

Loads SimulationNote JSONs from data/notes/ directories and generates
paper-ready CSVs and LaTeX tables for qualitative analysis.

Usage:
    python -m experiments.tau_voice.exp.qualitative_analysis

Directories:
    - data/notes/text_vs_audio_control: Notes comparing text vs audio control
    - data/notes/audio_clean_vs_regular: Notes comparing clean vs regular audio
"""

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from tau2.data_model.simulation import SimulationNote
from tau2.utils.utils import DATA_DIR

# =============================================================================
# Constants
# =============================================================================

NOTES_DIR = DATA_DIR / "notes"

# Analysis directories
ANALYSIS_DIRS = {
    "text_vs_audio_control": NOTES_DIR / "text_vs_audio_control",
    "audio_clean_vs_regular": NOTES_DIR / "audio_clean_vs_regular",
}

# Provider extraction patterns
PROVIDER_PATTERNS = {
    "xai": r"_xai[/_]|xai$",
    "openai": r"_openai[/_]|openai$",
    "gemini": r"_gemini[/_]|gemini$|google",
    "amazon": r"_amazon[/_]|amazon$|nova",
}


def extract_provider_from_source(source_file: str) -> str:
    """Extract provider name from source_results_file path."""
    source_lower = source_file.lower()
    for provider, pattern in PROVIDER_PATTERNS.items():
        if re.search(pattern, source_lower):
            return provider
    return "unknown"


def load_simulation_notes(notes_dir: Path) -> list[SimulationNote]:
    """Load all SimulationNote JSONs from a directory."""
    notes = []

    for json_file in notes_dir.glob("*_note.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            note = SimulationNote(**data)
            notes.append(note)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    return notes


def notes_to_dataframe(notes: list[SimulationNote]) -> pd.DataFrame:
    """Convert SimulationNote list to DataFrame using only JSON fields."""

    notes_data = []
    for note in notes:
        provider = extract_provider_from_source(note.source_results_file or "")
        # Handle enums - get value if it's an enum, otherwise use as-is
        error_source = note.error_source
        if hasattr(error_source, "value"):
            error_source = error_source.value

        error_type = note.error_type
        if hasattr(error_type, "value"):
            error_type = error_type.value

        notes_data.append(
            {
                "note_id": note.id,
                "task_id": note.task_id,
                "simulation_id": note.simulation_id,
                "trial": note.trial,
                "provider": provider,
                "author": note.author_email.split("@")[0] if note.author_email else "",
                "created_at": note.created_at,
                "source_file": note.source_results_file,
                "simulation_file": note.simulation_file,
                "error_source": error_source,
                "error_type": error_type,
                "note": note.note,
            }
        )

    return pd.DataFrame(notes_data)


def generate_qualitative_csv(
    analysis_name: str,
    notes_dir: Path,
    output_dir: Path,
) -> Optional[Path]:
    """Generate CSV for a single qualitative analysis directory."""

    logger.info(f"Processing {analysis_name}...")

    # Load notes
    notes = load_simulation_notes(notes_dir)
    if not notes:
        logger.warning(f"No notes found in {notes_dir}")
        return None

    logger.info(f"  Loaded {len(notes)} simulation notes")

    # Convert to DataFrame (using only JSON fields)
    df = notes_to_dataframe(notes)

    if df.empty:
        logger.warning(f"  No data to save")
        return None

    # Sort by task_id (numeric)
    df["task_id_num"] = df["task_id"].astype(int)
    df = df.sort_values("task_id_num").drop(columns=["task_id_num"])

    # Save CSV with summary columns only
    # Columns: Task ID, Provider, Reviewer, Error source, Error type, Notes
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_df = df[
        ["task_id", "provider", "author", "error_source", "error_type", "note"]
    ].copy()
    csv_df.columns = [
        "Task ID",
        "Provider",
        "Reviewer",
        "Error source",
        "Error type",
        "Notes",
    ]
    csv_path = output_dir / f"qualitative_{analysis_name}.csv"
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")

    return csv_path


def generate_error_source_aggregate_table(
    df: pd.DataFrame,
    analysis_name: str,
    output_dir: Path,
) -> Optional[Path]:
    """Generate aggregate LaTeX table for error source distribution (for main paper)."""

    if df.empty or "Error source" not in df.columns:
        return None

    # Count error sources (aggregate, not by provider)
    source_counts = df["Error source"].value_counts()
    total = len(df)

    # Generate LaTeX
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Error Source} & \textbf{Count} & \textbf{\%} \\",
        r"\midrule",
    ]

    for source in ["agent", "user", "system"]:
        if source in source_counts.index:
            count = source_counts[source]
            pct = int(round(count / total * 100))
            lines.append(f"{source.capitalize()} & {count} & {pct}\\% \\\\")

    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Total}} & \\textbf{{{total}}} & \\textbf{{100\\%}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = output_dir / f"error_source_aggregate_{analysis_name}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {tex_path}")

    return tex_path


def generate_error_type_aggregate_table(
    df: pd.DataFrame,
    analysis_name: str,
    output_dir: Path,
) -> Optional[Path]:
    """Generate aggregate LaTeX table for error type distribution (for main paper)."""

    if df.empty or "Error type" not in df.columns:
        return None

    # Count error types (aggregate)
    type_counts = df["Error type"].fillna("unknown").value_counts()
    total = len(df)

    # Generate LaTeX
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Error Type} & \textbf{Count} & \textbf{\%} \\",
        r"\midrule",
    ]

    type_display = {
        "logical": "Logical",
        "transcription": "Transcription",
        "vad": "VAD",
        "unresponsive": "Unresponsive",
        "hallucination": "Hallucination",
        "early_termination": "Early Termination",
    }

    for error_type, count in type_counts.items():
        display = type_display.get(error_type, error_type.replace("_", " ").title())
        pct = int(round(count / total * 100))
        lines.append(f"{display} & {count} & {pct}\\% \\\\")

    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Total}} & \\textbf{{{total}}} & \\textbf{{100\\%}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = output_dir / f"error_type_aggregate_{analysis_name}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {tex_path}")

    return tex_path


def generate_full_notes_table(
    df: pd.DataFrame,
    analysis_name: str,
    output_dir: Path,
) -> Optional[Path]:
    """Generate full LaTeX table with all 20 rows (for appendix), without notes column."""

    if df.empty:
        return None

    # Escape LaTeX special characters
    def escape_latex(text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text)
        # Escape special LaTeX characters
        replacements = [
            ("\\", r"\textbackslash{}"),
            ("&", r"\&"),
            ("%", r"\%"),
            ("$", r"\$"),
            ("#", r"\#"),
            ("_", r"\_"),
            ("{", r"\{"),
            ("}", r"\}"),
            ("~", r"\textasciitilde{}"),
            ("^", r"\textasciicircum{}"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    # Generate compact LaTeX table (no notes column for side-by-side layout)
    lines = [
        r"\begin{tabular}{clll}",
        r"\toprule",
        r"\textbf{Task} & \textbf{Provider} & \textbf{Source} & \textbf{Type} \\",
        r"\midrule",
    ]

    provider_display = {
        "xai": "xAI",
        "openai": "OpenAI",
        "gemini": "Google",
        "amazon": "Amazon",
    }

    for _, row in df.iterrows():
        task_id = row["Task ID"]
        provider = provider_display.get(row["Provider"], row["Provider"])
        source = (
            escape_latex(row["Error source"]) if pd.notna(row["Error source"]) else ""
        )
        error_type = (
            escape_latex(row["Error type"]) if pd.notna(row["Error type"]) else ""
        )

        lines.append(f"{task_id} & {provider} & {source} & {error_type} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = output_dir / f"full_notes_{analysis_name}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {tex_path}")

    return tex_path


def generate_error_breakdown_table(
    df: pd.DataFrame,
    analysis_name: str,
    output_dir: Path,
) -> Optional[Path]:
    """Generate LaTeX table summarizing error types by provider."""

    if df.empty or "Error source" not in df.columns:
        return None

    # Count error sources by provider
    error_counts = df.groupby(["Provider", "Error source"]).size().unstack(fill_value=0)

    # Ensure we have both agent and user columns
    for col in ["agent", "user"]:
        if col not in error_counts.columns:
            error_counts[col] = 0

    # Add total column
    error_counts["total"] = error_counts.sum(axis=1)

    # Calculate percentages
    error_counts["agent_pct"] = (
        (error_counts["agent"] / error_counts["total"] * 100).round(0).astype(int)
    )

    # Generate LaTeX
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Provider} & \textbf{Agent Errors} & \textbf{User Errors} & \textbf{Total} \\",
        r"\midrule",
    ]

    provider_display = {
        "xai": "xAI",
        "openai": "OpenAI",
        "gemini": "Google",
        "amazon": "Amazon",
    }

    for provider in ["gemini", "openai", "xai", "amazon"]:
        if provider in error_counts.index:
            row = error_counts.loc[provider]
            display = provider_display.get(provider, provider.capitalize())
            lines.append(
                f"{display} & {int(row['agent'])} ({row['agent_pct']}\\%) & "
                f"{int(row['user'])} & {int(row['total'])} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = output_dir / f"error_breakdown_{analysis_name}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {tex_path}")

    return tex_path


def generate_error_type_table(
    df: pd.DataFrame,
    analysis_name: str,
    output_dir: Path,
) -> Optional[Path]:
    """Generate LaTeX table summarizing error types."""

    if df.empty or "Error type" not in df.columns:
        return None

    # Normalize error types
    df = df.copy()
    df["error_type_clean"] = df["Error type"].fillna("unknown").str.lower().str.strip()

    # Common error type categories
    def categorize_error(error_type: str) -> str:
        error_lower = error_type.lower()
        if "transcription" in error_lower:
            return "Transcription"
        elif (
            "vad" in error_lower
            or "unresponsive" in error_lower
            or "disappear" in error_lower
        ):
            return "VAD/Unresponsive"
        elif "logical" in error_lower:
            return "Logical"
        elif "hallucination" in error_lower:
            return "Hallucination"
        elif "latency" in error_lower:
            return "Latency"
        elif "tool" in error_lower:
            return "Tool Call"
        elif "input" in error_lower or "collection" in error_lower:
            return "Input Collection"
        elif "instruction" in error_lower:
            return "Instruction Following"
        else:
            return "Other"

    df["error_category"] = df["error_type_clean"].apply(categorize_error)

    # Count by category
    category_counts = df["error_category"].value_counts().sort_values(ascending=False)

    # Generate LaTeX
    lines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"\textbf{Error Type} & \textbf{Count} \\",
        r"\midrule",
    ]

    for category, count in category_counts.items():
        lines.append(f"{category} & {count} \\\\")

    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Total}} & \\textbf{{{len(df)}}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = output_dir / f"error_types_{analysis_name}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {tex_path}")

    return tex_path


def generate_all_qualitative_outputs(
    output_dir: Optional[Path] = None,
    copy_to_paper_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Generate all qualitative analysis outputs."""

    if output_dir is None:
        output_dir = (
            DATA_DIR / "exp" / "tau-voice-results" / "analysis" / "qualitative_analysis"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("GENERATING QUALITATIVE ANALYSIS OUTPUTS")
    logger.info("=" * 70)

    result_paths = {}

    for analysis_name, notes_dir in ANALYSIS_DIRS.items():
        if not notes_dir.exists():
            logger.warning(f"Directory not found: {notes_dir}")
            continue

        # Generate main CSV
        csv_path = generate_qualitative_csv(analysis_name, notes_dir, output_dir)
        if csv_path:
            result_paths[f"{analysis_name}_csv"] = csv_path

            # Load and generate tables
            df = pd.read_csv(csv_path)

            # Aggregate tables for main paper (not by provider)
            tex_path = generate_error_source_aggregate_table(
                df, analysis_name, output_dir
            )
            if tex_path:
                result_paths[f"{analysis_name}_source_aggregate"] = tex_path

            tex_path = generate_error_type_aggregate_table(
                df, analysis_name, output_dir
            )
            if tex_path:
                result_paths[f"{analysis_name}_type_aggregate"] = tex_path

            # Full notes table for appendix
            tex_path = generate_full_notes_table(df, analysis_name, output_dir)
            if tex_path:
                result_paths[f"{analysis_name}_full_notes"] = tex_path

            # Error breakdown by provider (for appendix)
            tex_path = generate_error_breakdown_table(df, analysis_name, output_dir)
            if tex_path:
                result_paths[f"{analysis_name}_breakdown"] = tex_path

            # Error types table (for appendix)
            tex_path = generate_error_type_table(df, analysis_name, output_dir)
            if tex_path:
                result_paths[f"{analysis_name}_types"] = tex_path

    logger.info("=" * 70)
    logger.info(f"Qualitative outputs saved to {output_dir}")

    # Copy to paper directory if requested
    if copy_to_paper_dir:
        import shutil

        copy_to_paper_dir = Path(copy_to_paper_dir)
        copy_to_paper_dir.mkdir(parents=True, exist_ok=True)

        for pattern in ["*.tex", "*.csv"]:
            for src_file in output_dir.glob(pattern):
                dst_file = copy_to_paper_dir / src_file.name
                shutil.copy2(src_file, dst_file)
                logger.info(f"Copied: {src_file.name} -> {copy_to_paper_dir}")

    return result_paths


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate qualitative analysis outputs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--copy-to-paper-dir",
        type=Path,
        default=None,
        help="Copy outputs to paper directory (e.g., papers/tau-voice/results/)",
    )

    args = parser.parse_args()

    generate_all_qualitative_outputs(
        output_dir=args.output_dir,
        copy_to_paper_dir=args.copy_to_paper_dir,
    )


if __name__ == "__main__":
    main()
