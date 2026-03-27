"""
Shared data loading utilities for tau_voice experiment analysis.

This module provides common functions for loading and processing simulation
results from tau_voice experiments. Used by both performance_analysis.py
and voice_analysis.py.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from loguru import logger

from tau2.data_model.simulation import Results


def extract_params_from_results(sim_results: Results, folder_path: Path) -> dict:
    """
    Extract experiment parameters from a Results object.

    Args:
        sim_results: The loaded Results object
        folder_path: Path to the folder containing results.json

    Returns:
        dict with keys: domain, speech_complexity, llm, provider, num_tasks, max_steps, folder
    """
    info = sim_results.info

    # Extract domain from environment_info
    domain = info.environment_info.domain_name if info.environment_info else None

    # Extract speech_complexity from info
    speech_complexity = getattr(info, "speech_complexity", None)

    # Extract LLM from agent_info
    llm = info.agent_info.llm if info.agent_info else None

    # Extract provider from audio_native_config or from LLM string
    provider = None
    if hasattr(info, "audio_native_config") and info.audio_native_config:
        provider = getattr(info.audio_native_config, "provider", None)
    if provider is None and llm:
        # Try to extract from LLM string like "openai:gpt-realtime-2025-08-28"
        if ":" in llm:
            provider = llm.split(":")[0]

    # Extract other params
    num_tasks = len(sim_results.tasks) if sim_results.tasks else 0
    max_steps = info.max_steps if hasattr(info, "max_steps") else None

    return {
        "domain": domain,
        "speech_complexity": speech_complexity,
        "llm": llm,
        "provider": provider,
        "num_tasks": num_tasks,
        "max_steps": max_steps,
        "folder": folder_path.name,
        "folder_path": str(folder_path),
    }


def _load_single_result(
    results_file: Path, filter_domains: Optional[List[str]] = None
) -> Tuple[Optional[Tuple[dict, Results]], Optional[str], Optional[str]]:
    """
    Load a single results.json file.

    Returns:
        Tuple of (result, skipped_name, error_message)
        - result: (params, Results) if successfully loaded, None otherwise
        - skipped_name: folder name if skipped due to empty simulations, None otherwise
        - error_message: error message if failed, None otherwise
    """
    try:
        sim_results = Results.load(results_file)
        folder_path = results_file.parent

        # Skip files with empty simulations
        if not sim_results.simulations:
            return (None, folder_path.name, None)

        # Extract params from the results object
        params = extract_params_from_results(sim_results, folder_path)

        # Filter by domain if specified
        if filter_domains and params["domain"] not in filter_domains:
            return (None, None, None)

        return ((params, sim_results), None, None)
    except Exception as e:
        return (None, None, f"Failed to load {results_file}: {e}")


def load_simulation_results(
    data_dir: Path,
    filter_domains: Optional[List[str]] = None,
) -> List[Tuple[dict, Results]]:
    """
    Load all tau_voice simulation results from the given directory.

    Recursively searches for results.json files and extracts metadata from them.

    Args:
        data_dir: Directory containing simulation folders (searched recursively)
        filter_domains: Optional list of domains to include

    Returns:
        List of tuples (params_dict, Results)
    """
    results = []
    skipped_empty = []

    if not data_dir.exists():
        raise ValueError(f"Data directory {data_dir} does not exist.")

    # Find all results.json files recursively
    results_files = list(data_dir.rglob("results.json"))

    if not results_files:
        logger.warning(f"No results.json files found in {data_dir}")
        return results

    logger.info(f"Found {len(results_files)} results.json file(s)")
    logger.info("Loading results sequentially...")

    for results_file in results_files:
        result, skipped_name, error_msg = _load_single_result(
            results_file, filter_domains
        )

        if error_msg:
            logger.error(error_msg)
        elif skipped_name:
            logger.warning(f"Skipping {skipped_name}: results.json has 0 simulations")
            skipped_empty.append(skipped_name)
        elif result:
            params, sim_results = result
            results.append((params, sim_results))
            logger.info(
                f"Loaded: {params['domain']} / {params['speech_complexity']} / {params['llm']}"
            )

    # Print summary table of simulation counts
    if results:
        _print_simulation_summary(results, skipped_empty)

    return results


def _print_simulation_summary(
    results: List[Tuple[dict, Results]], skipped_empty: List[str]
) -> None:
    """
    Print a summary table of simulation counts and warn about discrepancies.
    """
    # Build summary data
    summary_rows = []
    for params, sim_results in results:
        summary_rows.append(
            {
                "domain": params.get("domain", "unknown"),
                "complexity": params.get("speech_complexity", "unknown"),
                "llm": params.get("llm", "unknown"),
                "num_simulations": len(sim_results.simulations),
            }
        )

    df_summary = pd.DataFrame(summary_rows)

    # Print the summary table
    logger.info("=" * 70)
    logger.info("SIMULATION COUNT SUMMARY")
    logger.info("=" * 70)

    # Group by domain and complexity to show counts
    for domain in sorted(df_summary["domain"].unique()):
        domain_df = df_summary[df_summary["domain"] == domain]
        for complexity in sorted(domain_df["complexity"].unique()):
            subset = domain_df[domain_df["complexity"] == complexity]
            counts_str = ", ".join(
                f"{row['llm']}: {row['num_simulations']}"
                for _, row in subset.iterrows()
            )
            logger.info(f"  {domain} / {complexity}: {counts_str}")

    # Check for discrepancies within each domain (counts should be same within a domain)
    discrepancies = []
    for domain, group in df_summary.groupby("domain"):
        unique_counts = group["num_simulations"].unique()
        if len(unique_counts) > 1:
            # Show which configs have which counts
            details = group.groupby("num_simulations").apply(
                lambda g: [f"{r['complexity']}/{r['llm']}" for _, r in g.iterrows()],
                include_groups=False,
            )
            details_str = "; ".join(
                f"{count}: {', '.join(configs)}" for count, configs in details.items()
            )
            discrepancies.append(f"{domain}: counts vary - {details_str}")

    if discrepancies:
        logger.warning(
            "Discrepancies found within domains (expected same count per domain):"
        )
        for d in discrepancies:
            logger.warning(f"  - {d}")

    if skipped_empty:
        logger.warning(f"Skipped {len(skipped_empty)} file(s) with 0 simulations:")
        for name in skipped_empty:
            logger.warning(f"  - {name}")

    logger.info("=" * 70)
