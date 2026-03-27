#!/usr/bin/env python3
"""
Combined analysis script for tau_voice experiments.

Loads simulation data once and runs both performance and voice analysis,
avoiding the expensive data loading step being repeated.

Usage:
    # Run both analyses with default output structure
    python -m experiments.tau_voice.exp.run_all_analysis --data-dir ./data/experiments

    # Specify custom output directory
    python -m experiments.tau_voice.exp.run_all_analysis \
        --data-dir ./data/experiments \
        --output-dir ~/results/analysis

    # Run only performance analysis
    python -m experiments.tau_voice.exp.run_all_analysis \
        --data-dir ./data/experiments \
        --performance-only

    # Run only voice analysis
    python -m experiments.tau_voice.exp.run_all_analysis \
        --data-dir ./data/experiments \
        --voice-only
"""

import argparse
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from experiments.tau_voice.exp.data_loader import load_simulation_results
from tau2.utils.utils import DATA_DIR


def _run_paper_outputs(
    base_output: Path,
    run_performance: bool = True,
    run_voice: bool = True,
    copy_to_paper_dir: Optional[Path] = None,
) -> None:
    """Generate paper-ready outputs from CSVs."""
    from experiments.tau_voice.exp.paper_outputs import generate_all_paper_outputs

    logger.info("")
    logger.info("=" * 70)
    logger.info("GENERATING PAPER OUTPUTS")
    logger.info("=" * 70)

    start_time = time.time()
    generate_all_paper_outputs(
        base_output,
        performance_only=not run_voice,
        voice_only=not run_performance,
        copy_to_paper_dir=copy_to_paper_dir,
    )
    logger.info(f"Paper outputs generated in {time.time() - start_time:.1f}s")


def run_all_analysis(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    filter_domains: Optional[list[str]] = None,
    run_performance: bool = True,
    run_voice: bool = True,
    clean: bool = False,
    plots_only: bool = False,
    copy_to_paper_dir: Optional[Path] = None,
) -> None:
    """
    Run both performance and voice analysis on the same loaded data.

    Args:
        data_dir: Directory containing simulation folders
        output_dir: Base output directory. If None, uses data_dir/analysis/
        filter_domains: Optional list of domains to include
        run_performance: Whether to run performance analysis
        run_voice: Whether to run voice analysis
        clean: Whether to clean output directories before running
        plots_only: If True, regenerate plots from existing CSVs without reloading data
    """
    import shutil

    # Determine output directories
    if output_dir is None:
        base_output = data_dir / "analysis"
    else:
        base_output = output_dir

    perf_output = base_output / "performance_analysis"
    voice_output = base_output / "voice_analysis"

    # Handle clean flag (not compatible with plots-only)
    if clean and not plots_only:
        for out_dir in [perf_output, voice_output]:
            if out_dir.exists():
                logger.warning(f"Cleaning output directory: {out_dir}")
                shutil.rmtree(out_dir)
                logger.info(f"Deleted: {out_dir}")

    # =========================================================================
    # Handle plots-only mode (no data loading needed)
    # =========================================================================
    if plots_only:
        logger.info("=" * 70)
        logger.info("REGENERATING PLOTS FROM EXISTING CSVs")
        logger.info("=" * 70)

        if run_performance:
            if not perf_output.exists():
                logger.error(f"Performance output directory not found: {perf_output}")
            else:
                from experiments.tau_voice.exp.performance_analysis import (
                    regenerate_plots_from_csv,
                )

                logger.info(f"Regenerating performance plots from: {perf_output}")
                start_time = time.time()
                regenerate_plots_from_csv(perf_output)
                logger.info(
                    f"Performance plots regenerated in {time.time() - start_time:.1f}s"
                )

        if run_voice:
            if not voice_output.exists():
                logger.error(f"Voice output directory not found: {voice_output}")
            else:
                from experiments.tau_voice.exp.voice_analysis import (
                    regenerate_plots_from_csv as regenerate_voice_plots,
                )

                logger.info(f"Regenerating voice plots from: {voice_output}")
                start_time = time.time()
                regenerate_voice_plots(voice_output)
                logger.info(
                    f"Voice plots regenerated in {time.time() - start_time:.1f}s"
                )

        # Generate paper outputs
        _run_paper_outputs(base_output, run_performance, run_voice, copy_to_paper_dir)

        logger.info("")
        logger.info("=" * 70)
        logger.info("PLOT REGENERATION COMPLETE")
        logger.info("=" * 70)
        return

    # =========================================================================
    # Load data once
    # =========================================================================
    logger.info("=" * 70)
    logger.info("LOADING SIMULATION DATA")
    logger.info("=" * 70)

    start_time = time.time()
    results = load_simulation_results(data_dir, filter_domains)
    load_time = time.time() - start_time

    if not results:
        logger.error("No results found. Exiting.")
        return

    logger.info(f"Loaded {len(results)} simulation results in {load_time:.1f}s")

    # =========================================================================
    # Run performance analysis
    # =========================================================================
    if run_performance:
        logger.info("")
        logger.info("=" * 70)
        logger.info("RUNNING PERFORMANCE ANALYSIS")
        logger.info("=" * 70)

        from experiments.tau_voice.exp.performance_analysis import analyze_results

        start_time = time.time()
        analyze_results(
            data_dir=data_dir,
            output_dir=perf_output,
            filter_domains=filter_domains,
            results=results,
        )
        perf_time = time.time() - start_time
        logger.info(f"Performance analysis completed in {perf_time:.1f}s")

    # =========================================================================
    # Run voice analysis
    # =========================================================================
    if run_voice:
        logger.info("")
        logger.info("=" * 70)
        logger.info("RUNNING VOICE ANALYSIS")
        logger.info("=" * 70)

        from experiments.tau_voice.exp.voice_analysis import analyze_voice_results

        start_time = time.time()
        analyze_voice_results(
            data_dir=data_dir,
            output_dir=voice_output,
            filter_domains=filter_domains,
            results=results,
        )
        voice_time = time.time() - start_time
        logger.info(f"Voice analysis completed in {voice_time:.1f}s")

    # =========================================================================
    # Generate paper outputs
    # =========================================================================
    _run_paper_outputs(base_output, run_performance, run_voice, copy_to_paper_dir)

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Data loaded from: {data_dir}")
    logger.info(f"Output saved to: {base_output}")
    if run_performance:
        logger.info(f"  - Performance analysis: {perf_output}")
    if run_voice:
        logger.info(f"  - Voice analysis: {voice_output}")
    logger.info(f"  - Paper outputs: {base_output / 'paper'}")


def main():
    parser = argparse.ArgumentParser(
        description="Run combined performance and voice analysis on tau_voice experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both analyses
  python -m experiments.tau_voice.exp.run_all_analysis --data-dir ./data/experiments

  # Run only performance analysis
  python -m experiments.tau_voice.exp.run_all_analysis --data-dir ./data/experiments --performance-only

  # Run only voice analysis  
  python -m experiments.tau_voice.exp.run_all_analysis --data-dir ./data/experiments --voice-only

  # Specify custom output directory
  python -m experiments.tau_voice.exp.run_all_analysis \\
      --data-dir ./data/experiments \\
      --output-dir ~/results/analysis

  # Clean output before running
  python -m experiments.tau_voice.exp.run_all_analysis --data-dir ./data/experiments --clean

  # Regenerate plots from existing CSVs (no data loading)
  python -m experiments.tau_voice.exp.run_all_analysis --data-dir ./data/experiments --plots-only
        """,
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
        help="Base output directory. Defaults to data_dir/analysis/",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific domains (e.g., retail airline telecom).",
    )
    parser.add_argument(
        "--performance-only",
        action="store_true",
        help="Run only performance analysis.",
    )
    parser.add_argument(
        "--voice-only",
        action="store_true",
        help="Run only voice analysis.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directories before running analysis.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plots from existing CSVs without reloading data.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Generate only paper outputs from existing CSVs.",
    )
    parser.add_argument(
        "--copy-to-paper",
        type=str,
        default=None,
        help="Copy paper outputs to this directory (e.g., papers/tau-voice/results/).",
    )

    args = parser.parse_args()

    # Resolve data directory
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        # Try relative to current directory first
        if not data_dir.exists():
            # Try relative to project root
            project_root = DATA_DIR.parent
            data_dir = project_root / args.data_dir

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Resolve output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()

    # Determine which analyses to run
    run_performance = True
    run_voice = True
    if args.performance_only:
        run_voice = False
    if args.voice_only:
        run_performance = False

    # Handle --paper flag (paper outputs only)
    copy_to_paper_dir = Path(args.copy_to_paper) if args.copy_to_paper else None

    if args.paper:
        if output_dir is None:
            output_dir = data_dir / "analysis"
        if not output_dir.exists():
            logger.error(f"Output directory not found: {output_dir}")
            return
        _run_paper_outputs(output_dir, run_performance, run_voice, copy_to_paper_dir)
        return

    run_all_analysis(
        data_dir=data_dir,
        output_dir=output_dir,
        filter_domains=args.domains,
        run_performance=run_performance,
        run_voice=run_voice,
        clean=args.clean,
        plots_only=args.plots_only,
        copy_to_paper_dir=copy_to_paper_dir,
    )


if __name__ == "__main__":
    main()
