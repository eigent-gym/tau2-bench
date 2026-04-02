#!/usr/bin/env python3
"""Create a trimmed copy of an experiment directory.

Copies the directory structure but:
- For each task, only keeps the sim directory referenced in results.json
- Within each sim, only keeps the audio/ directory (drops llm_debug/ and task.log)
- For dir-format experiments (simulations/ subdir), copies canonical sim JSON files
- Text-only experiments (no tasks/ or artifacts/ dir) get just their results.json copied

Supports both storage layouts:
- Old layout: results.json (monolithic) + tasks/<task>/<sim>/audio/
- Dir layout:  results.json (metadata)  + simulations/*.json + artifacts/<task>/<sim>/audio/

The original experiment directory is NEVER modified.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_results_json(results_path: Path) -> dict[str, str]:
    """Return a mapping of task_id -> sim_id from a results.json file.

    Handles both monolithic format (``simulations`` list) and dir format
    (``simulation_index`` list).
    """
    with open(results_path) as f:
        data = json.load(f)

    entries = data.get("simulation_index") or data.get("simulations", [])

    task_to_sim: dict[str, str] = {}
    for sim in entries:
        task_id = str(sim["task_id"])
        sim_id = sim["id"]
        task_to_sim[task_id] = sim_id

    return task_to_sim


def _copy_simulations_dir(
    exp_src: Path,
    exp_dst: Path,
    canonical_sim_ids: set[str],
    stats: dict[str, int],
    *,
    dry_run: bool,
) -> None:
    """Copy canonical simulation JSON files from the simulations/ subdir."""
    sims_src = exp_src / "simulations"
    if not sims_src.is_dir():
        return

    sims_dst = exp_dst / "simulations"
    for sim_file in sorted(sims_src.iterdir()):
        if not sim_file.name.endswith(".json"):
            continue
        if sim_file.stem not in canonical_sim_ids:
            stats["skipped_sims"] += 1
            continue
        if dry_run:
            log.info("  [dry-run] copy simulations/%s", sim_file.name)
        else:
            sims_dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sim_file, sims_dst / sim_file.name)
        stats["copied_sims"] += 1


def _copy_audio_dirs(
    audio_root: Path,
    dst_subdir_name: str,
    exp_dst: Path,
    task_to_sim: dict[str, str],
    stats: dict[str, int],
    *,
    dry_run: bool,
) -> None:
    """Copy audio/ dirs for canonical sims under an audio root (tasks/ or artifacts/)."""
    for task_dir in sorted(audio_root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue

        task_id = task_dir.name.split("_", 1)[1]
        canonical_sim_id = task_to_sim.get(task_id)

        if canonical_sim_id is None:
            log.warning(
                "  Task %s has no simulation in results.json, skipping all sims",
                task_dir.name,
            )
            sim_dirs = [d for d in task_dir.iterdir() if d.is_dir()]
            stats["skipped_sims"] += len(sim_dirs)
            continue

        canonical_sim_name = f"sim_{canonical_sim_id}"

        for sim_dir in sorted(task_dir.iterdir()):
            if not sim_dir.is_dir() or not sim_dir.name.startswith("sim_"):
                continue

            if sim_dir.name != canonical_sim_name:
                stats["skipped_sims"] += 1
                continue

            audio_src = sim_dir / "audio"
            if not audio_src.is_dir():
                log.warning(
                    "  Missing audio/ in %s/%s/audio",
                    task_dir.name,
                    sim_dir.name,
                )
                stats["missing_audio"] += 1
                continue

            audio_dst = (
                exp_dst / dst_subdir_name / task_dir.name / sim_dir.name / "audio"
            )
            if dry_run:
                log.info(
                    "  [dry-run] copy %s/%s/audio/",
                    task_dir.name,
                    sim_dir.name,
                )
            else:
                shutil.copytree(audio_src, audio_dst)
            stats["copied_audio"] += 1


def process_experiment(
    exp_src: Path, exp_dst: Path, *, dry_run: bool
) -> dict[str, int]:
    """Process a single experiment directory. Returns stats dict."""
    stats = {
        "copied_audio": 0,
        "copied_sims": 0,
        "skipped_sims": 0,
        "missing_audio": 0,
    }
    results_path = exp_src / "results.json"

    if dry_run:
        log.info("  [dry-run] copy %s", results_path.name)
    else:
        exp_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(results_path, exp_dst / "results.json")

    task_to_sim = parse_results_json(results_path)

    # Dir-format: copy canonical simulation JSON files
    _copy_simulations_dir(
        exp_src, exp_dst, set(task_to_sim.values()), stats, dry_run=dry_run
    )

    # Locate the audio artifacts directory (artifacts/ for dir-format, tasks/ for old)
    artifacts_dir = exp_src / "artifacts"
    tasks_dir = exp_src / "tasks"
    if artifacts_dir.is_dir():
        _copy_audio_dirs(
            artifacts_dir, "artifacts", exp_dst, task_to_sim, stats, dry_run=dry_run
        )
    elif tasks_dir.is_dir():
        _copy_audio_dirs(
            tasks_dir, "tasks", exp_dst, task_to_sim, stats, dry_run=dry_run
        )
    else:
        log.info("  No artifacts/ or tasks/ directory (text-only), copied results.json")

    return stats


def trim_experiment_data(input_dir: Path, output_dir: Path, *, dry_run: bool) -> None:
    """Walk the input directory, find all experiments, and create trimmed copies."""
    if not input_dir.is_dir():
        log.error("Input directory does not exist: %s", input_dir)
        raise SystemExit(1)

    if output_dir.exists() and not dry_run:
        log.error("Output directory already exists: %s", output_dir)
        log.error(
            "Remove it first or choose a different name to avoid accidental data mixing."
        )
        raise SystemExit(1)

    experiment_dirs = sorted(input_dir.rglob("results.json"))
    log.info("Found %d experiments in %s", len(experiment_dirs), input_dir)

    totals = {
        "copied_audio": 0,
        "copied_sims": 0,
        "skipped_sims": 0,
        "missing_audio": 0,
    }

    for results_path in experiment_dirs:
        exp_src = results_path.parent
        rel_path = exp_src.relative_to(input_dir)
        exp_dst = output_dir / rel_path

        log.info("Processing: %s", rel_path)
        stats = process_experiment(exp_src, exp_dst, dry_run=dry_run)

        for k in totals:
            totals[k] += stats[k]

    log.info("--- Summary ---")
    log.info("Experiments processed: %d", len(experiment_dirs))
    log.info("Audio dirs copied:     %d", totals["copied_audio"])
    if totals["copied_sims"] > 0:
        log.info("Sim JSON files copied: %d", totals["copied_sims"])
    log.info("Sim dirs skipped:      %d", totals["skipped_sims"])
    if totals["missing_audio"] > 0:
        log.warning("Missing audio dirs:    %d", totals["missing_audio"])
    if dry_run:
        log.info("(dry-run mode -- nothing was actually copied)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Source experiment directory (will NOT be modified)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for trimmed copy",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without actually copying",
    )
    args = parser.parse_args()

    trim_experiment_data(args.input_dir, args.output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
