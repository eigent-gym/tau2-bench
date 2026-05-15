#!/bin/bash
# Build the arXiv variant of the camera-ready paper.
# Output dir: <parent>/tau-voice_arXiv_icml (sibling of tau-voice-icml).
# Uses arxiv_latex_cleaner to:
#   - strip \todo / \authornote / \old / \why (commands + content)
#   - flatten \new{...} -> ... (keep content, drop macro)
# The reviewer-marks toggles in main.tex (revmarks, shownotes) are irrelevant
# after this cleaning because the macros themselves are removed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
ARXIV_DIR="${PARENT_DIR}/tau-voice_arXiv_icml"

rm -rf "$ARXIV_DIR"

arxiv_latex_cleaner "$SCRIPT_DIR" --config "$SCRIPT_DIR/cleaner_config.yaml"

CLEANER_OUTPUT="${SCRIPT_DIR}_arXiv"
mv "$CLEANER_OUTPUT" "$ARXIV_DIR"

rm -f "$ARXIV_DIR"/{UPDATE_STAT_SIG.md,cleaner_config.yaml,build_arxiv.sh}

cp "$SCRIPT_DIR/main.bbl" "$ARXIV_DIR/main.bbl"

echo "Done. Output: $ARXIV_DIR"
echo "To compile: cd $ARXIV_DIR && pdflatex -shell-escape main.tex && pdflatex -shell-escape main.tex"
