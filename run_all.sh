#!/usr/bin/env bash
# Reproduce every table and figure in the paper.
set -e
cd "$(dirname "$0")"
python src/generate_data.py
python src/train_models.py
python src/line_twin.py
python src/make_figures.py
echo "Done. Results in results/, figures in figures/."
