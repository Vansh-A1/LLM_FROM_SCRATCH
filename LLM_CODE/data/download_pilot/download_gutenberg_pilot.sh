#!/bin/bash

set -e

OUTPUT_DIR="/data/vansh/LLM_FROM_SCRATCH/LLM_CODE/data/raw/gutenberg"

mkdir -p "$OUTPUT_DIR"

cd "$OUTPUT_DIR"

echo "Downloading small Project Gutenberg pilot..."

# Download a small English TXT sample using Gutenberg's
# supported robot/harvest interface.
wget -w 2 -m -H \
"https://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en"

echo "Gutenberg pilot download complete."