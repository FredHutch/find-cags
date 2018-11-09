#!/bin/bash

set -e

temp_folder=test-$(date "+%Y-%m-%d-%H-%M-%S")
mkdir $temp_folder

./find-cags.py \
    --sample-sheet tests/sample_sheet.json \
    --output-prefix test \
    --output-folder $temp_folder \
    --temp-folder $temp_folder \
    --min-samples 9 \
    --normalization clr \
    --max-dist 0.1 \
    --threads 1
