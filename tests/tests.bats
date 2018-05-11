#!/usr/bin/env bats

@test "find-cags.py in the PATH" {
  v="$(find-cags.py -h 2>&1 || true )"
  [[ "$v" =~ "Find a set of co-abundant genes" ]]
}

@test "Analyze test dataset" {
  find-cags.py \
    --sample-sheet /usr/local/tests/sample_sheet_docker.json \
    --output-prefix test \
    --output-folder /usr/local/tests/ \
    --temp-folder /scratch \
    --normalization median \
    --threads 1 \
    --test

  # Make sure the output files exist
  [[ -s /usr/local/tests/test.feather ]]
  [[ -s /usr/local/tests/test.cags.feather ]]
  [[ -s /usr/local/tests/test.cags.json.gz ]]
  [[ -s /usr/local/tests/test.logs.txt ]]
}
