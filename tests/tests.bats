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
    --max-dist 0.3 \
    --threads 1 \
    --test

  # Make sure the output files exist
  [[ -s /usr/local/tests/test.feather ]]
  [[ -s /usr/local/tests/test.cags.json.gz ]]
  [[ -s /usr/local/tests/test.logs.txt ]]
}

@test "Analyze test dataset with automatic CLR floor" {
  find-cags.py \
    --sample-sheet /usr/local/tests/sample_sheet_docker.json \
    --output-prefix test_clr_floor \
    --output-folder /usr/local/tests/ \
    --temp-folder /scratch \
    --normalization clr \
    --max-dist 0.2 \
    --threads 1 \
    --clr-floor auto \
    --test

  # Make sure the output files exist
  [[ -s /usr/local/tests/test_clr_floor.feather ]]
  [[ -s /usr/local/tests/test_clr_floor.cags.json.gz ]]
  [[ -s /usr/local/tests/test_clr_floor.logs.txt ]]
}


@test "Analyze all normalization approaches" {
  for norm in median clr sum; do
    find-cags.py \
      --sample-sheet /usr/local/tests/sample_sheet_docker.json \
      --output-prefix test_$norm \
      --output-folder /usr/local/tests/ \
      --temp-folder /scratch \
      --normalization $norm \
      --max-dist 0.2 \
      --threads 1 \
      --clr-floor auto \
      --test

    # Make sure the output files exist
    [[ -s /usr/local/tests/test_$norm.feather ]]
    [[ -s /usr/local/tests/test_$norm.cags.json.gz ]]
    [[ -s /usr/local/tests/test_$norm.logs.txt ]]
  done
}
