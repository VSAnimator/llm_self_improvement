#!/bin/bash

# Define the pattern for test folders
FOLDER_PATTERN="logs/episodes/alfworld_test/eval_out_of_distribution/rap_flex/openai/gpt-4o-mini/pbt_trial_*_6ic_*"

# Define the regex pattern to extract the number of training tasks
# This assumes folders are named with a pattern that includes the number of training tasks
REGEX_PATTERN="pbt_trial_(\\d+)_6ic_(\\d+)"

# Output file path
OUTPUT_PATH="alfworld_test_pbt_success_rate_plot.png"

# Segment size (number of most recent tasks to consider)
SEGMENT_SIZE=1000

# Run the plotting script
echo "Plotting ALFWorld PBT test success rates..."
python scripts/test_plot.py "$FOLDER_PATTERN" "$REGEX_PATTERN" --output "$OUTPUT_PATH" --segment-size "$SEGMENT_SIZE"

echo "Plot generation complete."
