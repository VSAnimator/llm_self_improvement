#!/bin/bash

# Define the pattern for test folders
FOLDER_PATTERN="logs/episodes/wordcraft/test/rap_noplan/openai/gpt-4o-mini/zero_*_10ic_*"

# Define the regex pattern to extract the number of training tasks
# This assumes folders are named with a pattern that includes the number of training tasks
REGEX_PATTERN="zero_(\\d+)_10ic_(\\d+)"

# Output file path
OUTPUT_PATH="wordcraft_zero_test_success_rate_plot.png"

# Segment size (number of most recent tasks to consider)
SEGMENT_SIZE=1000

# Run the plotting script
echo "Plotting Wordcraft test success rates..."
python scripts/test_plot.py "$FOLDER_PATTERN" "$REGEX_PATTERN" --output "$OUTPUT_PATH" --segment-size "$SEGMENT_SIZE"

echo "Plot generation complete."
