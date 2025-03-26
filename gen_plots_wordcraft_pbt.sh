#!/bin/bash

# Define arrays for task types, trial IDs, and IC configurations
TASK_TYPES=("all" "pass_at_k")
TRIAL_IDS=(1 2 3 4 5)
SEGMENTS=(1 2 3 4 5 6 7 8 9)
IC_CONFIGS=("10ic")

# First, copy all segment runs to a single shared folder per trial and IC configuration
for ic in "${IC_CONFIGS[@]}"; do
    for id in "${TRIAL_IDS[@]}"; do
        # Create the destination directory if it doesn't exist
        mkdir -p "logs/episodes/wordcraft/train/rap_noplan/openai/gpt-4o-mini/pbt_combined_${ic}_trial_${id}/"
        
        # Copy all segment files to the combined folder
        for segment in "${SEGMENTS[@]}"; do
            # Source directory for this segment
            src_dir="logs/episodes/wordcraft/train/rap_noplan/openai/gpt-4o-mini/pbt_${ic}_seg10_trial_${id}_segment_${segment}/"
            
            # Copy all txt files from the segment folder to the combined folder
            if [ -d "$src_dir" ]; then
                cp "$src_dir"/*.txt "logs/episodes/wordcraft/train/rap_noplan/openai/gpt-4o-mini/pbt_combined_${ic}_trial_${id}/"
            fi
        done
        
        echo "Combined all segments for ${ic} trial ${id}"
    done
done

# Now generate plots for each combined trial folder and IC configuration
for ic in "${IC_CONFIGS[@]}"; do
    folder_paths=""
    for id in "${TRIAL_IDS[@]}"; do
        folder_paths+="logs/episodes/wordcraft/train/rap_noplan/openai/gpt-4o-mini/pbt_combined_${ic}_trial_${id}/ "
    done
    for task_type in "${TASK_TYPES[@]}"; do
        # First create multiple folder plots for each task type
        echo "Running multiple folder plot with ${task_type} for ${ic}"
        python scripts/plot_from_folder.py $folder_paths --task_type $task_type --granularity 200 --multiple_folders
        
        # Then do individual trial plots (only for "all" task type)
        if [ "$task_type" = "all" ]; then
            for id in "${TRIAL_IDS[@]}"; do
                echo "Running with ${task_type} for ${ic} trial ${id}"
                python scripts/plot_from_folder.py logs/episodes/wordcraft/train/rap_noplan/openai/gpt-4o-mini/pbt_combined_${ic}_trial_${id}/ --granularity 200
            done
        fi
    done
    python scripts/plot_from_folder.py $folder_paths --granularity 200 --multiple_folders
done