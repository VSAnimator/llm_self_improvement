#!/bin/bash

# Define arrays for task types, trial IDs, and IC configurations
TASK_TYPES=("substep" "substep_interaction" "all")
TRIAL_IDS=(1 2 3 4 5)
SEGMENTS=(1 2 3 4 5 6 7 8)
IC_CONFIGS=("6ic")

# First, copy all segment runs to a single shared folder per trial and IC configuration
for ic in "${IC_CONFIGS[@]}"; do
    for id in "${TRIAL_IDS[@]}"; do
        # Create the destination directory if it doesn't exist
        mkdir -p "logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/pbt_zero_combined_${ic}_trial_${id}/"
        
        # Copy all segment files to the combined folder
        for segment in "${SEGMENTS[@]}"; do
            # Source directory for this segment
            src_dir="logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/pbt_${ic}_seg20_trial_${id}_segment_${segment}/"
            
            # Copy all txt files from the segment folder to the combined folder
            if [ -d "$src_dir" ]; then
                cp "$src_dir"/*.txt "logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/pbt_zero_combined_${ic}_trial_${id}/"
            fi
        done
        
        echo "Combined all segments for ${ic} trial ${id}"
    done
done

# Now generate plots for each combined trial folder and IC configuration
for ic in "${IC_CONFIGS[@]}"; do
    for task_type in "${TASK_TYPES[@]}"; do
        # First create multiple folder plots for each task type
        echo "Running multiple folder plot with ${task_type} for ${ic}"
        folder_paths=""
        for id in "${TRIAL_IDS[@]}"; do
            folder_paths+="logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/pbt_zero_combined_${ic}_trial_${id}/ "
        done
        
        if [ "$task_type" = "substep" ]; then
            python scripts/plot_from_folder.py $folder_paths --task_type $task_type --granularity 200 --multiple_folders
        else
            python scripts/plot_from_folder.py $folder_paths --task_type $task_type --granularity 50 --multiple_folders
        fi
        
        # Then do individual trial plots
        for id in "${TRIAL_IDS[@]}"; do
            # Set the command with appropriate flags
            if [ -z "$task_type" ]; then
                echo "Running with no flag for ${ic} trial ${id}"
            else
                echo "Running with ${task_type} for ${ic} trial ${id}"
            fi
            
            # Execute the plot command
            if [ "$task_type" = "substep" ]; then
                python scripts/plot_from_folder.py logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/pbt_zero_combined_${ic}_trial_${id}/ --task_type $task_type --granularity 200
            else
                python scripts/plot_from_folder.py logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/pbt_zero_combined_${ic}_trial_${id}/ --task_type $task_type --granularity 50
            fi
        done
    done
    
    # Do the granularity 200 plots for both multiple folders and individual trials
    folder_paths=""
    for id in "${TRIAL_IDS[@]}"; do
        folder_paths+="logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/pbt_zero_combined_${ic}_trial_${id}/ "
    done
    python scripts/plot_from_folder.py $folder_paths --granularity 200 --multiple_folders
    
    for id in "${TRIAL_IDS[@]}"; do
        python scripts/plot_from_folder.py logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/pbt_zero_combined_${ic}_trial_${id}/ --granularity 200
    done
    
    # Also run pass@k plots
    python scripts/plot_from_folder.py $folder_paths --granularity 200 --multiple_folders --task_type pass_at_k
done