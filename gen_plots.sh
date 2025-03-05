
# Define arrays for task types and trial IDs
TASK_TYPES=("substep" "substep_interaction" "all")
TRIAL_IDS=(1 2 3 4 5)

# Loop through each IC run and task type, with trial ID as the innermost loop
for ic_run in "" "_6_ic"; do
    for task_type in "${TASK_TYPES[@]}"; do
        # First create multiple folder plots for each task type
        echo "Running multiple folder plot with ${task_type}"
        folder_paths=""
        for id in "${TRIAL_IDS[@]}"; do
            folder_paths+="logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/trial_${id}${ic_run}/ "
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
                echo "Running with no flag for trial ${id}"
            else
                echo "Running with ${task_type} for trial ${id}"
            fi
            
            # Execute the plot command
            if [ "$task_type" = "substep" ]; then
                python scripts/plot_from_folder.py logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/trial_${id}${ic_run}/ --task_type $task_type --granularity 200
            else
                python scripts/plot_from_folder.py logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/trial_${id}${ic_run}/ --task_type $task_type --granularity 50
            fi
        done
    done
    
    # Do the granularity 200 plots for both multiple folders and individual trials
    folder_paths=""
    for id in "${TRIAL_IDS[@]}"; do
        folder_paths+="logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/trial_${id}${ic_run}/ "
    done
    python scripts/plot_from_folder.py $folder_paths --granularity 200 --multiple_folders
    
    for id in "${TRIAL_IDS[@]}"; do
        python scripts/plot_from_folder.py logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/trial_${id}${ic_run}/ --granularity 200
    done

    # Also run pass@k plots
    python scripts/plot_from_folder.py $folder_paths --granularity 200 --multiple_folders --task_type pass_at_k
done