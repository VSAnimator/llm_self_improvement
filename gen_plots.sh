
# Define arrays for task types and trial IDs
TASK_TYPES=("substep" "substep_interaction" "all")
TRIAL_IDS=(1 2 3 4 5)

# Loop through each task type and trial ID
for id in "${TRIAL_IDS[@]}"; do
    for ic_run in "" "_6_ic"; do
        for task_type in "${TASK_TYPES[@]}"; do
            # Set the command with appropriate flags
            if [ -z "$task_type" ]; then
                echo "Running with no flag for trial ${id}"
            else
                echo "Running with ${task_type} for trial ${id}"
            fi
            
            # Execute the plot command
            python scripts/plot_from_folder.py logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/trial_${id}${ic_run}/ --task_type $task_type --granularity 50
        done
        python scripts/plot_from_folder.py logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/trial_${id}${ic_run}/ --granularity 50
    done
done