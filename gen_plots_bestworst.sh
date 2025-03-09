
# Define arrays for task types and DB sizes
TASK_TYPES=("substep" "substep_interaction" "all")
DB_SIZES=(10 20 40 60 100 200 400 600 800)
MODES=("best" "worst")

# Loop through each mode (best/worst examples)
for mode in "${MODES[@]}"; do
    # Loop through each DB size
    for db_size in "${DB_SIZES[@]}"; do
        # Loop through each task type
        for task_type in "${TASK_TYPES[@]}"; do
            # Set the command with appropriate flags
            if [ -z "$task_type" ]; then
                echo "Running with no flag for ${mode} examples with DB size ${db_size}"
            else
                echo "Running with ${task_type} for ${mode} examples with DB size ${db_size}"
            fi
            
            # Execute the plot command
            python scripts/plot_from_folder.py logs/episodes/alfworld_test/rap_flex/openai/gpt-4o-mini/expel_${mode}_${db_size}_examples/ --task_type $task_type --granularity 10
        done
        # Run once without task type
        python scripts/plot_from_folder.py logs/episodes/alfworld_test/rap_flex/openai/gpt-4o-mini/expel_${mode}_${db_size}_examples/ --granularity 10
    done
done