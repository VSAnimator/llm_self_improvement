
# Define arrays for task types and trial IDs
TRIAL_IDS=(1 2 3 4 5)

# Loop through each IC run and task type, with trial ID as the innermost loop
for ic_run in ""; do
    # Do the granularity 200 plots for both multiple folders and individual trials
    folder_paths=""
    for id in "${TRIAL_IDS[@]}"; do
        folder_paths+="logs/episodes/wordcraft/train/rap_noplan/openai/gpt-4o-mini/wordcraft_depth_2_humanic_train_${id}/ "
    done
    python scripts/plot_from_folder.py $folder_paths --task_type all --granularity 200 --multiple_folders

    for id in "${TRIAL_IDS[@]}"; do
        python scripts/plot_from_folder.py logs/episodes/wordcraft/train/rap_noplan/openai/gpt-4o-mini/wordcraft_depth_2_humanic_train_${id}/ --granularity 200
    done

    # Also run pass@k plots
    python scripts/plot_from_folder.py $folder_paths --granularity 200 --multiple_folders --task_type pass_at_k
done