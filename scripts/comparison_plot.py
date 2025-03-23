import os
import matplotlib.pyplot as plt
import numpy as np

def reward_heatmap(base_dir, folder_identifiers, exclusion_identifiers=None, num_tasks=134, start_task=0):
    # Identify matching folders
    folders = [
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f))
        and any(identifier in f for identifier in folder_identifiers)
        and (exclusion_identifiers is None or not any(ex_id in f for ex_id in exclusion_identifiers))
    ]

    success_matrix = []
    folder_names = []

    # Sort folders by numeric parts if they exist
    def extract_number(folder_name):
        # Extract numbers from the folder name
        parts = folder_name.split('_')
        # Search from the end to prioritize later numbers (like checkpoint numbers)
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
            # Handle cases like "3ic" where number is part of a string
            if part.endswith('ic') and part[:-2].isdigit():
                return int(part[:-2])
        return 0  # Default value if no number is found
    
    for folder in sorted(folders, key=extract_number):
        folder_path = os.path.join(base_dir, folder)
        success_row = []

        for i in range(start_task, num_tasks):
            file_path = os.path.join(folder_path, f"{i}.txt")
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                    success = "Reward: 1" in content
                    success_row.append(1 if success else 0)
            except FileNotFoundError:
                success_row.append(-1)  # Could be shown as gray
        success_matrix.append(success_row)
        folder_names.append(folder)

    # Convert to numpy array for plotting
    matrix = np.array(success_matrix)
    print(matrix.shape)

    # Define custom colormap: red (0), green (1), gray (-1)
    cmap = plt.cm.get_cmap("RdYlGn", 3)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(16, len(folders) * 0.5))
    plt.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)

    plt.yticks(ticks=np.arange(len(folder_names)), labels=folder_names)
    
    # Adjust x-ticks to account for start_task offset
    task_indices = np.arange(num_tasks - start_task)
    plt.xticks(ticks=task_indices[::10], labels=np.arange(start_task, num_tasks)[::10])
    plt.xlabel("Task Index")
    plt.ylabel("Folder")

    # Add a colorbar legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cmap(norm(1)), label='Success (Reward: 1)'),
        Patch(facecolor=cmap(norm(0)), label='Failure'),
        Patch(facecolor=cmap(norm(-1)), label='Missing File'),
    ]
    #plt.legend(handles=legend_elements, loc='upper right')

    plt.title("Success Heatmap per Folder")
    plt.tight_layout()
    plt.savefig(f"{base_dir}/{folder_identifiers[0]}_heatmap.png")

base_dir = "logs/episodes/alfworld_test/eval_out_of_distribution/rap_flex/openai/gpt-4o-mini"
'''
# Process all trials with 3ic
for trial in range(1, 6):  # Assuming trials 1-5
    folder_identifiers = [f"trial_{trial}_3ic"]
    reward_heatmap(base_dir, folder_identifiers)

# Process all trials with 6ic
for trial in range(1, 6):  # Assuming trials 1-5
    folder_identifiers = [f"trial_{trial}_6ic"]
    reward_heatmap(base_dir, folder_identifiers)
'''

#reward_heatmap(base_dir, ['6ic_400'])

# Process for intercode_sql
base_dir_sql = "logs/episodes/intercode_sql/test/rap_noplan/openai/gpt-4o-mini"
reward_heatmap(base_dir_sql, ['bird_gold_10_cont_spider_trial'], exclusion_identifiers=['test'], num_tasks=1000, start_task=800)
