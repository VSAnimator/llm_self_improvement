import os
import glob
import numpy as np
from llm_agent.in_context.alfworld_fewshots import get_task_type
import re
import csv

def nonhomogeneous_mean(arrays_list):
    """
    Calculate the mean of arrays that may have different lengths using numpy's nanmean.
    
    Args:
        arrays_list: List of arrays or lists that may have different lengths
        
    Returns:
        A numpy array containing the mean values at each index, where the mean at each
        position is calculated using only the arrays that have values at that position.
    """
    if not arrays_list:
        return np.array([])
    
    # Filter out empty arrays
    non_empty_arrays = [arr for arr in arrays_list if arr]
    if not non_empty_arrays:
        return np.array([])
    
    # Find the maximum length
    max_length = max(len(arr) for arr in non_empty_arrays)
    
    # Create a 2D array filled with NaNs
    padded_arrays = np.full((len(non_empty_arrays), max_length), np.nan)
    
    # Fill in the values from each array
    for i, arr in enumerate(non_empty_arrays):
        padded_arrays[i, :len(arr)] = arr
    
    # Calculate mean along axis 0, ignoring NaNs
    return np.nanmean(padded_arrays, axis=0)

def nonhomogeneous_max(arrays_list):
    """
    Calculate the max of arrays that may have different lengths using numpy's nanmax.
    
    Args:
        arrays_list: List of arrays or lists that may have different lengths
        
    Returns:
        A numpy array containing the max values at each index, where the max at each
        position is calculated using only the arrays that have values at that position.
    """
    if not arrays_list:
        return np.array([])
    
    # Filter out empty arrays
    non_empty_arrays = [arr for arr in arrays_list if arr]
    if not non_empty_arrays:
        return np.array([])
    
    # Find the minimum length
    min_length = min(len(arr) for arr in non_empty_arrays)
    # Create array with only the minimum length elements from each array
    padded_arrays = np.zeros((len(non_empty_arrays), min_length))
    for i, arr in enumerate(non_empty_arrays):
        padded_arrays[i, :min_length] = arr[:min_length]
    
    # Calculate max along axis 0, ignoring NaNs
    return np.nanmax(padded_arrays, axis=0)

def save_plot_data_to_csv(x_data, y_data, labels, output_path):
    """
    Save plot data to a CSV file.
    
    Args:
        x_data: List of x-axis data points for each series
        y_data: List of y-axis data points for each series
        labels: List of labels for each series
        output_path: Path to save the CSV file
    """
    # Create CSV filename from the plot filename
    csv_path = output_path.replace('.png', '.csv')
    
    # Determine the maximum length of any data series
    max_length = max(len(x) for x in x_data)
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row with series labels
        header = ['index']
        for label in labels:
            header.extend([f'{label}_x', f'{label}_y'])
        writer.writerow(header)
        
        # Write data rows
        for i in range(max_length):
            row = [i]
            for j in range(len(x_data)):
                if i < len(x_data[j]):
                    row.extend([x_data[j][i], y_data[j][i]])
                else:
                    row.extend(['', ''])
            writer.writerow(row)
    
    print(f"Plot data saved to {csv_path}")
    return csv_path

def plot_cumulative_rewards(folder_path, granularity=1, plot_task_type=None, cumulative=True, multiple_folders=False, existing_plot=None):
    """
    Plot the average cumulative reward over tasks.

    Args:
        folder_path: Path to the folder containing episode files or a list of folder paths
        granularity: Number of tasks to group together for each data point
        plot_task_type: Type of task to plot
        cumulative: Whether to plot cumulative rewards
        multiple_folders: Whether folder_path contains multiple folders to average
        existing_plot: Optional existing matplotlib figure to add to
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    pass_at_k = None
    # Handle multiple folders case
    if multiple_folders:
        # Get all subfolders if folder_path is a directory containing multiple folders
        if isinstance(folder_path, str):
            folder_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                           if os.path.isdir(os.path.join(folder_path, f))]
        else:
            # If folder_path is already a list of paths
            folder_paths = folder_path
            
        # Collect data from all folders
        all_final_rewards = []
        all_task_type_rewards = {}
        all_pick_up_successes = []
        all_interaction_successes = []
        all_put_successes = []
        all_interaction_subtype_successes = {}

        for folder in folder_paths:
            # Process each folder and collect data
            rewards, task_rewards, pick_ups, interactions, puts, interaction_subtypes = process_folder(
                folder, plot_task_type)
            
            all_final_rewards.append(rewards)
            
            # Merge task type rewards
            for task_type, rewards in task_rewards.items():
                if task_type not in all_task_type_rewards:
                    all_task_type_rewards[task_type] = []
                all_task_type_rewards[task_type].append(rewards)
            
            all_pick_up_successes.append(pick_ups)
            all_interaction_successes.append(interactions)
            all_put_successes.append(puts)
            
            # Merge interaction subtype successes
            for subtype, successes in interaction_subtypes.items():
                if subtype not in all_interaction_subtype_successes:
                    all_interaction_subtype_successes[subtype] = []
                all_interaction_subtype_successes[subtype].append(successes)
        
        # Average the data across folders
        final_rewards = nonhomogeneous_mean(all_final_rewards).tolist()
        # Calculate pass@k - success on any one trial for each task
        pass_at_k = nonhomogeneous_max(all_final_rewards).tolist()

        task_type_rewards = {}
        for task_type, rewards_list in all_task_type_rewards.items():
            if rewards_list:
                task_type_rewards[task_type] = nonhomogeneous_mean(rewards_list).tolist()
        
        pick_up_successes = nonhomogeneous_mean(all_pick_up_successes).tolist()
        interaction_successes = nonhomogeneous_mean(all_interaction_successes).tolist()
        put_successes = nonhomogeneous_mean(all_put_successes).tolist()
        
        interaction_subtype_successes = {}
        for subtype, successes_list in all_interaction_subtype_successes.items():
            if successes_list:
                interaction_subtype_successes[subtype] = nonhomogeneous_mean(successes_list).tolist()
        
        # Use the parent folder name for output
        if isinstance(folder_path, str):
            parent_folder = folder_path
            subfolder_name = "average" + ("_6_ic" if "6_ic" in folder_path else "_3_ic")
        else:
            # If folder_path is a list, use the parent of the first folder
            parent_folder = os.path.dirname(os.path.normpath(folder_paths[0]))
            # Include the last part of the first folder path in the subfolder name
            first_folder_basename = os.path.basename(os.path.normpath(folder_paths[0]))
            subfolder_name = f"average_multiple_folders_{first_folder_basename}" + ("_6_ic" if "6_ic" in folder_paths[0] else "_3_ic")
    else:
        # Original single folder processing
        final_rewards, task_type_rewards, pick_up_successes, interaction_successes, put_successes, interaction_subtype_successes = process_folder(
            folder_path, plot_task_type)
        
        # Get the subfolder name to include in the output filename
        subfolder_name = os.path.basename(os.path.normpath(folder_path))
        parent_folder = os.path.dirname(os.path.normpath(folder_path))
    
    # Calculate cumulative rewards
    if cumulative:
        cumulative_rewards = np.cumsum(final_rewards)
        # Group by granularity
        x_points = list(range(granularity, len(final_rewards) + 1, granularity))
        y_points = [cumulative_rewards[i-1] / i for i in x_points]
        if pass_at_k is not None:
            pass_at_k = [pass_at_k[i-1] for i in x_points]
    else:
        grouped_rewards = [np.mean(final_rewards[i:i+granularity]) for i in range(0, len(final_rewards) - granularity + 1)]
        x_points = list(range(granularity, len(final_rewards) + 1, granularity))
        y_points = grouped_rewards[::granularity]
        if pass_at_k is not None:
            grouped_pass_at_k = [np.mean(pass_at_k[i:i+granularity]) for i in range(0, len(pass_at_k) - granularity + 1)]
            pass_at_k = grouped_pass_at_k[::granularity]
    
    # Use existing plot or create a new one
    if existing_plot is not None:
        plt.figure(existing_plot.number)
    else:
        plt.figure(figsize=(10, 6))
    
    # Lists to store all plot data for CSV export
    all_x_data = []
    all_y_data = []
    all_labels = []
    
    # Plot based on task_type parameter
    if plot_task_type == 'substep':
        # Plot success rates for pickup, interaction, and put steps
        if pick_up_successes:
            if cumulative:
                pick_up_rate = np.cumsum(pick_up_successes) / np.arange(1, len(pick_up_successes) + 1)
            else:
                pick_up_rate = [np.mean(pick_up_successes[i:i+granularity]) for i in range(0, len(pick_up_successes) - granularity + 1)]
            x_pick_up = list(range(granularity, len(pick_up_successes) + 1, granularity))
            if x_pick_up:
                if cumulative:
                    y_pick_up = [pick_up_rate[i-1] for i in x_pick_up]
                else:
                    y_pick_up = pick_up_rate[::granularity]
                plt.plot(x_pick_up, y_pick_up, marker='o', label=f'{subfolder_name} - Pick up success rate')
                all_x_data.append(x_pick_up)
                all_y_data.append(y_pick_up)
                all_labels.append(f'{subfolder_name} - Pick up success rate')
        
        if interaction_successes:
            if cumulative:
                interaction_rate = np.cumsum(interaction_successes) / np.arange(1, len(interaction_successes) + 1)
            else:
                interaction_rate = [np.mean(interaction_successes[i:i+granularity]) for i in range(0, len(interaction_successes) - granularity + 1)]
            x_interaction = list(range(granularity, len(interaction_successes) + 1, granularity))
            if x_interaction:
                if cumulative:
                    y_interaction = [interaction_rate[i-1] for i in x_interaction]
                else:
                    y_interaction = interaction_rate[::granularity]
                plt.plot(x_interaction, y_interaction, marker='s', label=f'{subfolder_name} - Interaction success rate')
                all_x_data.append(x_interaction)
                all_y_data.append(y_interaction)
                all_labels.append(f'{subfolder_name} - Interaction success rate')
        
        if put_successes:
            if cumulative:
                put_rate = np.cumsum(put_successes) / np.arange(1, len(put_successes) + 1)
            else:
                put_rate = [np.mean(put_successes[i:i+granularity]) for i in range(0, len(put_successes) - granularity + 1)]
            x_put = list(range(granularity, len(put_successes) + 1, granularity))
            if x_put:
                if cumulative:
                    y_put = [put_rate[i-1] for i in x_put]
                else:
                    y_put = put_rate[::granularity]
                plt.plot(x_put, y_put, marker='^', label=f'{subfolder_name} - Put success rate')
                all_x_data.append(x_put)
                all_y_data.append(y_put)
                all_labels.append(f'{subfolder_name} - Put success rate')
    elif plot_task_type == 'substep_interaction':
        if interaction_subtype_successes:
            for interaction_subtype, successes in interaction_subtype_successes.items():
                if cumulative:
                    interaction_subtype_rate = np.cumsum(successes) / np.arange(1, len(successes) + 1)
                else:
                    interaction_subtype_rate = [np.mean(successes[i:i+granularity]) for i in range(0, len(successes) - granularity + 1)]
                    x_interaction_subtype = list(range(granularity, len(successes) + 1, granularity))
                if x_interaction_subtype:
                    if cumulative:
                        y_interaction_subtype = [interaction_subtype_rate[i-1] for i in x_interaction_subtype]
                    else:
                        y_interaction_subtype = interaction_subtype_rate[::granularity]
                    plt.plot(x_interaction_subtype, y_interaction_subtype, marker='s', label=f'{subfolder_name} - Interaction {interaction_subtype} success rate')
                    all_x_data.append(x_interaction_subtype)
                    all_y_data.append(y_interaction_subtype)
                    all_labels.append(f'{subfolder_name} - Interaction {interaction_subtype} success rate')
    elif plot_task_type is None:
        # Plot only overall cumulative reward
        print(f"Plotting overall cumulative reward")
        plt.plot(x_points, y_points, marker='o', label=f'{subfolder_name} - Overall')
        all_x_data.append(x_points)
        all_y_data.append(y_points)
        all_labels.append(f'{subfolder_name} - Overall')
    elif plot_task_type == 'pass_at_k':
        plt.plot(x_points[:len(pass_at_k)], pass_at_k, marker='x', label=f'{subfolder_name} - Pass@k')
        all_x_data.append(x_points[:len(pass_at_k)])
        all_y_data.append(pass_at_k)
        all_labels.append(f'{subfolder_name} - Pass@k')
    elif plot_task_type == 'all':
        # Plot cumulative rewards for all task types
        for task_type_name, rewards in task_type_rewards.items():
            if cumulative:
                cumulative_rewards_by_type = np.cumsum(rewards)
            else:
                cumulative_rewards_by_type = [np.mean(rewards[i:i+granularity]) for i in range(0, len(rewards) - granularity + 1)]
            x_points_by_type = list(range(granularity, len(rewards) + 1, granularity))
            if x_points_by_type:  # Only plot if there are data points
                if cumulative:
                    y_points_by_type = [cumulative_rewards_by_type[i-1] / i for i in x_points_by_type]
                else:
                    y_points_by_type = cumulative_rewards_by_type[::granularity]
                plt.plot(x_points_by_type, y_points_by_type, marker='x', label=f'{subfolder_name} - {task_type_name}')
                all_x_data.append(x_points_by_type)
                all_y_data.append(y_points_by_type)
                all_labels.append(f'{subfolder_name} - {task_type_name}')
    else:
        # Plot only the specified task type
        if plot_task_type in task_type_rewards:
            rewards = task_type_rewards[plot_task_type]
            cumulative_rewards_by_type = np.cumsum(rewards)
            x_points_by_type = list(range(granularity, len(rewards) + 1, granularity))
            if x_points_by_type:  # Only plot if there are data points
                y_points_by_type = [cumulative_rewards_by_type[i-1] / i for i in x_points_by_type]
                plt.plot(x_points_by_type, y_points_by_type, marker='x', label=f'{subfolder_name} - {plot_task_type}')
                all_x_data.append(x_points_by_type)
                all_y_data.append(y_points_by_type)
                all_labels.append(f'{subfolder_name} - {plot_task_type}')
        else:
            print(f"Warning: Task type '{plot_task_type}' not found in the data")
    
    plt.xlabel('Number of Tasks')
    plt.ylabel(f'{"Cumulative" if cumulative else "Average"} Reward')
    plt.title(f'{"Cumulative" if cumulative else "Average"} Reward vs Number of Tasks')
    plt.ylim(0, 1)  # Set y-axis to go from 0 to 1
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_path = os.path.join(parent_folder, f'cumulative_reward_plot_{subfolder_name}_{str(plot_task_type)}_{str(cumulative)}.png')
    if existing_plot is None:
        plt.savefig(output_path)
        # Save the plot data to CSV
        if all_x_data:
            save_plot_data_to_csv(all_x_data, all_y_data, all_labels, output_path)
    
    # Only close the plot if we created a new one
    if existing_plot is None:
        plt.close()
    
    print(f"Plot saved to {output_path}")
    return plt.gcf(), output_path

def process_folder(folder_path, plot_task_type=None):
    """
    Process a single folder to extract rewards and success metrics.
    
    Args:
        folder_path: Path to the folder containing episode files
        plot_task_type: Type of task to plot
        
    Returns:
        Tuple of (final_rewards, task_type_rewards, pick_up_successes, 
                 interaction_successes, put_successes, interaction_subtype_successes)
    """
    import os
    import re
    import numpy as np
    from llm_agent.in_context.alfworld_fewshots import get_task_type
    
    # Get all episode files in the folder
    episode_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".txt")
    ]

    # Get final rewards for all episodes
    final_rewards = []
    task_type_rewards = {}
    pick_up_successes = []
    interaction_successes = []
    put_successes = []
    interaction_subtype_successes = {}
    
    for episode_file in sorted(episode_files, key=lambda x: int(os.path.basename(x).split('.')[0]) if os.path.basename(x).split('.')[0].isdigit() else float('inf')):
        try:
            with open(episode_file, 'r') as f:
                content = f.read()
                goal = content.split('Goal:')[1].split('\n')[0].strip()
                task_type = get_task_type(goal)
                # Get final reward in file
                final_reward = content.split('Reward: ')[-1].split('\n')[0]
                # Log invalid rewards
                if not final_reward.replace('.', '', 1).isdigit():
                    print(f"Invalid reward {final_reward} for episode {episode_file}")
                    final_reward = "0"
                # Cast to float if valid, otherwise 0
                final_reward = float(final_reward) if final_reward.replace('.', '', 1).isdigit() else 0
                final_rewards.append(final_reward)
                if task_type not in task_type_rewards:
                    task_type_rewards[task_type] = []
                task_type_rewards[task_type].append(final_reward)

                # Also figure out success at 1) search, 2) take, 3) interact, 4) put
                put_loc = goal.split(' ')[-1]
                # Find the nouns in the goal
                all_words = re.findall(r'\b\w+\b', goal)
                # Filter to keep only the three most important nouns (typically object, action, and location)
                nouns = []
                for word in all_words:
                    if word.lower() not in ['your', 'task', 'is', 'to', 'the', 'a', 'an', 'and', 'in', 'on', 'some', 'it', 'them', 'at', 'under', 'find', 'with']:
                        nouns.append(word)
                has_put = 1 if "put" in nouns else 0
                # Remove "put" from nouns if it exists
                if "put" in nouns:
                    nouns.remove("put")
                interaction_words = ['examine', 'look', 'heat', 'hot', 'cool', 'cold', 'clean']
                interaction = None
                for word in nouns:
                    if word in interaction_words:
                        interaction = word
                        break
                for word in interaction_words:
                    if word in nouns:
                        nouns.remove(word)
                # Replace specific interaction words with their standardized versions
                if interaction == "examine" or interaction == "look":
                    interaction = None # Let's not handle this for now
                elif interaction == "hot":
                    interaction = "heat"
                elif interaction == "cold":
                    interaction = "cool"
                two_items = 1 if "two" in nouns else 0
                if "two" in nouns:
                    nouns.remove("two")
                interaction_words = ['examine', 'heat', 'cool', 'clean']
                pickup_words = ['take']
                # Now let's see if all the actions happened
                # Was the first noun picked up?
                pick_up_success = 0
                if "You pick up the " + nouns[0] in content:
                    pick_up_success = 1
                # Was the interaction done?
                interaction_success = None
                if interaction is not None and pick_up_success == 1:
                    if "You " + interaction + " the " + nouns[0] in content:
                        interaction_success = 1
                        interaction_subtype = interaction
                        if interaction_subtype not in interaction_subtype_successes:
                            interaction_subtype_successes[interaction_subtype] = []
                        interaction_subtype_successes[interaction_subtype].append(1)
                    else:
                        interaction_success = 0
                        interaction_subtype = interaction
                        if interaction_subtype not in interaction_subtype_successes:
                            interaction_subtype_successes[interaction_subtype] = []
                        interaction_subtype_successes[interaction_subtype].append(0)
                # Was the first noun put in/on the second noun?
                put_success = 0 if has_put and pick_up_success == 1 and (interaction_success is None or interaction_success == 1) else None
                if "You put the " + nouns[0] in content and put_success is not None:
                    key_line = content.split("You put the " + nouns[0])[1].split('\n')[0] # + " in/on the " + nouns[1] in content:
                    if "in/on the " + nouns[1] in key_line:
                        put_success = 1

                if pick_up_success is not None:
                    pick_up_successes.append(pick_up_success)
                if interaction_success is not None:
                    interaction_successes.append(interaction_success)
                if put_success is not None:
                    put_successes.append(put_success)
        except Exception as e:
            print(f"Error processing {episode_file}: {e}")
            continue
    
    return final_rewards, task_type_rewards, pick_up_successes, interaction_successes, put_successes, interaction_subtype_successes


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot cumulative rewards from episode files')
    parser.add_argument('folder_path', type=str, nargs='+', help='Path to the folder(s) containing episode files')
    parser.add_argument('--granularity', type=int, default=1, help='Number of tasks to group together for each data point')
    parser.add_argument('--task_type', type=str, default=None, help='Task type to plot')
    parser.add_argument('--cumulative', action='store_true', help='Plot cumulative rewards')
    parser.add_argument('--multiple_folders', action='store_true', help='Average results from multiple folders')

    args = parser.parse_args()
    # If multiple folders are provided or multiple_folders flag is set
    multiple_folders = args.multiple_folders or len(args.folder_path) > 1
    # Always pass the multiple_folders flag and handle folder_path appropriately
    if len(args.folder_path) == 1:
        folder_path = args.folder_path[0]
    else:
        folder_path = args.folder_path
    
    if len(args.folder_path) > 1 and args.task_type != "pass_at_k":
        # Loop through each folder and plot them together on the same figure
        import matplotlib.pyplot as plt
        
        # Create a single figure that will be shared across all folders
        fig = plt.figure(figsize=(12, 8))
        
        # Lists to store all plot data for CSV export
        all_x_data = []
        all_y_data = []
        all_labels = []
        
        # Plot each folder individually but on the same figure
        for folder in args.folder_path:
            print(f"Processing folder: {folder}")
            # Pass the existing figure to add to it
            fig, _ = plot_cumulative_rewards(folder, args.granularity, args.task_type, 
                                          args.cumulative, False, existing_plot=fig)
        
        # Save the combined plot with a descriptive filename
        import os
        output_dir = os.path.dirname(os.path.normpath(args.folder_path[0]))
        task_type_str = str(args.task_type) if args.task_type else "all_tasks"
        cumulative_str = "cumulative" if args.cumulative else "average"
        # Identify if any of the folders contain 3_ic or 6_ic in their path
        ic_type = "_6_ic" if any("6_ic" in folder for folder in args.folder_path) else "_3_ic"
        # Get the name of the first folder to include in the output filename
        first_folder_name = os.path.basename(os.path.normpath(args.folder_path[0]))
        output_path = os.path.join(output_dir, f'combined_plot_{task_type_str}_{cumulative_str}{ic_type}_{first_folder_name}.png')
        plt.savefig(output_path)
        
        # Extract data from the plot for CSV export
        for line in plt.gca().get_lines():
            all_x_data.append(line.get_xdata())
            all_y_data.append(line.get_ydata())
            all_labels.append(line.get_label())
        
        # Save the plot data to CSV
        if all_x_data:
            save_plot_data_to_csv(all_x_data, all_y_data, all_labels, output_path)
        
        plt.close()
        print(f"Combined plot saved to {output_path}")
        
    plot_cumulative_rewards(folder_path, args.granularity, args.task_type, args.cumulative, multiple_folders)
