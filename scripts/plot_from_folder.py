import os
import glob
import numpy as np
from llm_agent.in_context.alfworld_fewshots import get_task_type
import re

def plot_cumulative_rewards(folder_path, granularity=1, plot_task_type=None):
    """
    Plot the average cumulative reward over tasks.

    Args:
        folder_path: Path to the folder containing episode files
        granularity: Number of tasks to group together for each data point
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

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
                    exit()
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
    
    # Calculate cumulative rewards
    cumulative_rewards = np.cumsum(final_rewards)

    # Group by granularity
    x_points = list(range(granularity, len(final_rewards) + 1, granularity))
    y_points = [cumulative_rewards[i - 1] / i for i in x_points]

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Plot based on task_type parameter
    if plot_task_type == 'substep':
        # Plot success rates for pickup, interaction, and put steps
        if pick_up_successes:
            pick_up_rate = np.cumsum(pick_up_successes) / np.arange(1, len(pick_up_successes) + 1)
            x_pick_up = list(range(granularity, len(pick_up_successes) + 1, granularity))
            if x_pick_up:
                y_pick_up = [pick_up_rate[i-1] for i in x_pick_up]
                plt.plot(x_pick_up, y_pick_up, marker='o', label='Pick up success rate')
        
        if interaction_successes:
            interaction_rate = np.cumsum(interaction_successes) / np.arange(1, len(interaction_successes) + 1)
            x_interaction = list(range(granularity, len(interaction_successes) + 1, granularity))
            if x_interaction:
                y_interaction = [interaction_rate[i-1] for i in x_interaction]
                plt.plot(x_interaction, y_interaction, marker='s', label='Interaction success rate')
        
        if put_successes:
            put_rate = np.cumsum(put_successes) / np.arange(1, len(put_successes) + 1)
            x_put = list(range(granularity, len(put_successes) + 1, granularity))
            if x_put:
                y_put = [put_rate[i-1] for i in x_put]
                plt.plot(x_put, y_put, marker='^', label='Put success rate')
    elif plot_task_type == 'substep_interaction':
        if interaction_subtype_successes:
            for interaction_subtype, successes in interaction_subtype_successes.items():
                interaction_subtype_rate = np.cumsum(successes) / np.arange(1, len(successes) + 1)
                x_interaction_subtype = list(range(granularity, len(successes) + 1, granularity))
                if x_interaction_subtype:
                    y_interaction_subtype = [interaction_subtype_rate[i-1] for i in x_interaction_subtype]
                    plt.plot(x_interaction_subtype, y_interaction_subtype, marker='s', label=f'Interaction {interaction_subtype} success rate')
    elif plot_task_type is None:
        # Plot only overall cumulative reward
        print(f"Plotting overall cumulative reward")
        plt.plot(x_points[8:], y_points[8:], marker='o', label='Overall')
    elif plot_task_type == 'all':
        # Plot cumulative rewards for all task types
        for task_type_name, rewards in task_type_rewards.items():
            cumulative_rewards_by_type = np.cumsum(rewards)
            x_points_by_type = list(range(granularity, len(rewards) + 1, granularity))
            if x_points_by_type:  # Only plot if there are data points
                y_points_by_type = [cumulative_rewards_by_type[i-1] / i for i in x_points_by_type]
                plt.plot(x_points_by_type, y_points_by_type, marker='x', label=task_type_name)
    else:
        # Plot only the specified task type
        if plot_task_type in task_type_rewards:
            rewards = task_type_rewards[plot_task_type]
            cumulative_rewards_by_type = np.cumsum(rewards)
            x_points_by_type = list(range(granularity, len(rewards) + 1, granularity))
            if x_points_by_type:  # Only plot if there are data points
                y_points_by_type = [cumulative_rewards_by_type[i-1] / i for i in x_points_by_type]
                plt.plot(x_points_by_type, y_points_by_type, marker='x', label=plot_task_type)
        else:
            print(f"Warning: Task type '{plot_task_type}' not found in the data")
    
    plt.xlabel('Number of Tasks')
    plt.ylabel('Average Cumulative Reward')
    plt.title('Average Cumulative Reward vs Number of Tasks')
    plt.legend()
    plt.grid(True)

    # Save the plot
    # Get the subfolder name to include in the output filename
    subfolder_name = os.path.basename(os.path.normpath(folder_path))
    parent_folder = os.path.dirname(os.path.normpath(folder_path))
    output_path = os.path.join(parent_folder, f'cumulative_reward_plot_{subfolder_name}_{str(plot_task_type)}.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot cumulative rewards from episode files')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing episode files')
    parser.add_argument('--granularity', type=int, default=1, help='Number of tasks to group together for each data point')
    parser.add_argument('--task_type', type=str, default=None, help='Task type to plot')
    
    args = parser.parse_args()
    plot_cumulative_rewards(args.folder_path, args.granularity, args.task_type)

