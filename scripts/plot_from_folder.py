import os
import glob
import numpy as np

def plot_cumulative_rewards(folder_path, granularity=1):
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
    episode_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Get final rewards for all episodes
    final_rewards = []
    for episode_file in sorted(episode_files, key=lambda x: int(os.path.basename(x).split('.')[0]) if os.path.basename(x).split('.')[0].isdigit() else float('inf')):
        with open(episode_file, 'r') as f:
            content = f.read()
            # Get final reward in file
            final_reward = content.split('Reward: ')[-1].split('\n')[0]
            # Cast to float if valid, otherwise 0
            final_reward = float(final_reward) if final_reward.replace('.', '', 1).isdigit() else 0
            final_rewards.append(final_reward)
    
    # Calculate cumulative rewards
    cumulative_rewards = np.cumsum(final_rewards)
    
    # Group by granularity
    x_points = list(range(granularity, len(final_rewards) + 1, granularity))
    y_points = [cumulative_rewards[i-1] / i for i in x_points]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_points, y_points, marker='o')
    plt.xlabel('Number of Tasks')
    plt.ylabel('Average Cumulative Reward')
    plt.title('Average Cumulative Reward vs Number of Tasks')
    plt.grid(True)
    
    # Save the plot
    output_path = os.path.join(folder_path, 'cumulative_reward_plot.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Plot saved to {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot cumulative rewards from episode files')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing episode files')
    parser.add_argument('--granularity', type=int, default=1, help='Number of tasks to group together for each data point')
    
    args = parser.parse_args()
    plot_cumulative_rewards(args.folder_path, args.granularity)


