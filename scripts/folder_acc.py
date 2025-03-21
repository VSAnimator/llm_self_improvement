import os
import glob
import argparse
import sys

def calculate_accuracy(folder_path, segment_size=100):
    """
    Calculate the accuracy on the last 100 tasks in the given folder.
    Tasks are identified by episode files with numeric names.
    """
    # Get all episode files
    episode_files = glob.glob(f'{folder_path}/*.txt')

    # Filter files which can be cast to int
    episode_files = [f for f in episode_files if os.path.basename(f).split('.')[0].isdigit()]
    
    # If no valid episode files found
    if not episode_files:
        #print(f"No valid episode files found in {folder_path}")
        return 0.0
    
    # Sort files numerically
    episode_files = sorted(episode_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # Get the last 100 files (or all if less than 100)
    last_files = episode_files[-segment_size:] if len(episode_files) > segment_size else episode_files
    
    total_episodes = len(last_files)
    successful_episodes = 0
    
    # Process each episode file
    for episode_file in last_files:
        with open(episode_file, 'r') as f:
            content = f.read()
            # Look for final reward of 1
            if 'Reward: 1' in content:
                successful_episodes += 1
            # If Reward: 1 is not in the file, and Step 4 is not in the file, then skip and decrement the total count
            elif 'Step 4' not in content:
                total_episodes -= 1
    
    # Calculate success rate
    if total_episodes == 0:
        return 0.0
    
    success_rate = (successful_episodes / total_episodes)
    return success_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate accuracy on the last 100 tasks in a folder.')
    parser.add_argument('folder', type=str, help='Path to the folder containing episode files')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder):
        #print(f"Error: {args.folder} is not a valid directory")
        sys.exit(1)
    
    accuracy = calculate_accuracy(args.folder)
    print(f"{accuracy:.2f}")
