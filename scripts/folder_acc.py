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
            # If Reward: 1 is not in the file, and Step 1 is not in the file, then skip and decrement the total count
            elif 'Step 1' not in content:
                total_episodes -= 1
    
    # Calculate success rate
    if total_episodes == 0:
        return 0.0
    
    success_rate = (successful_episodes / total_episodes)
    return success_rate

def calculate_pass_at_k(folder_paths):
    """
    Calculate pass@k metric across multiple folders where files with the same name
    correspond to the same task. Simulates the probability of at least 1 success in k attempts
    for all values of k from 1 to len(folder_paths).
    
    Args:
        folder_paths: List of folder paths containing episode files
        segment_size: Number of most recent tasks to consider
        
    Returns:
        tuple: (dict mapping k values to their corresponding pass@k accuracy, 
               dict mapping k values to standard deviation)
    """
    if not folder_paths:
        return {1: 0.0}, {1: 0.0}
    
    # Get all episode files from the first folder to establish task IDs
    base_folder = folder_paths[0]
    base_files = glob.glob(f'{base_folder}/*.txt')
    base_files = [f for f in base_files if os.path.basename(f).split('.')[0].isdigit()]
    
    if not base_files:
        return {k: 0.0 for k in range(1, len(folder_paths) + 1)}, {k: 0.0 for k in range(1, len(folder_paths) + 1)}
    
    # Sort files numerically
    base_files = sorted(base_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # Extract task IDs from filenames
    task_ids = [os.path.basename(f).split('.')[0] for f in base_files]
    
    total_tasks = len(task_ids)
    results = {}
    
    # For each task, collect success/failure results from each folder
    for task_id in task_ids:
        task_results = []
        
        for folder in folder_paths:
            task_file = os.path.join(folder, f"{task_id}.txt")
            
            if os.path.exists(task_file):
                with open(task_file, 'r') as f:
                    content = f.read()
                    # 1 for success, 0 for failure
                    task_results.append(1 if 'Reward: 1' in content else 0)
            else:
                # If file doesn't exist, count as failure
                task_results.append(0)
        
        # Store results for this task
        results[task_id] = task_results
    
    # Calculate pass@k for each k from 1 to len(folder_paths)
    pass_at_k = {}
    task_probabilities = {}
    
    for k in range(1, len(folder_paths) + 1):
        successful_tasks = 0
        task_probs = []
        
        for task_id, outcomes in results.items():
            # Calculate probability of at least 1 success in k attempts
            # For each task, we're sampling k trials without replacement from all trials
            n_success = sum(outcomes)
            n_trials = len(outcomes)
            
            if n_success == 0:
                # If no successes, probability is 0
                prob_success = 0
            elif k >= n_trials:
                # If k is at least as large as the number of trials and there's at least one success,
                # then probability is 1
                prob_success = 1 if n_success > 0 else 0
            else:
                # Calculate probability of at least 1 success in k attempts
                # 1 - probability of getting no successes
                # = 1 - (ways to choose k failures / ways to choose k trials)
                n_failure = n_trials - n_success
                
                # Handle edge cases to avoid division by zero
                if n_failure < k:
                    prob_success = 1  # Not enough failures to fill all k slots
                else:
                    import math
                    # Calculate binomial coefficients
                    choose_k_failures = math.comb(n_failure, k)
                    choose_k_trials = math.comb(n_trials, k)
                    prob_success = 1 - (choose_k_failures / choose_k_trials)
            
            successful_tasks += prob_success
            task_probs.append(prob_success)
        
        # Calculate average across all tasks
        pass_at_k[k] = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        task_probabilities[k] = task_probs
    
    # Calculate standard error of the mean for each k
    stdev_at_k = {}
    import numpy as np
    for k, probs in task_probabilities.items():
        if probs:
            stdev_at_k[k] = np.std(probs) / np.sqrt(len(probs))
        else:
            stdev_at_k[k] = 0.0
    
    return pass_at_k, stdev_at_k

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate accuracy on the last 100 tasks in a folder.')
    parser.add_argument('folder', type=str, help='Path to the folder containing episode files')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder):
        #print(f"Error: {args.folder} is not a valid directory")
        sys.exit(1)
    
    accuracy = calculate_accuracy(args.folder)
    print(f"{accuracy:.2f}")
