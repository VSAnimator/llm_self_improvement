import os
import re
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
from folder_acc import calculate_accuracy

def extract_folder_info(folder_path, regex_pattern):
    """
    Extract trial ID and number of training tasks from a folder path using the provided regex pattern.
    """
    folder_path = folder_path.split('/')[-1]
    print(f"Extracting folder info from {folder_path}")
    print(f"Regex pattern: {regex_pattern}")
    match = re.search(regex_pattern, folder_path)
    num_groups = len(match.groups()) if match else 0
    if num_groups == 2:
        return int(match.group(1)), int(match.group(2))
    elif num_groups == 1:
        return 1, int(match.group(1))
    else:
        return None, None

def save_plot_data_to_csv(x_data, y_data, labels, output_path):
    """
    Save the plot data to a CSV file.
    """
    csv_path = output_path.replace('.png', '.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Number of Training Tasks']
        header.extend(labels)
        writer.writerow(header)
        
        # Create rows with data for each x value
        for i, x in enumerate(x_data):
            row = [x]
            for series in y_data:
                if i < len(series):
                    row.append(series[i])
                else:
                    row.append('')
            writer.writerow(row)
    print(f"Plot data saved to {csv_path}")

def plot_success_rates(folder_pattern, regex_pattern, output_path, segment_size=100, reverse_args=False):
    """
    Plot task success rates against folder labels for folders matching the pattern.
    Shows individual trials as light dashed lines and the average as a darker solid line.
    """
    # Find all folders matching the pattern
    folders = glob.glob(folder_pattern)
    
    if not folders:
        print(f"No folders found matching pattern: {folder_pattern}")
        return
    
    # Extract trial IDs and training task counts, and calculate success rates
    trial_data = {}  # Dictionary to organize data by trial ID
    training_tasks_set = set()  # Set to track unique training task counts
    
    for folder in folders:
        trial_id, training_tasks = extract_folder_info(folder, regex_pattern)
        if reverse_args:
            trial_id, training_tasks = training_tasks, trial_id
        if trial_id is not None and training_tasks is not None:
            success_rate = calculate_accuracy(folder, segment_size)
            if trial_id not in trial_data:
                trial_data[trial_id] = []
            trial_data[trial_id].append((training_tasks, success_rate))
            training_tasks_set.add(training_tasks)
    
    if not trial_data:
        print("No valid data points found. Check your regex pattern.")
        return
    
    # Sort training tasks
    training_tasks_list = sorted(list(training_tasks_set))
    
    # Create a dictionary to store success rates by training task count
    success_rates_by_task = {task_count: [] for task_count in training_tasks_list}
    
    # Organize data for plotting
    trial_plots = {}
    for trial_id, data_points in trial_data.items():
        # Sort data points by training tasks
        data_points.sort(key=lambda x: x[0])
        x_values = [point[0] for point in data_points]
        y_values = [point[1] for point in data_points]
        trial_plots[trial_id] = (x_values, y_values)
        
        # Add success rates to the dictionary for averaging
        for task_count, success_rate in data_points:
            success_rates_by_task[task_count].append(success_rate)
    
    # Calculate average success rates
    avg_x_values = training_tasks_list
    avg_y_values = [np.mean(success_rates_by_task[task_count]) for task_count in training_tasks_list]
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot individual trials with lighter, dashed lines
    for trial_id, (x_values, y_values) in trial_plots.items():
        plt.plot(x_values, y_values, '--', alpha=0.5, linewidth=1, label=f'Trial {trial_id}')
    
    # Plot the average with a darker, solid line
    plt.plot(avg_x_values, avg_y_values, 'o-', color='darkblue', linewidth=2.5, markersize=8, label='Average')
    
    plt.xlabel('Number of Training Tasks')
    plt.ylabel('Task Success Rate')
    plt.title('Task Success Rate vs. Number of Training Tasks')
    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Save data to CSV
    all_y_values = [y_values for _, y_values in trial_plots.values()]
    all_y_values.append(avg_y_values)
    labels = [f'Trial {trial_id}' for trial_id in trial_plots.keys()]
    labels.append('Average')
    save_plot_data_to_csv(avg_x_values, all_y_values, labels, output_path)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot task success rates against folder labels.')
    parser.add_argument('folder_pattern', help='Glob pattern to match folders')
    parser.add_argument('regex_pattern', help='Regex pattern with two capture groups: trial ID and number of training tasks')
    parser.add_argument('--output', '-o', default='success_rate_plot.png', help='Output file path')
    parser.add_argument('--segment-size', '-s', type=int, default=1000, 
                        help='Number of most recent tasks to consider for accuracy calculation')
    parser.add_argument('--reverse_args', '-r', action='store_true', help='Reverse the order of the regex pattern')
    args = parser.parse_args()
    
    plot_success_rates(args.folder_pattern, args.regex_pattern, args.output, args.segment_size, args.reverse_args)
