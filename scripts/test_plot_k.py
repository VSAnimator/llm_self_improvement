import os
import re
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
from folder_acc import calculate_accuracy, calculate_pass_at_k

def save_plot_data_to_csv(x_data, y_data, labels, output_path):
    """
    Save the plot data to a CSV file.
    """
    csv_path = output_path.replace('.png', '.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['k']
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

def plot_pass_at_k(folder_pattern, output_path):
    """
    Plot pass@k success rates for all folders matching the pattern.
    Creates a single plot showing pass@k across all folders.
    """
    # Find all folders matching the pattern
    folders = glob.glob(folder_pattern)
    
    if not folders:
        print(f"No folders found matching pattern: {folder_pattern}")
        return
    
    print(f"Found {len(folders)} folders matching pattern")
    
    # Calculate pass@k for all folders together
    pass_at_k_data = calculate_pass_at_k(folders)
    
    if not pass_at_k_data:
        print("No valid data points found.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot pass@k
    k_values = sorted(pass_at_k_data.keys())
    success_rates = [pass_at_k_data[k] for k in k_values]
    plt.plot(k_values, success_rates, 'o-', linewidth=2, markersize=6, 
             label='All Folders')
    
    plt.xlabel('k')
    plt.ylabel('Pass@k Success Rate')
    plt.title('Pass@k Success Rate')
    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Save data to CSV
    all_y_values = [success_rates]
    labels = ['All Folders']
    
    save_plot_data_to_csv(k_values, all_y_values, labels, output_path)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot pass@k success rates for different training task counts.')
    parser.add_argument('folder_pattern', help='Glob pattern to match folders')
    parser.add_argument('--output', '-o', default='pass_at_k_plot.png', help='Output file path')
    args = parser.parse_args()
    
    plot_pass_at_k(args.folder_pattern, args.output)
