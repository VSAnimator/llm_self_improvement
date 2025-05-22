import os
import matplotlib.pyplot as plt
import numpy as np

def load_curve_data(file_path):
    import csv
    x_values = []
    y_values = []
    print(file_path)
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if row and len(row) == 2:  # Ensure we have two values
                try:
                    x, y = float(row[0]), float(row[1])
                    x_values.append(x)
                    y_values.append(y)
                except ValueError:
                    # Skip rows that can't be converted to float
                    continue
    return x_values, y_values

def plot_curves_from_folder(folder_path, title):
    plt.figure(figsize=(4, 3))
    trial_files = []
    all_roc_aucs = []
    all_x_values = []
    # Get all trial folders (e.g., alfworld_pbt_trial_1, alfworld_pbt_trial_2, etc.)
    for trial_dir in sorted(os.listdir(folder_path)):
        trial_dir_path = os.path.join(folder_path, trial_dir)
        if os.path.isdir(trial_dir_path):
            # Find the subfolder with the highest number at the end
            highest_num = -1
            highest_subfolder = None
            
            # Need to read the roc_auc for each subfolder
            roc_aucs = []
            x_values = []
            for subfolder in os.listdir(trial_dir_path):
                subfolder_path = os.path.join(trial_dir_path, subfolder)
                if os.path.isdir(subfolder_path):
                    # Extract the number at the end of the subfolder name
                    parts = subfolder.split('_')
                    # Find the largest digit in the parts
                    largest_digit = -1
                    if not parts:
                        continue
                    for part in parts:
                        if part.isdigit():
                            num = int(part)
                            if num > largest_digit:
                                largest_digit = num
                    num = largest_digit
                    if num > highest_num:
                        highest_num = num
                        highest_subfolder = subfolder_path
                        print(highest_subfolder)
                    x_values.append(num)
                    # Read the roc_auc from the file
                    roc_auc_file = os.path.join(subfolder_path, "roc_auc.txt")
                    with open(roc_auc_file, 'r') as f:
                        roc_aucs.append(float(f.read().split(":")[-1].strip()))
            
            if highest_subfolder:
                curve_file = os.path.join(highest_subfolder, 'calibration_curve.csv')
                if os.path.exists(curve_file):
                    trial_files.append(curve_file)

            # Sort the x_values and roc_aucs by x_values
            x_values, roc_aucs = zip(*sorted(zip(x_values, roc_aucs)))
            all_x_values.append(x_values)
            all_roc_aucs.append(roc_aucs)

    # Now let's plot the roc_auc for all trials
    plt.figure(figsize=(4, 3))
    
    # Calculate and plot the average across all trials if we have multiple trials
    if len(all_x_values) > 1:
        # Find common x values across all trials
        common_x = sorted(set.intersection(*[set(x) for x in all_x_values]))
        
        if common_x:
            # For each common x value, calculate the average ROC AUC
            avg_roc_aucs = []
            for x in common_x:
                values = []
                for i, x_vals in enumerate(all_x_values):
                    if x in x_vals:
                        idx = x_vals.index(x)
                        values.append(all_roc_aucs[i][idx])
                avg_roc_aucs.append(sum(values) / len(values))
            
            # Plot the average as a thicker line
            plt.plot(common_x, avg_roc_aucs, marker='o', linewidth=3, label='Average')
    
    # Plot each trial's ROC AUC curve
    for i, (x_vals, roc_vals) in enumerate(zip(all_x_values, all_roc_aucs)):
        plt.plot(x_vals, roc_vals, linestyle='--', marker='o', label=f'Trial {i+1}', alpha=0.3)

    plt.xlabel("Num training tasks", fontsize=17)
    plt.ylabel("ROC AUC", fontsize=17)
    # Make ticks big too
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tick_params(axis='both', which='minor', labelsize=17)
    #plt.legend()
    plt.grid(False)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(folder_path, f"{title.lower().replace(' ', '_')}_roc_auc.pdf"), bbox_inches='tight')
    
    fig, ax_1 = plt.subplots()
    # Set figure width/height
    fig.set_size_inches(4, 3)

    for idx, trial_file in enumerate(trial_files):
        x, y = load_curve_data(trial_file)
        ax_1.plot(x, y, label=f'Trial {idx+1}', linestyle='-', alpha=0.7)
    
    #plt.title(title)
    ax_1.set_xlabel("Predicted Prob of Task Success", fontsize=12)
    ax_1.set_ylabel("Observed Task Success Rate", fontsize=12)
    # Make the ticks a 
    ax_1.tick_params(axis='both', which='major', labelsize=17)
    ax_1.tick_params(axis='both', which='minor', labelsize=17)
    #ax_1.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))
    
    # Add a diagonal reference line (perfect calibration)
    ax_1.plot([0, 1], [0, 1], 'k--', alpha=0.25)
    ax_1.set_xlim(-0.05, 1.05)
    ax_1.set_ylim(-0.05, 1.05)

    # Remove grid
    ax_1.grid(False)
    # Adjust figure size to accommodate legend
    plt.tight_layout()
    # Add padding to the right for the legend
    plt.subplots_adjust(right=0.85)
    
    # Save a version without the bars
    plt.savefig(os.path.join(folder_path, f"{title.lower().replace(' ', '_')}_plot_no_bars.pdf"), 
                bbox_inches='tight')

    ax_2 = ax_1.twinx()
    for idx, trial_file in enumerate(trial_files):
        # Get folder path
        file_path = os.path.dirname(trial_file)
        # Append the name "calibration_curve.csv" to the file name
        file_path = file_path + "/calibration_counts.csv"
        # Load the data
        y, x = load_curve_data(file_path)
        print(x, y)
        # Normalize the y values
        y = np.array(y) / sum(y)
        # Create a bar plot instead of a line plot
        ax_2.bar(x, y, alpha=0.3, label=f'Trial {idx+1}', width=0.1)

    ax_2.set_ylabel("Fraction of predictions (bars)")
    # Set the y max to be double the max of the bar data
    ax_2.set_ylim(0, 1)

    # Set both x and y axes to range from 0 to 1
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig(os.path.join(folder_path, f"{title.lower().replace(' ', '_')}_plot.pdf"), 
                bbox_inches='tight')
    plt.close()

def main(data_root):
    plot_curves_from_folder(data_root, data_root.split('/')[-1])

if __name__ == "__main__":
    main("calibration_plots/wordcraft")  # replace "data" with your root folder name if different
