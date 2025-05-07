import os
import matplotlib.pyplot as plt

def load_curve_data(file_path):
    import csv
    x_values = []
    y_values = []
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
    plt.figure(figsize=(8, 6))
    trial_files = []
    # Get all trial folders (e.g., alfworld_pbt_trial_1, alfworld_pbt_trial_2, etc.)
    for trial_dir in sorted(os.listdir(folder_path)):
        trial_dir_path = os.path.join(folder_path, trial_dir)
        if os.path.isdir(trial_dir_path):
            # Find the subfolder with the highest number at the end
            highest_num = -1
            highest_subfolder = None
            
            for subfolder in os.listdir(trial_dir_path):
                subfolder_path = os.path.join(trial_dir_path, subfolder)
                if os.path.isdir(subfolder_path):
                    # Extract the number at the end of the subfolder name
                    parts = subfolder.split('_')
                    if parts and parts[-1].isdigit():
                        num = int(parts[-1])
                        if num > highest_num:
                            highest_num = num
                            highest_subfolder = subfolder_path
                            print(highest_subfolder)
            
            if highest_subfolder:
                curve_file = os.path.join(highest_subfolder, 'calibration_curve.csv')
                if os.path.exists(curve_file):
                    trial_files.append(curve_file)
    
    for idx, trial_file in enumerate(trial_files):
        x, y = load_curve_data(trial_file)
        plt.plot(x, y, marker='o', label=f'Trial {idx+1}')
    
    #plt.title(title)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    # Set both x and y axes to range from 0 to 1
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add a diagonal reference line (perfect calibration)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.savefig(os.path.join(folder_path, f"{title.lower().replace(' ', '_')}_plot.png"))
    plt.close()

def main(data_root):
    plot_curves_from_folder(data_root, data_root.split('/')[-1])

if __name__ == "__main__":
    main("calibration_plots/wordcraft")  # replace "data" with your root folder name if different
