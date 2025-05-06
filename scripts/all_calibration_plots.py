import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from train_classification_vanilla import success_probability

def generate_all_calibration_plots(train_dbs, test_folders, output_base_dir):
    """
    Generate calibration plots for each train db and test folder pair.
    
    Args:
        train_dbs (list): List of paths to training database files
        test_folders (list): List of paths to test folders
        output_base_dir (str): Base directory to save outputs
    """
    # Ensure the output base directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each train db and test folder pair
    for i, (db_path, test_folder) in enumerate(zip(train_dbs, test_folders)):
        # Create a folder name based on the db name
        db_name = os.path.basename(db_path).replace('.db', '')
        test_folder_name = os.path.basename(test_folder)
        output_dir = os.path.join(output_base_dir, test_folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output path for this specific run
        output_path = os.path.join(output_dir, "calibration_dataset.csv")
        
        print(f"Processing {db_name} ({i+1}/{len(train_dbs)})...")
        
        # Call the success_probability function to generate calibration plots
        success_probability(db_path, test_folder, output_path)
        
        print(f"Completed processing {db_name}")

if __name__ == "__main__":
    train_dbs = []
    test_folders = []
    output_base_dir = "calibration_plots"

    # Wordcraft db path "data/wordcraft/depth2_humanic_train_1_backups/*/learning.db/learning.db" for * in 100 200 400 1000 2000 3700
    # Test folder path "logs/episodes/wordcraft/test/rap_noplan/openai/gpt-4o-mini/wordcraft_depth_2_humanic_test_1_10ic_*"
    for trial in range(1, 6):  # Loop through trials 1 to 5
        for i in [100, 200, 400, 1000, 2000, 3700]:
            train_dbs.append(f"data/wordcraft/depth2_humanic_train_{trial}_backups/{i}/learning.db/learning.db")
            test_folders.append(f"logs/episodes/wordcraft/test/rap_noplan/openai/gpt-4o-mini/wordcraft_depth_2_humanic_test_{trial}_10ic_{i}")
        
        # Create trial-specific output directory
        trial_output_dir = f"{output_base_dir}/wordcraft_trial_{trial}"
        generate_all_calibration_plots(train_dbs, test_folders, trial_output_dir)
        
        # Clear lists for next trial
        train_dbs = []
        test_folders = []
