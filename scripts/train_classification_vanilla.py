import os
import argparse
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
from llm_agent.database.learning_db import LearningDB
import json
import random
import re

def success_probability(db_path, test_set_path, output_path):
    # Load the LearningDB
    db = LearningDB(db_path)
    
    # Create dataset
    print("Constructing calibration dataset...")
    data = []
    
    # Get all trajectory IDs
    db.trajectory_cursor.execute("SELECT id, goal, observations, rewards FROM trajectories ORDER BY id")
    rows = db.trajectory_cursor.fetchall()
    
    for row in tqdm(rows):
        trajectory_id, goal, observations_json, rewards_json = row
        
        # Parse observations and rewards
        try:
            observations = json.loads(observations_json)
        except Exception:
            observations = []
        try:
            rewards = json.loads(rewards_json)
        except Exception:
            rewards = []
        
        # Extract first observation
        first_observation = observations[0] if observations else ''
        
        # Concatenate goal and first observation
        input_text = f"Goal: {goal}, Initial observation: {first_observation}"
        
        # Check if there's a 1 in rewards (success)
        success = 1 if 1 in rewards else 0
        
        # Add to dataset
        data.append({
            'input': input_text,
            'success': success
        })

    # We will have to mine a folder to get the test set
    if test_set_path is not None:
        test_data = []
        for file in sorted(os.listdir(test_set_path)):
            #print(f"Processing {file}")
            with open(os.path.join(test_set_path, file), 'r') as f:
                content = f.read()
                
                # Extract goal and initial observation
                goal_match = re.search(r'Goal:(.*?)(?:\n|$)', content, re.DOTALL)
                observation_match = re.search(r'Initial observation:(.*?)(?:\n|$)', content, re.DOTALL)
                
                if goal_match and observation_match:
                    #print(f"Found goal and observation")
                    goal = goal_match.group(1).strip()
                    first_observation = observation_match.group(1).strip()
                    
                    # Concatenate goal and first observation
                    input_text = f"Goal: {goal}, Initial observation: {first_observation}"
                    
                    # Check if Reward 1 is in the content
                    success = 1 if "Reward: 1" in content else 0
                    
                    # Add to dataset
                    test_data.append({
                        'input': input_text,
                        'success': success
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Keep only last 500 examples
    #df = df.tail(500)
    if test_set_path is not None:
        test_df = pd.DataFrame(test_data)
        # Create train and validation splits from the main dataset
        train_size = int(len(df) * 0.8)
        train_df = df.head(train_size)
        val_df = df.tail(len(df) - train_size)
    else:
        # Do the train/val/test split
        # Split off the last 20% of data for testing, and 10% for validation
        test_size = int(len(df) * 0.2)
        val_size = int(len(df) * 0.1)
        test_df = df.tail(test_size)
        val_df = df.iloc[-(test_size + val_size):-test_size]
        train_df = df.head(len(df) - test_size - val_size)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Compute embeddings for entire dataset, then use sklearn to train a classifier
    print("Computing embeddings for the dataset...")
    from sentence_transformers import SentenceTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    # Pull in some other classification models
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.calibration import CalibratedClassifierCV
    import joblib

    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Compute embeddings for all inputs
    train_inputs = train_df['input'].tolist()
    val_inputs = val_df['input'].tolist()
    test_inputs = test_df['input'].tolist()
    
    train_embeddings = model.encode(train_inputs, show_progress_bar=True)
    val_embeddings = model.encode(val_inputs, show_progress_bar=True)
    test_embeddings = model.encode(test_inputs, show_progress_bar=True)
    
    # Prepare the data
    X_train, y_train = train_embeddings, train_df['success']
    X_val, y_val = val_embeddings, val_df['success']
    X_test, y_test = test_embeddings, test_df['success']

    # Combine train and validation data for k-fold cross-validation
    print("Combining train and validation data for k-fold cross-validation...")
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))
    
    # Train a random forest classifier with k-fold cross-validation for calibration
    print("Training random forest classifier with cross-validation...")
    base_classifier = RandomForestClassifier(n_estimators=len(X_combined)//7, random_state=42, class_weight='balanced')
    
    # Use CalibratedClassifierCV with k-fold cross-validation
    print("Using k-fold cross-validation for classifier calibration...")
    classifier = CalibratedClassifierCV(base_classifier, cv=5, method='sigmoid', n_jobs=-1)
    classifier.fit(X_combined, y_combined)
    
    # Evaluate the classifier
    y_pred = classifier.predict(X_test)
    # Also get the probabilities
    y_prob = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classifier accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # Also compute the correlation between the predicted and true values
    corr = np.corrcoef(y_pred, y_test)[0, 1]
    print(f"Correlation between predicted and true values: {corr:.4f}")

    # Correlation with the probabilities
    corr = np.corrcoef(y_prob, y_test)[0, 1]
    print(f"Correlation between predicted probabilities and true values: {corr:.4f}")

    # ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")
    # Log the roc auc to a file
    with open(os.path.join(os.path.dirname(output_path), "roc_auc.txt"), 'w') as f:
        f.write(f"ROC AUC: {roc_auc:.4f}")

    # Save a plot of the ROC curve
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(output_path), "roc_curve.png"))

    # Check the calibration of the classifier
    from sklearn.calibration import CalibrationDisplay
    
    # Create calibration display
    # Define consistent bins for both display and histogram
    n_bins = 10
    disp = CalibrationDisplay.from_estimator(classifier, X_test, y_test, n_bins=n_bins)
    
    # Get the bin edges from the calibration display
    bin_edges = np.linspace(0, 1, n_bins + 1)
    # Use the same bins for the histogram
    counts = np.histogram(disp.y_prob, bins=bin_edges)
    
    # Save the counts to a csv file
    with open(os.path.join(os.path.dirname(output_path), "calibration_counts.csv"), 'w') as f:
        for x, y in zip(counts[0], counts[1]):
            f.write(f"{x},{y}\n")
    
    plt.title('Calibration Curve')
    plt.savefig(os.path.join(os.path.dirname(output_path), "calibration_curve.png"))
    
    # Access and print data from the calibration display object attributes
    print("\nCalibration Data from Display Object:")
    
    # Get data from the line_ attribute (contains the actual curve data)
    line_data = disp.line_.get_data()
    print("Calibration curve data (from line_):")
    print(f"X values: {line_data[0]}")
    print(f"Y values: {line_data[1]}")

    # Save x and y values to a csv file
    with open(os.path.join(os.path.dirname(output_path), "calibration_curve.csv"), 'w') as f:
        for x, y in zip(line_data[0], line_data[1]):
            f.write(f"{x},{y}\n")
    
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a calibration model from LearningDB data')
    parser.add_argument('--db_path', type=str, required=True, help='Path to the learning.db file')
    parser.add_argument('--test_set_path', type=str, default=None, 
                        help='Path to read the test set from')
    parser.add_argument('--output_path', type=str, default='data/calibration_dataset.csv', 
                        help='Path to save the output dataset')
    args = parser.parse_args()
    success_probability(args.db_path, args.test_set_path, args.output_path)
'''
