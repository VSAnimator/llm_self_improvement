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

def main():
    parser = argparse.ArgumentParser(description='Train a calibration model from LearningDB data')
    parser.add_argument('--db_path', type=str, required=True, help='Path to the learning.db file')
    parser.add_argument('--test_set_path', type=str, default=None, 
                        help='Path to read the test set from')
    parser.add_argument('--output_path', type=str, default='data/calibration_dataset.csv', 
                        help='Path to save the output dataset')
    args = parser.parse_args()
    
    # Load the LearningDB
    print(f"Loading database from {args.db_path}")
    db = LearningDB(args.db_path)
    
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
    if args.test_set_path is not None:
        test_set_path = args.test_set_path
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
    if args.test_set_path is not None:
        test_df = pd.DataFrame(test_data)
    else:
        # Do the train/test split
        # Split off the last 20% of data for testing
        test_size = int(len(df) * 0.2)
        test_df = df.tail(test_size)
        df = df.head(len(df) - test_size)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save dataset
    print(f"Saving dataset to {args.output_path}")
    df.to_csv(args.output_path, index=False)
    
    # Print statistics
    success_rate = df['success'].mean()
    print(f"Dataset created with {len(df)} examples")
    print(f"Success rate: {success_rate:.2f}")
    print(f"Successful examples: {df['success'].sum()}")
    print(f"Failed examples: {len(df) - df['success'].sum()}")
    
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
    import joblib

    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Compute embeddings for all inputs
    inputs = df['input'].tolist()
    embeddings = model.encode(inputs, show_progress_bar=True)
    test_inputs = test_df['input'].tolist()
    test_embeddings = model.encode(test_inputs, show_progress_bar=True)
    
    # Split the data into training and testing sets
    X_train, y_train = embeddings, df['success']
    X_test, y_test = test_embeddings, test_df['success']
    
    # Train a logistic regression classifier
    print("Training logistic regression classifier...")
    #classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
    #classifier.fit(X_train, y_train)

    # Also train a random forest classifier
    print("Training random forest classifier...")
    classifier = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
    classifier.fit(X_train, y_train)

    # Boosted tree
    print("Training boosted tree classifier...")
    #classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    #classifier.fit(X_train, y_train)

    # Also train a support vector machine classifier
    print("Training support vector machine classifier...")
    #classifier = SVC(kernel='rbf', probability=True)
    #classifier.fit(X_train, y_train)
    
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
    plt.savefig(os.path.join(os.path.dirname(args.output_path), "roc_curve.png"))

    # Compute the flipped ROC curve
    fpr_flipped, tpr_flipped, thresholds_flipped = roc_curve(1 - y_test, 1 - y_prob)
    roc_auc_flipped = auc(fpr_flipped, tpr_flipped)
    print(f"Flipped ROC AUC: {roc_auc_flipped:.4f}")

    # Save the flipped ROC curve
    plt.figure()
    plt.plot(fpr_flipped, tpr_flipped, label='Flipped ROC curve (area = %0.2f)' % roc_auc_flipped)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(args.output_path), "flipped_roc_curve.png"))

    # Save the precision-recall curve
    from sklearn.metrics import precision_recall_curve
    # For this curve flip the probabilities and the labels--we want to find failures
    y_prob_flipped = 1 - y_prob
    y_test_flipped = 1 - y_test
    precision, recall, thresholds = precision_recall_curve(y_test_flipped, y_prob_flipped)
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(args.output_path), "precision_recall_curve.png"))

    # Check the calibration of the classifier
    from sklearn.calibration import CalibrationDisplay
    CalibrationDisplay.from_estimator(classifier, X_test, y_test, n_bins=10)
    plt.savefig(os.path.join(os.path.dirname(args.output_path), "calibration_curve.png"))

    # Save the model and encoder
    model_output_path = os.path.join(os.path.dirname(args.output_path), "classification_model")
    os.makedirs(model_output_path, exist_ok=True)
    
    # Save the classifier
    joblib.dump(classifier, os.path.join(model_output_path, "classifier.joblib"))
    
    # Save the sentence transformer model
    model.save(os.path.join(model_output_path, "sentence_transformer"))
    
    print(f"Classification model saved to {model_output_path}")
    
    

if __name__ == "__main__":
    main()
