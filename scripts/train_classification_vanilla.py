import os
import argparse
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
from llm_agent.database.learning_db import LearningDB
import json
import random

def main():
    parser = argparse.ArgumentParser(description='Train a calibration model from LearningDB data')
    parser.add_argument('--db_path', type=str, required=True, help='Path to the learning.db file')
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
    db.trajectory_cursor.execute("SELECT id, goal, observations, rewards FROM trajectories")
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
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
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
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, df['success'], test_size=0.2, random_state=42, stratify=df['success']
    )
    
    # Train a logistic regression classifier
    print("Training logistic regression classifier...")
    #classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
    #classifier.fit(X_train, y_train)

    # Also train a random forest classifier
    print("Training random forest classifier...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
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
