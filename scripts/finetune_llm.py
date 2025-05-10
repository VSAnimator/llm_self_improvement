import os
import argparse
import json
import pandas as pd
from tqdm import tqdm
from llm_agent.database.learning_db import LearningDB
import openai
from openai import OpenAI
import numpy as np
import time

def main():
    parser = argparse.ArgumentParser(description='Finetune GPT-4o-mini with LearningDB data for a ReAct agent')
    parser.add_argument('--db_path', type=str, required=True, help='Path to the learning.db file')
    parser.add_argument('--output_dir', type=str, default='data/finetune', help='Directory to save the output files')
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini-2024-07-18', help='Base model to finetune')
    parser.add_argument('--tag_name', type=str, default='react_agent', help='Tag name for the fine-tuning job')
    args = parser.parse_args()
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the LearningDB
    print(f"Loading database from {args.db_path}")
    db = LearningDB(args.db_path)
    
    # Get all trajectory IDs
    query = "SELECT id, goal, observations, reasoning, actions, rewards FROM trajectories WHERE 1 IN (SELECT json_each.value FROM json_each(rewards))"
    
    db.trajectory_cursor.execute(query)
    rows = db.trajectory_cursor.fetchall()
    
    print(f"Processing {len(rows)} trajectories...")
    
    # Prepare data for fine-tuning
    finetune_data = []
    
    system_prompt = """You are a ReAct agent that helps users accomplish tasks. 
Given a goal, you will receive observations about the environment and respond with your reasoning and actions.
For each observation, first think through the problem step by step (Thought), then decide on an action (Action).
Your actions should be clear, concise, and directly executable in the environment."""
    
    for row in tqdm(rows):
        trajectory_id, goal, observations_json, reasoning_json, action_json, reward_json = row
        try:
            observations = json.loads(observations_json)
            reasoning = json.loads(reasoning_json)
            action = json.loads(action_json)
            reward = json.loads(reward_json)
            
            # Skip if any of the lists are empty
            if not observations or not reasoning or not action:
                continue
                
            # Create conversation for this trajectory
            messages = []
            
            # Add system message
            messages.append({
                "role": "system",
                "content": system_prompt
            })
            
            # Add goal as the first user message
            messages.append({
                "role": "user",
                "content": f"Goal: {goal}\nInitial observation: {observations[0]}"
            })
            
            # Process each step in the trajectory
            for i in range(len(reasoning)):
                # Assistant's response with reasoning and action
                assistant_content = f"Thought: {reasoning[i]}\nAction: {action[i]}"
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                
                # Add next observation if available
                if i + 1 < len(observations):
                    messages.append({
                        "role": "user",
                        "content": f"Observation: {observations[i+1]}"
                    })

            # If the last message is a user message, delete it
            if messages[-1]['role'] == 'user':
                messages.pop()
            
            # Add this conversation to the fine-tuning data
            finetune_data.append({
                "messages": messages
            })
            
        except Exception as e:
            print(f"Error processing trajectory {trajectory_id}: {e}")
            continue
    
    # If empty, skip
    if len(finetune_data) == 0:
        print("No data to fine-tune on")
        return
    
    # Save the fine-tuning data to a JSONL file
    finetune_file = os.path.join(args.output_dir, "finetune_data.jsonl")
    with open(finetune_file, 'w') as f:
        for item in finetune_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {len(finetune_data)} conversations to {finetune_file}")
    
    # Upload the file for fine-tuning
    try:
        print("Uploading file for fine-tuning...")
        response = client.files.create(
            file=open(finetune_file, "rb"),
            purpose="fine-tune"
        )
        file_id = response.id
        print(f"File uploaded with ID: {file_id}")
        
        # Create fine-tuning job
        print(f"Creating fine-tuning job for model {args.model_name}...")
        job = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=args.model_name,
            suffix=f"{args.tag_name}_{int(time.time())}"
        )
        
        print(f"Fine-tuning job created with ID: {job.id}")
        print(f"You can monitor the job status with: openai api fine_tuning.jobs.get -i {job.id}")
        
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
    
    # Print statistics
    print(f"\nFine-tuning dataset statistics:")
    print(f"Total conversations: {len(finetune_data)}")
    
    # Calculate average turns per conversation
    avg_turns = np.mean([len(conv['messages']) // 2 for conv in finetune_data])
    print(f"Average turns per conversation: {avg_turns:.2f}")
    
    # Calculate total tokens (rough estimate)
    total_tokens = 0
    for conv in finetune_data:
        for msg in conv['messages']:
            total_tokens += len(msg['content'].split())
    print(f"Estimated total tokens: {total_tokens}")

if __name__ == "__main__":
    main()
