import os
from llm_agent.database.learning_db import LearningDB
from llm_agent.env.base_env import Observation, Action
from llm_agent.in_context.alfworld_fewshots import get_task_type

def parse_alfworld_log(log_content):
    """
    Parse the Alfworld log file and extract trajectory information.
    
    Args:
        log_content (str): Content of the log file
    
    Returns:
        tuple: Extracted trajectory components (goal, plan, obs_list, reasoning_list, act_list, rewards)
    """
    # Extract goal
    goal_match = log_content.split("Goal: ")[1].split("\n")[0]
    category = get_task_type(goal_match)
    
    # Extract plan
    plan_match = log_content.split("Plan: ")[1].split("\n")[0] if "Plan: " in log_content else None
    
    # Extract observations, actions, reasoning, and rewards
    obs_list = []
    act_list = []
    reasoning_list = []
    rewards = []
    
    lines = log_content.split("\n")
    for line in lines:
        if line.startswith("Initial observation: "):
            obs_list.append(line[22:].strip())
        elif line.startswith("Obs: "):
            obs_reward = line[5:].strip()
            obs_list.append(obs_reward.split("Reward: ")[0].strip()[:-1])
            rewards.append(int(obs_reward.split("Reward: ")[1].strip()))
        elif line.startswith("Selected action: "):
            wrapped_action = line.split("Selected action: ")[1].strip()
            # Now remove "Action(text=" and ")"
            act_text = wrapped_action.split("Action(text='")[1].split("')")[0].strip()
            act_list.append(act_text)
        elif line.startswith("Reasoning: "):
            reasoning_list.append(line[11:].strip())
        elif line.startswith("Reward: "):
            rewards.append(int(line.split("Reward: ")[1].strip()))
    
    return goal_match, category, plan_match, obs_list, reasoning_list, act_list, rewards

def ingest_alfworld_log(log_file_path, db):
    """
    Ingest Alfworld trajectory log into the learning database.
    
    Args:
        log_file_path (str): Path to the log file
        db_path (str): Path to the SQLite database
    """
    # Read log file
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    # Parse log content
    goal, category, plan, obs_list, reasoning_list, act_list, rewards = parse_alfworld_log(log_content)
    
    # Validate lists match in length
    # Validate that observations list is one longer than actions list
    if len(obs_list) != len(act_list) + 1:
        raise ValueError(f"Observation list should be one longer than action list. Found {len(obs_list)} observations and {len(act_list)} actions.")
    
    # Validate that reasoning list matches action list in length
    if len(reasoning_list) != len(act_list):
        raise ValueError(f"Reasoning list length ({len(reasoning_list)}) doesn't match action list length ({len(act_list)}). This may cause issues.")
    
    # Ensure rewards matches actions length
    if not rewards:
        rewards = [0] * (len(act_list) - 1) + [1]
    
    # Convert obs_list and act_list to Observation and Action objects
    obs_list = [Observation(o) for o in obs_list]
    act_list = [Action(a) for a in act_list]
    
    # Store trajectory
    environment_id = os.path.basename(log_file_path).replace('.txt', '')
    
    db.store_episode(
        environment_id=environment_id,
        goal=goal,
        category=category,
        observations=obs_list,
        reasoning=reasoning_list,
        actions=act_list,
        rewards=rewards,
        plan=plan,
        reflection=None,
        summary=None,
        nosave=True
    )

def ingest_multiple_logs(log_files, db_path):
    """
    Ingest multiple Alfworld log files from a directory.
    
    Args:
        log_directory (str): Directory containing log files
        db_path (str): Path to the SQLite database
    """
    # Create the database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Initialize database
    db = LearningDB(db_path)
    
    # Iterate through log files
    for log_file_path in log_files:
        try:
            print(f"Ingesting log file: {log_file_path}")
            ingest_alfworld_log(log_file_path, db)
        except Exception as e:
            print(f"Error ingesting {log_file_path}: {e}")

if __name__ == "__main__":
    # Example usage
    import json
    import csv

    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Ingest ALFWorld logs into a database')
    parser.add_argument('--mode', type=str, default='best', help='Mode for selecting examples (e.g., best, all)')
    parser.add_argument('--json_file', type=str, default='compare_3ic/{mode}_examples_per_task.json', 
                        help='Path to JSON file with examples (use {mode} for substitution)')
    parser.add_argument('--log_file_base', type=str, default='logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/',
                        help='Path to base of log files (use {mode} for substitution)')
    parser.add_argument('--db_path', type=str, default='./data/alfworld_filtered/alfworld_{mode}_examples/learning.db',
                        help='Path to output database (use {mode} for substitution)')
    parser.add_argument('--task_offset', type=int, default=18,
                        help='Offset for task IDs (default is 18)')
    
    args = parser.parse_args()
    
    # Format paths with the mode
    mode = args.mode
    json_file = args.json_file.format(mode=mode)
    log_file_base = args.log_file_base.format(mode=mode)
    db_path = args.db_path.format(mode=mode)
    task_offset = args.task_offset + 1

    # Load best examples from JSON or CSV
    log_files = []
    with open(json_file, "r") as f:
        best_examples = json.load(f)
        for task_id, data in best_examples.items():
            if int(task_id) >= task_offset:
                run = data["run"]
                run = run.replace("analysis", "trial")
                task_id = int(task_id) - task_offset
                log_file = f"{log_file_base}/{run}/{task_id}.txt"
                log_files.append(log_file)
    ingest_multiple_logs(log_files, db_path)
