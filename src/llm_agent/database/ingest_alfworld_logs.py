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

    '''
    mode = "best"
    
    # Load best examples from JSON or CSV
    log_files = []
    try:
        with open(f"compare_3ic/{mode}_examples_per_task.json", "r") as f:
            best_examples = json.load(f)
            for task_id, data in best_examples.items():
                if int(task_id) >= 19:
                    run = data["run"]
                    run = run.replace("analysis", "trial")
                    task_id = int(task_id) - 19
                    log_file = f"/mnt/ssd/agent_algo_bench/logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/{run}/{task_id}.txt"
                    log_files.append(log_file)
    except:
        # Fallback to CSV if JSON fails
        with open(f"compare_3ic/{mode}_examples_per_task.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_id = row["task_id"]
                if int(task_id) >= 19:
                    run = row["best_run"]
                    run = run.replace("analysis", "trial")
                    task_id = int(task_id) - 19
                    log_file = f"/mnt/ssd/agent_algo_bench/logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/{run}/{task_id}.txt"
                    log_files.append(log_file)
    db_path = f"./data/alfworld_filtered/alfworld_{mode}_examples/learning.db"
    
    ingest_multiple_logs(log_files, db_path)
    '''

    # Read the text files from the folders for each of the 5 trials and ingest them into the database
    for trial in [5]:
        log_files = []
        for task_id in range(1000):
            log_file = f"./logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/trial_{trial}/{task_id}.txt"
            log_files.append(log_file)
        ingest_multiple_logs(log_files, f"./data/alfworld_filtered/alfworld_trial_{trial}_examples/learning.db")
