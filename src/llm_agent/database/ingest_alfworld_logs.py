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
    
    return
    # Validate lists match in length
    min_length = min(len(obs_list), len(act_list), len(reasoning_list))
    obs_list = obs_list[:min_length]
    act_list = act_list[:min_length]
    reasoning_list = reasoning_list[:min_length]
    
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
        summary=None
    )

def ingest_multiple_logs(log_directory, db_path):
    """
    Ingest multiple Alfworld log files from a directory.
    
    Args:
        log_directory (str): Directory containing log files
        db_path (str): Path to the SQLite database
    """
    # Create the database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Initialize database
    db = None #LearningDB(db_path)
    
    # Iterate through log files
    for filename in sorted(os.listdir(log_directory)):
        if filename.endswith('.txt'):
            log_file_path = os.path.join(log_directory, filename)
            try:
                print(f"Ingesting log file: {filename}")
                ingest_alfworld_log(log_file_path, db)
            except Exception as e:
                print(f"Error ingesting {filename}: {e}")

if __name__ == "__main__":
    # Example usage
    log_directory = "/mnt/ssd/agent_algo_bench/logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/trial_1"  # Directory containing log files
    db_path = "./data/alfworld_learning.db"
    
    ingest_multiple_logs(log_directory, db_path)