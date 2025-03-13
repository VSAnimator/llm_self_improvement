import os
from llm_agent.database.learning_db import LearningDB
from llm_agent.env.base_env import Observation, Action

def parse_wordcraft_log(log_content):
    """
    Parse the Wordcraft log file and extract trajectory information.
    
    Args:
        log_content (str): Content of the log file
    
    Returns:
        tuple: Extracted trajectory components (goal, plan, obs_list, reasoning_list, act_list, rewards)
    """

    # Go to the last instance of "Initial observation: "
    initial_observation_index = log_content.rfind("Initial observation: ")
    log_content = log_content[initial_observation_index:]

    # Extract goal
    goal_match = log_content.split("Goal: ")[1].split("\n")[0]
    category = "wordcraft"
    
    # Extract plan
    plan_match = log_content.split("Plan: ")[1].split("\n")[0] if "Plan: " in log_content else None
    
    # Extract observations, actions, reasoning, and rewards
    obs_list = []
    act_list = []
    reasoning_list = []
    rewards = []
    
    lines = log_content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Initial observation: "):
            obs_list.append(line[22:].strip())
        elif line.startswith("Obs: "):
            obs_reward = line[5:].strip()
            obs_list.append(obs_reward.split("Reward: ")[0].strip()[:-1])
            rewards.append(float(line.split("Reward: ")[1].strip()))
        elif line.startswith("Selected action: "):
            wrapped_action = line.split("Selected action: ")[1].strip()
            # Now remove "Action(text=" and ")"
            act_text = wrapped_action.split("Action(text=")[1][1:][:-2]
            act_list.append(act_text)
        elif line.startswith("Reasoning: "):
            # Handle multiline reasoning
            reasoning_text = line[11:].strip()
            i += 1
            while i < len(lines) and not (lines[i].startswith("Output:") or 
                                         lines[i].startswith("Selected action:") or 
                                         lines[i].startswith("Obs:") or 
                                         lines[i].startswith("Initial observation:") or
                                         lines[i].startswith("Reward:")):
                reasoning_text += " " + lines[i].strip()
                i += 1
            reasoning_list.append(reasoning_text)
            continue  # Skip the increment at the end of the loop
        elif line.startswith("Reward: "):
            rewards.append(float(line.split("Reward: ")[1].strip()))
        i += 1
    
    return goal_match, category, plan_match, obs_list, reasoning_list, act_list, rewards

def ingest_wordcraft_log(log_file_path, db):
    """
    Ingest Wordcraft trajectory log into the learning database.
    
    Args:
        log_file_path (str): Path to the log file
        db_path (str): Path to the SQLite database
    """
    # Read log file
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    # Parse log content
    goal, category, plan, obs_list, reasoning_list, act_list, rewards = parse_wordcraft_log(log_content)

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
        nosave=False
    )

def ingest_multiple_logs(log_files, db_path):
    """
    Ingest multiple Wordcraft log files from a directory.
    
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
            ingest_wordcraft_log(log_file_path, db)
        except Exception as e:
            print(f"Error ingesting {log_file_path}: {e}")

if __name__ == "__main__":
    # Example usage
    import glob
    
    # Read the wordcraft log files and ingest them into the database
    log_files = glob.glob("./data/wordcraft/depth2_humanic/*.txt")
    ingest_multiple_logs(log_files, "./data/wordcraft/depth2_humanic/learning.db")
