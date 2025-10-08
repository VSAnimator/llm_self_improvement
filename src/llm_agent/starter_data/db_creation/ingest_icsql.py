import os

from llm_agent.database.learning_db import LearningDB
from llm_agent.env.base_env import Action, Observation


def parse_icsql_log(log_content):
    """
    Parse the Alfworld log file and extract trajectory information.

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
    category = "intercode_sql"

    # Extract plan
    plan_match = (
        log_content.split("Plan: ")[1].split("\n")[0]
        if "Plan: " in log_content
        else None
    )

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
            rewards.append(float(line.split("Reward: ")[1].strip()))
        elif line.startswith("Selected action: "):
            wrapped_action = line.split("Selected action: ")[1].strip()
            # Now remove "Action(text=" and ")"
            print("Wrapped action: ", wrapped_action)
            act_text = wrapped_action.split("Action(text=")[1][1:][:-2]
            print("Act text: ", act_text)
            act_list.append(act_text)
        elif line.startswith("Reasoning: "):
            reasoning_list.append(line[11:].strip())
        elif line.startswith("Reward: "):
            rewards.append(float(line.split("Reward: ")[1].strip()))

    return goal_match, category, plan_match, obs_list, reasoning_list, act_list, rewards


def ingest_icsql_log(log_file_path, db):
    """
    Ingest Intercode SQL trajectory log into the learning database.

    Args:
        log_file_path (str): Path to the log file
        db_path (str): Path to the SQLite database
    """
    # Read log file
    with open(log_file_path, "r") as f:
        log_content = f.read()

    # Parse log content
    goal, category, plan, obs_list, reasoning_list, act_list, rewards = parse_icsql_log(
        log_content
    )

    # Validate lists match in length
    # Validate that observations list is one longer than actions list
    if len(obs_list) != len(act_list) + 1:
        raise ValueError(
            f"Observation list should be one longer than action list. Found {len(obs_list)} observations and {len(act_list)} actions."
        )

    # Validate that reasoning list matches action list in length
    if len(reasoning_list) != len(act_list):
        raise ValueError(
            f"Reasoning list length ({len(reasoning_list)}) doesn't match action list length ({len(act_list)}). This may cause issues."
        )

    # Ensure rewards matches actions length
    if not rewards:
        rewards = [0] * (len(act_list) - 1) + [1]

    # Convert obs_list and act_list to Observation and Action objects
    obs_list = [Observation(o) for o in obs_list]
    act_list = [Action(a) for a in act_list]

    # Store trajectory
    environment_id = os.path.basename(log_file_path).replace(".txt", "")

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
        nosave=True,
    )


def ingest_multiple_logs(log_files, db_path):
    """
    Ingest multiple Intercode SQL log files from a directory.

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
            ingest_icsql_log(log_file_path, db)
        except Exception as e:
            print(f"Error ingesting {log_file_path}: {e}")


if __name__ == "__main__":
    # Example usage
    import csv
    import json

    log_files = []
    for task_id in range(10):
        log_file = f"./logs/episodes/intercode_sql/test/rap_noplan/openai/gpt-4o-mini/create_gold_spider/{task_id}.txt"
        log_files.append(log_file)
    ingest_multiple_logs(
        log_files,
        f"./data/intercode_sql_filtered/intercode_sql_gold_examples_spider/learning.db",
    )
