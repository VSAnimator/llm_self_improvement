import os
from llm_agent.database.learning_db import LearningDB
from llm_agent.env.base_env import Observation, Action
from llm_agent.in_context.alfworld_fewshots import get_task_type

def ingest_multiple_logs(log_files, db_path, available_dbs, lower_threshold, upper_threshold):
    """
    Ingest multiple Alfworld entries from source databases.
    
    Args:
        log_files (list): List of [task_id, db_id] pairs to ingest
        db_path (str): Path to the target SQLite database
        available_dbs (dict): Dictionary of available source databases
    """
    # Create the database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Initialize database
    db = LearningDB(db_path)
    
    # Prepare all task_ids for each source database
    db_to_tasks = {}
    for task_id, db_id in log_files:
        if db_id not in db_to_tasks:
            db_to_tasks[db_id] = []
        db_to_tasks[db_id].append(task_id)
    
    # Fetch all entries at once from each source database
    all_entries = []
    for db_id, task_ids in db_to_tasks.items():
        print(f"Fetching {len(task_ids)} entries from {db_id}")
        placeholders = ','.join(['?'] * len(task_ids))
        source_db = available_dbs[db_id]
        
        # Split task_ids into environment_id and goal parts
        env_ids = []
        goals = []
        for task_id in task_ids:
            # Split at the last underscore
            parts = task_id.rsplit('_', 1)
            if len(parts) == 2:
                env_ids.append(parts[0])
                goals.append(parts[1])
            else:
                env_ids.append(task_id)
                goals.append(None)
        
        # Create pairs of parameters for the query
        query_params = []
        for env_id, goal in zip(env_ids, goals):
            query_params.append((env_id, goal))
        
        # Use placeholders for both environment_id and goal
        placeholders = ','.join(['(?,?)'] * len(env_ids))
        source_db.trajectory_cursor.execute(
            f"SELECT environment_id, goal, category, observations, reasoning, actions, rewards, plan, id FROM trajectories WHERE (environment_id, goal) IN ({placeholders})",
            [param for pair in query_params for param in pair]  # Flatten the list of tuples
        )
        entries = source_db.trajectory_cursor.fetchall()

        # Remove all entries that have a id greater than the threshold
        entries = [entry for entry in entries if int(entry[8]) >= lower_threshold and int(entry[8]) < upper_threshold]
        all_entries.extend(entries)

    # Ingest all entries
    for i, entry in enumerate(all_entries):
        environment_id = entry[0]
        goal = entry[1]
        category = entry[2]
        obs_list = json.loads(entry[3])
        reasoning_list = json.loads(entry[4])
        act_list = json.loads(entry[5])
        rewards = json.loads(entry[6])
        plan = entry[7]
        
        # Convert to proper objects
        obs_list = [Observation(o) for o in obs_list]
        act_list = [Action(a) for a in act_list]
        
        # Only save on the last entry, use nosave=True for all others
        is_last_entry = (i == len(all_entries) - 1)
        
        # Store in target database
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
            nosave=not is_last_entry
        )
        print(f"Ingested entry for task: {environment_id}")

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
    parser.add_argument('--db_path', type=str, default='./data/alfworld_filtered/alfworld_{mode}_examples/learning.db',
                        help='Path to output database (use {mode} for substitution)')
    
    args = parser.parse_args()
    
    # Format paths with the mode
    mode = args.mode
    json_file = args.json_file.format(mode=mode)
    db_path = args.db_path.format(mode=mode)

    # Now get the sets of environment ids that are viable for each threshold
    env_ids = []
    # Get the key_swaps.json file from the same folder as the json_file
    key_swaps_file = os.path.join(os.path.dirname(json_file), "key_swaps.json")
    with open(key_swaps_file, "r") as f:
        key_swaps = json.load(f)

    # For each value, get the lowest key it's associated with. This is a dict of dicts
    env_id_mapping = {}
    for key, value in key_swaps.items():
        for inner_key, inner_value in value.items():
            if inner_value not in env_id_mapping or int(inner_key) < int(env_id_mapping[inner_value]):
                env_id_mapping[inner_value] = inner_key
    #print(env_id_mapping)

    # Load best examples from JSON or CSV
    log_files = []
    unique_dbs = set()
    with open(json_file, "r") as f:
        best_examples = json.load(f)
        for task_id, data in best_examples.items():
            db_id = data["db_id"]
            log_files.append([task_id, db_id])
            unique_dbs.add(db_id)
    # Let's load the dbs
    dbs = {}
    for db_id in unique_dbs:
        print(f"Loading db: {db_id}")
        dbs[db_id] = LearningDB(db_id)

    import shutil
    # DB starts with 18 entries. We'd like to save the DB with inner key up to 40, 100, 200, 400, 1000, 1500, 2000, 2500
    # For each, create a copy of the db path's folder, then ingest only the logs that pass the threshold
    i = 0
    current_db_dir = None
    all_thresholds = [40, 100, 200, 400, 1000, 1500, 2000, 2500]
    for threshold in all_thresholds:
        # Create a copy of the db path's folder
        # Create the new database path with the threshold
        # Get the directory paths
        db_dir = os.path.dirname(db_path)
        new_db_dir = db_dir + "_" + str(threshold)
        new_db_path = new_db_dir + "/learning.db"
        # Skip if the new db path already exists
        if os.path.exists(new_db_path):
            print(f"Skipping {new_db_path} because it already exists")
            current_db_dir = new_db_dir
            continue
        # Create the new directory if it doesn't exist
        os.makedirs(new_db_dir, exist_ok=True)
        # If current db dir is not none, copy the folder
        if current_db_dir is not None:
            shutil.copytree(current_db_dir, new_db_dir, dirs_exist_ok=True)
        current_db_dir = new_db_dir
        # Filter the log files to only include those that pass the threshold
        filtered_log_files = []
        for log_file in log_files:
            # Use the env_id_mapping to get the inner key
            inner_key = env_id_mapping[log_file[0]]
            # Check if inner_key is above the previous threshold (if any) and below or equal to current threshold
            previous_threshold = 0 if i == 0 else all_thresholds[i-1]
            if previous_threshold <= int(inner_key) < threshold:
                filtered_log_files.append(log_file)
        # Ingest only the logs that pass the threshold
        ingest_multiple_logs(filtered_log_files, new_db_path, dbs, previous_threshold, threshold)
        i += 1
