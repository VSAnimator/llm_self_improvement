import json
import os
import sqlite3
import subprocess
import sys

from llm_agent.database.learning_db_postgresql_new import (
    LearningDB,
)  # Replace with actual import path for your LearningDB class


# Define classes to match the structure expected by store_episode
class Observation:
    def __init__(self, structured):
        self.structured = structured


class Action:
    def __init__(self, text):
        self.text = text


# Check and get command-line arguments
if len(sys.argv) != 4:
    print(
        "Usage: python transform_and_dump.py <sqlite_db_path> <dump_file_path> <socket_file>"
    )
    sys.exit(1)

sqlite_db_path = sys.argv[1]
dump_file_path = sys.argv[2]
socket_file = sys.argv[3]

# PostgreSQL connection parameters (update these to match your setup)
db_host = socket_file
db_name = "learning_db"
db_user = "agent"
db_password = "password"

# Step 1: Connect to SQLite and load data
sqlite_conn = sqlite3.connect(sqlite_db_path)
sqlite_conn.row_factory = sqlite3.Row  # Allows accessing columns by name
sqlite_cur = sqlite_conn.cursor()

# Step 2: Connect to PostgreSQL
learning_db = LearningDB(
    db_path=db_host, db_name=db_name  # , user=db_user, password=db_password
)

# Retrieve all rows from the SQLite trajectories table
sqlite_cur.execute("SELECT * FROM trajectories")
rows = sqlite_cur.fetchall()

# Step 3: Transform and insert data into PostgreSQL
for row in rows:
    # Extract scalar fields
    environment_id = row["environment_id"]
    goal = row["goal"]
    category = row["category"]
    plan = row["plan"]
    reflection = row["reflection"]
    summary = row["summary"]

    # Parse JSON-encoded fields
    observations_json = row["observations"]
    observations = [
        Observation(structured) for structured in json.loads(observations_json)
    ]

    actions_json = row["actions"]
    actions = [Action(text) for text in json.loads(actions_json)]

    reasoning_json = row["reasoning"]
    reasoning = json.loads(reasoning_json) if reasoning_json else None

    rewards_json = row["rewards"]
    rewards = json.loads(rewards_json)

    # Insert into PostgreSQL using store_episode
    learning_db.store_episode(
        environment_id=environment_id,
        goal=goal,
        category=category,
        observations=observations,
        reasoning=reasoning,
        actions=actions,
        rewards=rewards,
        plan=plan,
        reflection=reflection,
        summary=summary,
    )

# Close the SQLite connection
sqlite_conn.close()

# Step 4: Dump the PostgreSQL database to the specified file path
os.environ["PGPASSWORD"] = db_password  # Set password for pg_dump
subprocess.run(
    [
        "pg_dump",
        "-h",
        db_host,
        "-U",
        db_user,
        "-d",
        db_name,
        "-f",
        dump_file_path,
    ]
)
# Unset PGPASSWORD for security
del os.environ["PGPASSWORD"]
