from intercode.envs import (
    BashEnv, SqlEnv, CTFEnv
)
from typing import Dict, List
from intercode.assets import sql_build_docker, sql_image_name, sql_test_data
import time

def preprocess_ctf(record: Dict) -> List:
    cmds = [f"cd /ctf/{record['task_id']}"]
    if "setup" in record:
        cmds.append(record["setup"])
    return cmds

def preprocess_sql(record: Dict) -> str:
    print("Record", record)
    db = record['extra']["db"]
    return f"use {db}; "
    #return "show databases;"

base_path = "/mnt/ssd/intercode/intercode_github/data/"
DEMO_MAP = {
    "bash": {"env": BashEnv, "image_name": "intercode-nl2bash", "data_path": base_path + "nl2bash/nl2bash_fs_1.json"},
    #"sql": {"env": SqlEnv, "image_name": "docker-env-sql-ic-bird", "data_path": base_path + "sql/bird/ic_bird.json", "preprocess": preprocess_sql},
    "sql": {"env": SqlEnv, "image_name": "docker-env-sql-spider", "data_path": base_path + "sql/spider/ic_spider_dev.json", "preprocess": preprocess_sql},
    "ctf": {"env": CTFEnv, "image_name": "intercode-ctf", "data_path": base_path + "ctf/ic_ctf.json", "preprocess": preprocess_ctf},
}

from ..base_env import BaseEnv, Observation, Action

class InterCodeSqlEnv(BaseEnv):
    def __init__(self, config):
        super().__init__(config)
        self.max_steps = config.get('max_steps', 100)
        self.problem_id = config.get('problem_id', 0)
        demo = "sql"

        image_name = DEMO_MAP[demo]["image_name"]
        data_path = DEMO_MAP[demo]["data_path"] if "data_path" in DEMO_MAP[demo] else None
        self.env = DEMO_MAP[demo]["env"](image_name, data_path=data_path, verbose=True, preprocess=preprocess_sql)
        self.data_path = data_path

        #sql_build_docker()
        #self.env = SqlEnv(sql_image_name, data_path=sql_test_data, verbose=True, preprocess=preprocess_sql)
        #self.data_path = sql_test_data
        self.category = "sql"
        # Okay in mapping the problem id to the actual problem id we need a random permutation of the problem ids
        num_entries = None
        if self.data_path.endswith(".json"):
            import json
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                num_entries = len(data)
        else:
            import csv
            with open(self.data_path, 'r') as f:
                reader = csv.reader(f)
                num_entries = len(list(reader)) - 1
        # Define the permutation
        import random
        random.seed(0)
        problem_ids = list(range(num_entries))
        random.shuffle(problem_ids)
        # Select the problem id
        if self.problem_id >= num_entries:
            raise ValueError(f"Problem id {self.problem_id} is out of range for train split")
        self.problem_id = problem_ids[self.problem_id]
        self.id = self.problem_id

    def reset(self):
        x = self.env.reset(self.problem_id)
        print("Reset", x)
        #time.sleep(30)
        obs = self.env.observation
        gold = None
        # If obs is none, get it from the data path
        if obs is None:
            if self.data_path.endswith(".json"):
                import json
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    obs = data[self.problem_id]['query']
                    gold = data[self.problem_id]['gold']
            else:
                import csv
                with open(self.data_path, 'r') as f:
                    reader = csv.reader(f)
                    obs = list(reader)[self.problem_id + 1][0]
                    gold = list(reader)[self.problem_id + 1][1]
        self.goal = obs
        if False and gold is not None:
            self.goal += f"\nHere is the gold answer: {gold}. Don't directly use this command, your goal is to look as though you are solving the problem yourself."
        info = {}
        # Wait for the environment to be ready
        #time.sleep(30)
        obs = repr(obs)
        return obs, info

    def step(self, action):
        action = action.strip()
        # Remove "execute[ " and "]" from the action
        if action.startswith("execute["):
            action = action[len("execute["):-1]
        '''
        if action.startswith("action: ") or action.startswith("Action: "):
            action = action[len("action: "):]
        # Also strip out ```sql and ``` from the action
        if action.startswith("```sql"):
            action = action[len("```sql"):]
        if "```" in action:
            action = action.split("```")[0]
        '''
        obs, reward, done, info = self.env.step(action)
        if obs is None:
            obs = "No output"
        obs = repr(obs)
        return obs, reward, done, info

    def get_action_space(self):
        return {
            "type": "string",
            "description": """
                Your action space is outputting valid mysql commands to solve the sql task.
                You will be evaluated on the Latest Standard Output.
                If you believe the latest observation is the final answer, you can complete the task by running 'submit' by itself.
                You have 10 iterations to solve the task.
                Follow the syntax and logical flow from the provided examples exactly.
            """.strip()

        }

    def get_available_actions(self, info):
        return ['Any valid bash command', 'submit']

    def close(self):
        self.env.close()
