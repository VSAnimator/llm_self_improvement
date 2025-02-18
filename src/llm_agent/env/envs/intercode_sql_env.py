from intercode.envs import (
    BashEnv, SqlEnv, CTFEnv
)
from typing import Dict, List
from intercode.assets import sql_build_docker, sql_image_name, sql_test_data


def preprocess_ctf(record: Dict) -> List:
    cmds = [f"cd /ctf/{record['task_id']}"]
    if "setup" in record:
        cmds.append(record["setup"])
    return cmds

def preprocess_sql(record: Dict) -> List:
    db = record["db"]
    return [f"use {db}"]

base_path = "/mnt/ssd/intercode/intercode_github/data/"
DEMO_MAP = {
    "bash": {"env": BashEnv, "image_name": "intercode-nl2bash", "data_path": base_path + "nl2bash/nl2bash_fs_1.json"},
    "sql": {"env": SqlEnv, "image_name": "docker-env-sql_ic_ctr", "data_path": base_path + "sql/bird/ic_bird.json", "preprocess": preprocess_sql},
    "ctf": {"env": CTFEnv, "image_name": "intercode-ctf", "data_path": base_path + "ctf/ic_ctf.json", "preprocess": preprocess_ctf},
}

from ..base_env import BaseEnv, Observation, Action

class InterCodeSqlEnv(BaseEnv):
    def __init__(self, config):
        super().__init__(config)
        self.max_steps = config.get('max_steps', 100)
        self.problem_id = config.get('problem_id', 0)
        demo = "sql"
        '''
        image_name = DEMO_MAP[demo]["image_name"]
        data_path = DEMO_MAP[demo]["data_path"] if "data_path" in DEMO_MAP[demo] else None
        preprocess = None #DEMO_MAP[demo]["preprocess"] if "preprocess" in DEMO_MAP[demo] else None
        '''

        #self.env = DEMO_MAP[demo]["env"](image_name, data_path=data_path, verbose=True)
        sql_build_docker()
        self.env = SqlEnv(sql_image_name, data_path=sql_test_data, verbose=True)
        self.category = "sql"
        self.id = self.problem_id

    def reset(self):
        self.env.reset(self.problem_id)
        obs = self.env.observation
        self.goal = obs
        info = {}
        return obs, info

    def step(self, action):
        action = action.strip()
        obs, reward, done, info = self.env.step(action)
        if obs is None:
            obs = "No output"
        return obs, reward, done, info

    def get_action_space(self):
        return {
            "type": "string",
            "description": """
                Your action space is outputting valid mysql commands to solve the sql task.
                You will be evaluated on the Latest Standard Output.
                If you believe the latest observation is the final answer, you can complete the task by running 'submit' BY ITSELF WITH NO OTHER TEXT.
                Otherwise, you can run any valid mysql command. Do not wrap your text with ```sql'''. You can run multiple commands in one action, separated by semicolons.
                You have up to 10 iterations of interaction to obtain a solution.
                The environment will evaluate your solution and provide a score between 0.0 and 1.0.
                There is no post-processing of your output. The command you provide will be used as-is.
            """.strip()
        }

    def get_available_actions(self, info):
        return ['Any valid bash command', 'submit']

    def close(self):
        self.env.close()
