from intercode.envs import BashEnv

from ..base_env import BaseEnv, Observation, Action

class InterCodeEnv(BaseEnv):
    def __init__(self, config):
        super().__init__(config)
        self.max_steps = config.get('max_steps', 100)
        self.problem_id = config.get('problem_id', 0)
        if config.get('split') == 'train':
            # Train splits 
            split_counts = {1: 60, 2: 53, 3: 60, 4: 27}
            total_count = sum(split_counts.values())
            # Generate a random permutation of the problem ids
            import random
            random.seed(0)
            problem_ids = list(range(total_count))
            random.shuffle(problem_ids)
            # Select the problem id
            if self.problem_id >= total_count:
                raise ValueError(f"Problem id {self.problem_id} is out of range for train split")
            problem_index = problem_ids[self.problem_id]
            # Get split and index within split
            split = 1
            while problem_index >= split_counts[split]:
                problem_index -= split_counts[split]
                split += 1
            # Need to set image name and test_data
            bash_image_name = f"intercode-nl2bash-{split}"
            self.problem_id = problem_index
            bash_test_data = f"/mnt/ssd/intercode/intercode_github/data/nl2bash/nl2bash_fs_{split}.json"
        else:
            from intercode.assets import bash_image_name, bash_test_data
        self.env = BashEnv(bash_image_name, 
                          data_path=bash_test_data,
                          traj_dir=config.get('traj_dir', 'logs/'),
                          verbose=config.get('verbose', True))
        self.category = "bash"
        self.id = self.problem_id

    def reset(self):
        self.env.reset(self.problem_id)
        obs = self.env.observation
        self.goal = obs
        print(f"Goal: {self.goal}")
        info = {}
        return obs, info

    def step(self, action):
        action = action.strip()
        # Remove "execute[ " and "]" from the action
        if action.startswith("execute["):
            action = action[len("execute["):-1]
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def get_action_space(self):
        return {
            "type": "string",
            "description": """
                Submit bash commands to solve the programming task.
                Use 'submit' as the final action when you believe your solution is complete.
                You will be evaluated on the Latest Standard Output + File System State.
                You have up to 10 steps to submit your solution.
                The environment will evaluate your solution and provide a score between 0.0 and 1.0.
                Your action must be a valid bash command or 'submit'. The action MUST be a single line. Do not use newlines, do not wrap your text with ```bash''', and by no means should you use the 'submit' command in the same line as any other command.
                There is no post-processing of your output. The command you provide will be used as-is.
            """.strip()
        }
    
    def get_action_space_new(self):
        return {
            "type": "string",
            "description": """
                Submit bash commands to solve the programming task.
                Use 'submit' as the final action when you believe your solution is complete.
                You will be evaluated on the Latest Standard Output + File System State.
                If you believe the latest observation is the final answer, you can complete the task by running 'submit' by itself.
                You have 10 iterations to solve the task.
                Follow the syntax and logical flow from the provided examples exactly.
            """.strip()
        }

    def get_available_actions(self, info):
        return ['Any valid bash command', 'submit']

    def close(self):
        self.env.close()
