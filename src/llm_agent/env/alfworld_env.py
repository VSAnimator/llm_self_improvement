import json
import os
from typing import Dict, List, Optional, Tuple

import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import (
    AlfredDemangler,
    AlfredExpert,
    AlfredExpertType,
)
from alfworld.agents.utils.misc import add_task_to_grammar

from llm_agent.env.alfworld_examples import get_task_type
from llm_agent.env.base_env import Action, BaseEnv, Observation


class AlfWorldEnv(BaseEnv):
    """Environment wrapper for ALFWorld text-based environments"""

    def __init__(self, config: Dict):
        """Initialize ALFWorld environment

        Args:
            config: Configuration dictionary containing:
                - problem: Path to problem folder containing initial_state.pddl and traj_data.json
                - domain: Path to PDDL domain file
                - grammar: Path to grammar file
                - max_steps: Maximum steps per episode (default: 50)
        """
        super().__init__(config)

        self.config = config["logic"]
        self.max_steps = config.get("max_steps", 50)

        # Load game logic files
        self.game_logic = {
            "pddl_domain": open(self.config["domain"]).read(),
            "grammar": open(self.config["grammar"]).read(),
        }

        print("config", config)

        if "problem" not in config:
            """
            problems = glob.glob(pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True)

            # Remove problem which contains movable receptacles.
            problems = [p for p in problems if "movable_recep" not in p]

            if len(problems) == 0:
                raise ValueError(f"Can't find problem files in {ALFWORLD_DATA}. Did you run alfworld-data?")

            print("Using problem: ", problems[1])
            config['problem'] = os.path.dirname(problems[1]) #os.path.dirname(random.choice(problems))
            """

            # Read alfworld tasks suffix
            with open("data/alfworld/alfworld_tasks_suffix.json", "r") as f:
                self.tasks = json.load(f)

            print("Problem ID", config.get("problem_id"))

            # Select random task
            if config.get("problem_id") is not None:
                self.task = self.tasks[config["problem_id"]]
            else:
                # Throw error
                raise ValueError("Problem ID not specified")
                # self.task = self.tasks[9] #random.choice(self.tasks)

            print("self.task", self.task)

            config["problem"] = os.path.dirname(self.task["gamefile"])
            # Also cut off everything before "Your task is to: "
            self.goal = (
                "Your task is to: "
                + self.task["goal"].split("Your task is to: ")[1].split("___")[0]
            )
            self.category = get_task_type(self.goal)

        # Load state and trajectory files
        pddl_file = os.path.join(config["problem"], "initial_state.pddl")
        json_file = os.path.join(config["problem"], "traj_data.json")
        with open(json_file, "r") as f:
            traj_data = json.load(f)

        # Add task to grammar
        self.game_logic["grammar"] = add_task_to_grammar(
            self.game_logic["grammar"], traj_data
        )

        # Create game file
        gamedata = dict(**self.game_logic, pddl_problem=open(pddl_file).read())
        self.gamefile = os.path.join(os.path.dirname(pddl_file), "game.tw-pddl")
        json.dump(gamedata, open(self.gamefile, "w"))

        # Initialize expert
        self.expert = AlfredExpert(expert_type=AlfredExpertType.PLANNER)

        # Register game environment
        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            score=True,
            max_score=True,
            intermediate_reward=True,
            extras=["expert_plan"],
        )

        # Create environment ID and initialize
        self.env_id = textworld.gym.register_game(
            self.gamefile,
            request_infos,
            max_episode_steps=self.max_steps,
            wrappers=[AlfredDemangler(), self.expert],
        )

        # Create the environment
        self.env = textworld.gym.make(self.env_id)

        self.id = self.gamefile

        # Track current state
        self._observation = None
        self.steps = 0

    def reset(self) -> Observation:
        """Reset environment to initial state

        Returns:
            Initial observation
        """
        obs, info = self.env.reset()
        # Remove the "Your task is to: " text and everything after
        obs = (
            obs.split("-= Welcome to TextWorld, ALFRED! =-")[1]
            .split("Your task is to: ")[0]
            .strip()
        )
        self._observation = obs
        self.steps = 0
        return obs, info

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """Take action in environment

        Args:
            action: Text action to take

        Returns:
            Tuple containing:
            - Next observation (observation text)
            - Reward (1 for success, 0 otherwise)
            - Done flag
            - Info dict
        """
        self.steps += 1

        action = action.replace(" in ", " in/on ").replace(" on ", " in/on ")

        obs, reward, done, info = self.env.step(action)
        # Clean the obs following prior work
        if obs.startswith("You arrive at "):
            obs = obs[obs.find(". ") + 2 :]
        self._observation = obs

        # End episode if max steps reached
        if self.steps >= self.max_steps:
            done = True

        return obs, reward, done, info

    def get_action_space(self) -> Dict:
        """Get JSON schema describing valid action format

        Returns:
            JSON schema for text actions
        """
        # This action space is from the TRAD paper
        return {
            "type": "string",
            "description": """
                * go to target: Move to the target, and you will observe what is in/on the target or know it is closed or opened.
                * open target: Open the target when it is closed, and you will observe what is in/on the target. Only cabinets, drawers, fridges, safes, and microwaves can be opened.
                * take object from target: Take the object from the target when the object is in/on the target. You can only take one object at the same time.
                * put object in/on target: Put an object you have taken/picked up in/on the target. You should go to the target in your last action. You can put no matter there are other objects in/on the target or not.
                * clean object with target: Clean an object you have taken/picked up with the target. The target should be a sinkbasin. You should go to the target in your last action. You can clean no matter there are other objects in/on the target or not.
                * heat object with target: Heat an object you have taken/picked up with the target. The target should be a microwave. You should go to the target in your last action. You can heat no matter there are other objects in/on the target or not.
                * cool object with target: Cool an object you have taken/picked up with the target. The target should be a fridge. You should go to the target in your last action. You can cool no matter there are other objects in/on the target or not.
                * use target: Use the object. The object should be a desklamp. You should be in/on a place where the object appears.
                * look: Look around and see what you are facing. Only look when nothing happens.
                Replace "target" with the desired location/object.
                Replace "object" with the desired object.
                Neither the word "target" nor "object" should be in the action command.
            """.strip(
                "\n"
            ),
        }

    def get_available_actions(self, info: Optional[Dict] = None) -> List[Action]:
        """Get list of available actions in current/given state

        Args:
            state: State to get actions for (uses current state if None)

        Returns:
            List of valid actions in current state
        """
        if info is None:
            return []

        # Get admissible commands from env info
        return info["admissible_commands"]

    def render(self) -> str:
        """Render environment state as text

        Returns:
            Text description of current state
        """
        return self._observation
