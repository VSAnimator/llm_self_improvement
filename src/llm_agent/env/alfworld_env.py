import os
import json
from typing import Dict, List, Optional, Tuple
import textworld
import textworld.gym
from alfworld.agents.utils.misc import add_task_to_grammar
from alfworld.agents.environment.alfred_tw_env import AlfredExpert, AlfredDemangler, AlfredExpertType
from alfworld.info import ALFWORLD_DATA
import random
import glob
import random
from os.path import join as pjoin

from .base_env import BaseEnv, Observation, Action

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
        
        self.config = config['logic']
        self.max_steps = config.get('max_steps', 50)
        
        # Load game logic files
        self.game_logic = {
            "pddl_domain": open(self.config['domain']).read(),
            "grammar": open(self.config['grammar']).read(),
        }

        print("config", config)

        if 'problem' not in config:
            '''
            problems = glob.glob(pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True)

            # Remove problem which contains movable receptacles.
            problems = [p for p in problems if "movable_recep" not in p]

            if len(problems) == 0:
                raise ValueError(f"Can't find problem files in {ALFWORLD_DATA}. Did you run alfworld-data?")
            
            print("Using problem: ", problems[1])
            config['problem'] = os.path.dirname(problems[1]) #os.path.dirname(random.choice(problems))
            '''

            # Read alfworld tasks suffix
            with open('data/alfworld/alfworld_tasks_suffix.json', 'r') as f:
                self.tasks = json.load(f)

            print("Problem ID", config.get('problem_id'))

            # Select random task
            if config.get('problem_id') is not None:
                self.task = self.tasks[config['problem_id']]
            else:
                # Throw error
                raise ValueError("Problem ID not specified")
                #self.task = self.tasks[9] #random.choice(self.tasks)

            print("self.task", self.task)
                
            config['problem'] = os.path.dirname(self.task['gamefile'])
            self.goal = self.task['goal'].split('___')[0]

        # Load state and trajectory files
        pddl_file = os.path.join(config['problem'], 'initial_state.pddl')
        json_file = os.path.join(config['problem'], 'traj_data.json')
        with open(json_file, 'r') as f:
            traj_data = json.load(f)
            
        # Add task to grammar
        self.game_logic['grammar'] = add_task_to_grammar(self.game_logic['grammar'], traj_data)

        # Create game file
        gamedata = dict(**self.game_logic, pddl_problem=open(pddl_file).read())
        self.gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
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
            extras=["expert_plan"]
        )
        
        # Create environment ID and initialize
        self.env_id = textworld.gym.register_game(
            self.gamefile,
            request_infos,
            max_episode_steps=self.max_steps,
            wrappers=[AlfredDemangler(), self.expert]
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
        
        obs, reward, done, info = self.env.step(action)
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
        return {
            "type": "string",
            "description": "Text command to execute in the environment"
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
