import os
import json
from typing import Dict, List, Optional, Tuple
import textworld
import textworld.gym
from alfworld.agents.utils.misc import add_task_to_grammar
from alfworld.agents.environment.alfred_tw_env import AlfredExpert, AlfredDemangler, AlfredExpertType
from alfworld.info import ALFWORLD_DATA
import alfworld
import random
import glob
import random
from os.path import join as pjoin

from ..base_env import BaseEnv, Observation, Action
from ...in_context.alfworld_fewshots import get_task_type

class AlfWorldTrainEnv(BaseEnv):
    """Environment wrapper for ALFWorld text-based environments"""
    
    def __init__(self, config: Dict):
        """Initialize ALFWorld environment

        Args:
            config: Configuration dictionary containing:
                - env: Environment configuration
                - max_steps: Maximum steps per episode (default: 50)
        """
        super().__init__(config)
        
        self.max_steps = config.get('max_steps', 50)
        
        # Initialize alfworld environment
        split = config['split']
        self.env = getattr(alfworld.agents.environment, config["type"])(config, train_eval=split)
        self.env = self.env.init_env(batch_size=1)
        
        # Track current state
        self._observation = None
        self.steps = 0
        self.id = None

        # Seed the environment
        self.env.seed(config.get('problem_id'))
        
    def reset(self) -> Observation:
        """Reset environment to initial state
        
        Returns:
            Initial observation
        """
        obs, info = self.env.reset()
        obs = obs[0]
        # Get goal out of obs
        self.goal = "Your task is to: " + obs.split('Your task is to: ')[1].split('___')[0]
        self.category = get_task_type(self.goal)
        obs = obs.split("-= Welcome to TextWorld, ALFRED! =-")[1].split("Your task is to: ")[0].strip()
        self._observation = obs
        self.steps = 0
        self.id = info['extra.gamefile'][0] # Set ID from info
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
        
        obs, reward, done, info = self.env.step([action])
        obs = obs[0]
        reward = reward[0]
        done = done[0]
        print(reward, done)
        # Clean the obs following prior work
        if obs.startswith('You arrive at '):
            obs = obs[obs.find('. ')+2:]
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
            """.strip("\n")
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