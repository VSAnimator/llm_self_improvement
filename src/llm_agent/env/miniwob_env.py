import time
import gymnasium
import miniwob
from miniwob.action import ActionTypes
from typing import Dict, List, Tuple
from selenium.webdriver.common.keys import Keys

from ..synapse.envs.miniwob.action import (
    MiniWoBType,
    MiniWoBElementClickXpath,
    MiniWoBElementClickOption,
    MiniWoBMoveXpath,
)
from .base_env import Action, BaseEnv

class MiniWoBEnv(BaseEnv):
    """Environment for MiniWoB++ tasks"""

    def __init__(self, task_name: str):
        """Initialize environment
        
        Args:
            task_name: Name of MiniWoB++ task to load
        """
        super().__init__()
        self.task_name = task_name
        self.env = None
        self.steps = 0

    def _initialize_goal(self):
        # Combined utterance and fields
        return 
        
    def reset(self):
        """Reset environment to initial state
        
        Returns:
            Initial observation and info dict
        """
        if self.env is None:
            gymnasium.register_envs(miniwob)
            self.env = gymnasium.make(f'miniwob/{self.task_name}')
            
        obs, info = self.env.reset()
        self.steps = 0

        # Set utterance and fields
        utterance = obs["utterance"]
        fields = obs["fields"]

        # Also set goal
        self.goal = f"Goal utterance: {utterance}. Goal fields: {fields}."

        return obs, info
        
    def step(self, action):
        """Take step in environment
        
        Args:
            action: Action to take
            
        Returns:
            observation: Next observation
            reward: Reward received 
            done: Whether episode is complete
            info: Additional information
        """
        '''
        # Parse action into MiniWoB format
        element = None
        for elem in self.env.unwrapped.obs["dom_elements"]:
            if str(elem["ref"]) == action.action:
                element = elem
                break
                
        if element is None:
            return Observation(""), 0.0, True, {"error": "Invalid element reference"}
            
        miniwob_action = self.env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=element["ref"])
        '''
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        
        return obs, reward, terminated or truncated, info
        
    def render(self):
        """Render current state"""
        pass
        
    def close(self):
        """Clean up environment"""
        if self.env is not None:
            self.env.close()
            
    def get_action_space(self, obs) -> List[str]:
        """Get list of possible actions in current state
        
        Returns:
            List of valid action strings
        """
        # Return list of element refs that can be clicked
        if self.env is None or obs is None:
            return []
            
        # Get all clickable elements from DOM
        action_space = []
        for element in obs["dom_elements"]:
            # Add element ref as a valid action
            action_space.append(str(element["ref"]))
            
        return action_space
        
    '''
    def get_observation_space(self) -> List[str]:
        """Get list of possible observations
        
        Returns:
            List of valid observation strings
        """
        # MiniWoB has continuous observation space consisting of:
        # - Task utterance (text)
        # - Fields (key-value pairs)
        # - Screenshot (RGB array) 
        # - DOM elements (list of dicts)
        return []
    '''

    # Action: type a string via the keyboard
    def type(self, characters: str) -> None:
        action = MiniWoBType(characters)
        self.step(action)

    def click_xpath(self, xpath: str):
        action = MiniWoBElementClickXpath(xpath)
        self.step(action)

    def press(self, key_type: str) -> None:
        if key_type == "enter":
            action = MiniWoBType("\n")
        elif key_type == "space":
            action = MiniWoBType(" ")
        elif key_type == "arrowleft":
            action = MiniWoBType(Keys.LEFT)
        elif key_type == "arrowright":
            action = MiniWoBType(Keys.RIGHT)
        elif key_type == "backspace":
            action = MiniWoBType(Keys.BACKSPACE)
        elif key_type == "arrowup":
            action = MiniWoBType(Keys.UP)
        elif key_type == "arrowdown":
            action = MiniWoBType(Keys.DOWN)
        elif key_type in ["command+a", "command+c", "command+v"]:
            action = MiniWoBType(key_type)
        else:
            raise ValueError("Invalid instruction")
        self.step(action)

    def click_option(self, xpath: str):
        action = MiniWoBElementClickOption(xpath)
        self.step(action)

    def movemouse(self, xpath: str):
        action = MiniWoBMoveXpath(xpath)
        self.step(action)
