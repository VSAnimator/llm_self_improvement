
from textcraft import TextCraft
from typing import Tuple, Dict, Optional, List
from ..base_env import BaseEnv, Observation, Action

class TextCraftEnv(BaseEnv):
    """Environment for text-based crafting game"""
    
    def __init__(self, config: Dict):
        """Initialize TextCraft environment
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.max_steps = config.get('max_steps', 100)
        self._observation = None
        self.steps = 0
        self.id = config.get('problem_id', 0)
        self.category = "textcraft"
        self.env = TextCraft()
        print("TextCraftEnv initialized")
        
    def reset(self) -> Tuple[Observation, Dict]:
        """Reset environment to initial state
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation and info dict
        """
        self.steps = 0
        obs, info = self.env.reset(self.id)
        self._observation = obs
        self.goal = obs.split("Goal:")[1].strip()

        return obs, info
        
    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, Dict]:
        """Take action in environment
        
        Args:
            action: Text command from player
            
        Returns:
            Tuple containing:
            - Next observation (game response)
            - Reward
            - Done flag
            - Info dict
        """
        self.steps += 1
        
        # Process action and update game state
        #combined_observation = []
        #actions = action.split(";")
        #print("Actions: ", actions)
        #for action in actions:
        action = action.lower().strip()
        self._observation, reward, terminated, truncated, info = self.env.step(action)
        
        done = terminated or truncated
        return self._observation, reward, done, info
        
    def get_action_space(self) -> Dict:
        """Get JSON schema describing valid action format
        
        Returns:
            JSON schema for text commands
        """
        return {
            "type": "string",
            "description": 'You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format "craft [target object] using [input ingredients]". You can either "get" an object (ingredients) from the inventory or the environment or "craft" (target) using any of the crafting commands. The only viable commands are to "get [count] [x]" where x is an ingredient that cannot be crafted with the provided recipes, or "craft [x] using [y]" where the command is a provided crafting command. You can use ONLY these crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use special types of the same ingredient e.g. "dark oak planks" in the command instead. Each command must be a single line only without any other text.'
        }
        # (note that you are free to multiply the count of the ingredients if needed)
        #If you have forgotten what is in your inventory, you can use the "inventory" command to see what is in your inventory. You start with an empty inventory.
        
    def get_available_actions(self, info: Optional[Dict] = None) -> List[Action]:
        """Get list of available actions in current state
        
        Args:
            info: Additional state information (unused)
            
        Returns:
            Empty list since actions are free-form text commands
        """
        return []
        
    def render(self) -> str:
        """Render environment state as text
        
        Returns:
            Current observation text
        """
        return self._observation

