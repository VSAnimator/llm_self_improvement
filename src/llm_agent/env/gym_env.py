import gymnasium as gym
from typing import Dict, List, Tuple, Optional
from .base_env import BaseEnv, Observation, Action

class GymEnv(BaseEnv):
    """Environment wrapper for OpenAI Gym/Gymnasium environments"""
    
    def __init__(self, config: Dict):
        """Initialize Gym environment
        
        Args:
            config: Configuration dictionary containing:
                - env_name: Name of Gym environment to create
        """
        super().__init__()
        
        self.env_name = config['env_name']
        self.env = gym.make(self.env_name)

        self.goal = self._initialize_goal() # Need a string describing the goal of the gym task. May also need information on the semantics of the action and observation spaces

        self._observation = None

    def _initialize_goal(self):
        # Need to read a file corresponding to the gym environment name
        # Will have to interpret the file to get the goal + action and observation space semantics string
        return "Keep the Cartpole upright for as long as possible. The state space for the task is [cart_pos, cart_vel, pole_pos, pole_vel]. The action space for the task is [0, 1], where 0 is accelerating the cart left and 1 is accelerating the cart right. The cart position limits are (-2.4, 2.4) and the pole position limits are (-.2095, .2095)."
        
    def reset(self) -> Observation:
        """Reset environment to initial state
        
        Returns:
            Initial observation
        """
        self._observation, info = self.env.reset()
        return self._observation, info
        
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """Take action in environment
        
        Args:
            action: Action to take
            
        Returns:
            Tuple containing:
            - Next observation
            - Reward
            - Done flag 
            - Info dict
        """
        self._observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self._observation, reward, done, info
        
    def get_action_space(self) -> Dict:
        """Get JSON schema describing valid action format
        
        Returns:
            JSON schema for actions
        """
        space = self.env.action_space
        if isinstance(space, gym.spaces.Discrete):
            return {
                "type": "integer",
                "minimum": 0,
                "maximum": space.n - 1
            }
        elif isinstance(space, gym.spaces.Box):
            return {
                "type": "array",
                "items": {
                    "type": "number",
                    "minimum": float(space.low[0]),
                    "maximum": float(space.high[0])
                },
                "minItems": space.shape[0],
                "maxItems": space.shape[0]
            }
        else:
            return {"type": "any"}
            
    def get_available_actions(self, info: Optional[Dict] = None) -> List[Action]:
        """Get list of available actions in current state
        
        Args:
            info: Optional info dict from environment (unused)
            
        Returns:
            List of valid actions in current state
        """
        space = self.env.action_space
        if isinstance(space, gym.spaces.Discrete):
            return list(range(space.n))
        return []
