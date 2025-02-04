from typing import Dict, List, Tuple, Optional
from .base_env import BaseEnv, Observation, Action

class CodeEnv(BaseEnv):
    """Environment wrapper that executes code containing sub-environment actions"""
    
    def __init__(self, config: Dict):
        """Initialize Code environment
        
        Args:
            config: Configuration dictionary containing:
                - env: Sub-environment instance to wrap
        """
        super().__init__(config)
        self.env = config['env']
        self._observation = None
        self.steps = 0

    def reset(self) -> Observation:
        """Reset environment to initial state
        
        Returns:
            Initial observation from sub-environment
        """
        obs, info = self.env.reset()
        self._observation = obs
        self.steps = 0
        return obs, info
        
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """Execute code containing sub-environment actions
        
        Args:
            action: String containing Python code with sub-environment action calls
            
        Returns:
            Final observation, reward, done flag and info from last sub-environment step
        """
        self.steps += 1
        
        # Create local namespace with access to sub-env
        local_ns = {'env': self.env}
        
        # Execute the code
        try:
            # Execute code and get returned values
            result = exec(action.text, {}, local_ns)
            if 'obs' not in local_ns or 'reward' not in local_ns or 'done' not in local_ns or 'info' not in local_ns:
                raise Exception("Code must assign obs, reward, done, info variables")
            return local_ns['obs'], local_ns['reward'], local_ns['done'], local_ns['info']
        except Exception as e:
            # Return failure if code execution fails
            return self._observation, -1, True, {'error': str(e)}
            
    def get_action_space(self) -> Dict:
        """Get JSON schema describing valid action format
        
        Returns:
            JSON schema for Python code actions
        """
        # Get action space from wrapped env
        wrapped_action_space = self.env.get_action_space()
        
        return {
            "type": "string",
            "description": f"""
                Valid Python code that can contain one or more calls to the sub-environment.
                The sub-environment is available as 'env' in the code namespace.
                The code MUST assign the following variables:
                - obs: The final observation  
                - reward: The final reward
                - done: Whether the episode is done
                - info: Additional info dictionary
                
                The wrapped environment accepts actions with this schema:
                {wrapped_action_space}
                
                Example:
                    obs, reward, done, info = env.step(some_action) # where action matches the above schema
            """.strip()
        }
