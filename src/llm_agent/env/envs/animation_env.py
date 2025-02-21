from typing import Dict, List, Optional, Tuple
from llm_agent.env.base_env import BaseEnv, Observation, Action

class AnimationEnv(BaseEnv):
    """Environment for generating animation code based on user instructions"""
    
    def __init__(self, config: Dict):
        """Initialize animation environment
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.max_steps = config.get('max_steps', 10)
        self._observation = None
        self.steps = 0
        
    def reset(self) -> Observation:
        """Reset environment to initial state
        
        Returns:
            Initial observation containing user instructions and current animation state
        """
        self.steps = 0
        # Get initial user instructions and animation state
        instructions = input("Enter animation instructions: ")
        current_animation = input("Enter current animation state: ")
        
        self._observation = f"Instructions: {instructions}\nCurrent Animation: {current_animation}"
        return self._observation, {}
        
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """Take action in environment by executing generated code
        
        Args:
            action: Python code to execute
            
        Returns:
            Tuple containing:
            - Next observation (user feedback)
            - Reward (1 for success, 0 otherwise)
            - Done flag 
            - Info dict
        """
        self.steps += 1
        
        # Get user feedback on generated code
        print("\nGenerated code:")
        print(action)
        feedback = input("\nIs this code good? (y/n): ")
        
        # Calculate reward based on feedback
        reward = 1.0 if feedback.lower() == 'y' else 0.0
        done = feedback.lower() == 'y' or self.steps >= self.max_steps
        
        self._observation = f"Previous feedback: {feedback}"
        
        return self._observation, reward, done, {}
        
    def get_action_space(self) -> Dict:
        """Get JSON schema describing valid action format
        
        Returns:
            JSON schema for Python animation code
        """
        return {
            "type": "string",
            "description": """
                Generate Python code using the following animation functions:
                * create_character(name: str) -> Character
                * set_pose(character: Character, pose: str)
                * add_keyframe(character: Character, time: float)
                * play_animation(character: Character)
                * save_animation(character: Character, filename: str)
            """.strip()
        }
        
    def get_available_actions(self, info: Optional[Dict] = None) -> List[Action]:
        """Get list of available actions in current state
        
        Args:
            info: Additional state information (unused)
            
        Returns:
            Empty list since actions are free-form Python code
        """
        return []
        
    def render(self) -> str:
        """Render environment state as text
        
        Returns:
            Current observation text
        """
        return self._observation
