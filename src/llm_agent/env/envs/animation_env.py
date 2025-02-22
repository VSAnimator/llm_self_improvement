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
        self.id = config.get('problem_id', 0)
        self.category = "animation"
        
    def reset(self) -> Observation:
        """Reset environment to initial state
        
        Returns:
            Initial observation containing user instructions and current animation state
        """
        self.steps = 0
        # Get initial user instructions and animation state
        instructions = "kick higher."
        current_animation = "the person is kicking with their right leg."
        self.goal = instructions
        
        self._observation = f"Current animation state: {current_animation}\nUser instructions: {instructions}"
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
        # If ("right_hip", "flex") is not in the action, need to add feedback
        if '("right_hip", "flex")' not in action:
            feedback = "Flexing the hip helps kick higher."
        else:
            feedback = 'y' #input("\nIs this code good? (y/n): ")
        
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
                * from actions import translate_joint, rotate_joint, relative_translate_joint, fix_joint
                * from timing import when_joint, at_global_moment, before_joint, as_joint, at_frame, get_current_frame
                * from speed import change_speed
                * from motion_io import load_motion, save_motion
                You can only choose from the following method parameters:
                relative_moments = ["highest", "lowest", "midrange_height", "furthest_from_body", "closest_to_body"]
                global_moments = ["start_of_motion", "end_of_motion", "middle_of_motion", "entire_motion"]
                translate_directions = ["forward", "backward", "up", "down"]
                rotation_directions = ["abduct", "adduct", "extend", "flex"]
                relative_translate_directions = ["towards", "above", "below", "in_front", "contact", "next_to"]
                # IMMUTABLE: joints that move with rotation
                joints_that_rotate = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow", "right_knee", "left_knee", "right_hip", "left_hip"]
                # IMMUTABLE: joints that move with translation
                joints_that_translate = ["right_foot", "left_foot", "right_hand", "left_hand", "waist"]
                other_locations = ["ground"]
                speeds = ["fast", "slow", "pause"]
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
