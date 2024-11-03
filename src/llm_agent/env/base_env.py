from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Observation:
    """Represents an environment observation with both structured and text descriptions"""
    structured: Dict[str, Any]  # Structured representation of observation

@dataclass 
class Action:
    """Represents an action with text descriptions"""
    text: str  # Natural language description of action

class BaseEnv(ABC):
    """Abstract base class for environments that can interact with LLM agents"""
    
    def __init__(self):
        """Initialize environment"""
        pass
    
    @abstractmethod
    def reset(self) -> Observation:
        """Reset environment to initial state
        
        Returns:
            Initial observation
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_action_space(self) -> Dict:
        """Get JSON schema describing valid action format
        
        Returns:
            JSON schema for actions
        """
        pass
    
    def get_available_actions(self) -> List[Action]:
        """Get list of available actions
            
        Returns:
            List of available actions
        """
        return []  # Default implementation returns empty list
        
    def render(self) -> str:
        """Render environment observation as text
        
        Returns:
            Text description of current observation
        """
        return ""  # Default implementation returns empty string

    @property
    def current_observation(self) -> Observation:
        """Get current environment observation"""
        return self._observation
