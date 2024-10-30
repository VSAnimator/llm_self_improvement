import os
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from logging import getLogger

from ..env.base_env import State, Action
from ..llm.lite_llm import LiteLLMWrapper

logger = getLogger(__name__)

class BaseAgent:
    """Base agent class that uses an LLM to select actions"""
    
    def __init__(self, llm: LiteLLMWrapper, config: Dict):
        """Initialize the agent
        
        Args:
            llm: LLM instance to use for decision making
            config: Configuration dictionary containing agent parameters
        """
        self.llm = llm
        self.config = config
        self.max_retries = config.get('max_retries', 3)
        self.memory_size = config.get('memory_size', 10)
        self.temperature = config.get('temperature', 0.7)
        
        # Keep track of recent states and actions
        self.state_history: List[State] = []
        self.action_history: List[Action] = []
        
    async def select_action(self, state: State, available_actions: List[Action]) -> Action:
        """Select an action from available actions given the current state"""
        # Add state to history
        self.state_history.append(state)
        self.state_history = self.state_history[-self.memory_size:]
            
        # Format prompt with state and action info
        prompt = self._format_prompt(state, available_actions)
        
        # Get LLM response
        for _ in range(self.max_retries):
            try:
                # Await the response
                response = await self.llm.generate_chat([{"role": "user", "content": prompt}])
                action = self._parse_action(response, available_actions)
                if action is not None:
                    self.action_history.append(action)
                    self.action_history = self.action_history[-self.memory_size:]
                    return action
            except Exception as e:
                logger.error(f"Error selecting action: {str(e)}")
                
        # If all retries failed, return first available action
        logger.warning("Failed to select action, defaulting to first available")
        action = available_actions[0]
        self.action_history.append(action)
        self.action_history = self.action_history[-self.memory_size:]
        return action
        
    def _format_prompt(self, state: State, available_actions: List[Action]) -> str:
        """Format prompt for LLM with state and action information
        
        Args:
            state: Current environment state
            available_actions: List of available actions
            
        Returns:
            Formatted prompt string
        """
        # Basic prompt template - override in subclasses for custom prompting
        prompt = f"Current state: {repr(state)}\n\n"
        prompt += "Available actions:\n"
        for i, action in enumerate(available_actions):
            prompt += f"{i+1}. {action.text}\n"
        prompt += "\nSelect the most appropriate action number:"
        return prompt
        
    def _parse_action(self, response: str, available_actions: List[Action]) -> Optional[Action]:
        """Parse LLM response to get selected action
        
        Args:
            response: Raw response from LLM
            available_actions: List of available actions
            
        Returns:
            Selected action or None if parsing failed
        """
        try:
            # Try to parse action number from response
            print("Response: ", response)
            action_num = int(response.strip())
            if 1 <= action_num <= len(available_actions):
                return available_actions[action_num - 1]
        except:
            pass
        return None
        
    def reset(self):
        """Reset agent state between episodes"""
        self.state_history = []
        self.action_history = []
