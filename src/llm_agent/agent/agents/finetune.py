import asyncio
from typing import List, Tuple, Dict, Optional, Any
from ..base_agent_v2 import BaseAgent
from ...env.base_env import Observation, Action

class FinetuneAgent(BaseAgent):
    """Agent that uses a fine-tuned LLM to select actions"""
    
    def __init__(self, llm, db, env, config):
        """Initialize the agent
        
        Args:
            llm: LLM instance to use for decision making
            db: Database instance for storing and retrieving episodes
            env: Environment instance
            config: Configuration dictionary containing agent parameters
        """
        super().__init__(llm, db, env, config)
    
    async def choose_action(self, obs: Observation, valid_actions: List[Action]) -> Tuple[Action, List[Dict]]:
        """Choose an action from available actions given the current observation
        
        Args:
            obs: Current observation
            valid_actions: List of valid actions
            
        Returns:
            Tuple of selected action and conversation history
        """
        # Add observation to history
        self.observation_history.append(obs)
        
        # Use the fine-tuned model to select an action
        action = await self.act_finetune(obs)
        
        return action
    
    async def process_feedback(self, reward: float, done: bool) -> None:
        self.clean_history()
