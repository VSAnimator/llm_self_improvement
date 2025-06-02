from typing import List, Tuple, Dict, Optional, Any
from ..base_agent_v2 import BaseAgent
from ...env.base_env import Observation, Action

class FinetuneAgent(BaseAgent):
    """Agent that uses a fine-tuned LLM to select actions"""
    
    def __init__(self, llm, db, env, config):
        super().__init__(llm, db, env, config)
    
    async def choose_action(self, obs: Observation, valid_actions) -> Tuple[Action, List[Dict]]:
        self.observation_history.append(obs)
        action = await self.act_finetune(obs)
        return action
    
    async def analyze_episode(self):
        self.clean_history()
