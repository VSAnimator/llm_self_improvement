from llm_agent.agent.base_agent_v2 import BaseAgent
from ...env.base_env import Observation, Action

class Synapse(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions):
        """Choose an action from available actions given the current observation"""
        in_context_data = self.get_trajectory_data(key_types=["goal"], keys=[self.goal], value_types=["goal","observation", "reasoning", "action"], outcome="winning", k=2)
        reasoning = await self.reason(obs, valid_actions, in_context_data)
        action = await self.act(obs, valid_actions, reasoning) 
        return action
    
        