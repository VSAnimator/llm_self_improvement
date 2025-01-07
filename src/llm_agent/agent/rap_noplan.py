from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action

class RAPNoPlan(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        if self.in_context_data is None:
            self.in_context_data = self.get_in_context_data(key_type=["goal", "observation"], key=[self.goal, obs.structured], value_type=["goal", "observation", "reasoning", "action"], outcome="winning", k=4, window=5)
        reasoning = await self.reason(obs, valid_actions, in_context_data=self.in_context_data)
        self.in_context_data = self.get_in_context_data(key_type=["goal", "reasoning"], key=[self.goal, reasoning], value_type=["goal", "observation", "reasoning", "action"], outcome="winning", k=4, window=5)
        action = await self.act(obs, valid_actions, reasoning, in_context_data=self.in_context_data) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        