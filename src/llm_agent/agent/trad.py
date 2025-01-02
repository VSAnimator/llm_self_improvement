from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action

class TRAD(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        reasoning_data = self.get_in_context_data(key_type=["goal", "category"], key=[self.goal, self.category], value_type=["state", "reasoning", "action"])
        reasoning = await self.reason(obs, valid_actions, in_context_data=reasoning_data)
        action_data = self.get_in_context_data(key_type=["reasoning"], key=[reasoning], value_type=["state", "reasoning", "action"], length=3)
        action = await self.act(obs, valid_actions, reasoning, in_context_data=action_data) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        