from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action
from ..in_context.alfworld_fewshots import get_fewshots_for_goal

class VanillaTrain(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        data = self.get_in_context_data(key_type=["goal"], key=[self.goal], value_type=["goal", "observation", "reasoning", "action"], k=2, window=20) # Window should have no effect here but just in case get the full trajectory
        if not self.plan:
            await self.create_plan(obs, valid_actions, in_context_data=data)
        reasoning = await self.reason(obs, valid_actions, in_context_data=data)
        action = await self.act(obs, valid_actions, reasoning, in_context_data=data) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        return