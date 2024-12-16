from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action

class RAP(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        if not self.plan:
            await self.create_plan(obs, valid_actions) # Re-planning based off reflection can go in here
        reasoning = await self.reason(obs, valid_actions)
        in_context_data = self.get_in_context_data(key_type="reasoning", key=reasoning, value_type=["state", "action"])
        action = await self.act(obs, valid_actions, reasoning, in_context_data) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        new_obs = Observation(structured=new_obs)
        if done and reward == 1:
            self.store_episode(None, None)
        