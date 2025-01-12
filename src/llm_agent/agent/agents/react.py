from llm_agent.agent.base_agent_v2 import BaseAgent
from ...env.base_env import Observation, Action

class React(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        if not self.plan:
            await self.create_plan(obs, valid_actions) # Re-planning based off reflection can go in here
        reasoning = await self.reason(obs, valid_actions)
        action = await self.act(obs, valid_actions, reasoning) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        return