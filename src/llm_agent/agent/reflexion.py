from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action

class Reflexion(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        # Use the previos reflections as in-context data
        in_context_data = self.get_in_context_data(key_type="environment_id", key=self.environment_id, value_type="reflection")
        # Agent config should control the behavior here, reflect all algorithms we want to encompass
        if not self.plan:
            await self.create_plan(obs, valid_actions, in_context_data) # Re-planning based off reflection can go in here
        reasoning = await self.reason(obs, valid_actions, in_context_data)
        action = await self.act(obs, valid_actions, reasoning, in_context_data) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        if done:
            reflection = None if reward == 1 else await self.reflect(new_obs)
            self.store_episode(reflection, None)
        