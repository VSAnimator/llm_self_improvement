from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action
from ..in_context.alfworld_fewshots import get_fewshots_for_goal

class ExpelSummaryTrain(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        # Use the previos reflections as in-context data
        expel_in_context = get_fewshots_for_goal(self.goal)
        expel_in_context = [repr(entry) for entry in expel_in_context]
        in_context_data = self.get_in_context_data(key_type="environment_id", key=self.environment_id, value_type="reflection", outcome="losing")
        in_context_data['low_level'] = (True, expel_in_context)
        # Let's use a summarized version of obs
        print(f"Original obs: {obs}")
        obs = await self.summarize(obs)
        print(f"Summarized obs: {obs}")
        if not self.plan:
            await self.create_plan(obs, valid_actions, in_context_data) # Re-planning based off reflection can go in here
        reasoning = await self.reason(obs, valid_actions, in_context_data)
        action = await self.act(obs, valid_actions, reasoning, in_context_data) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        if done:
            reflection = await self.reflect(new_obs, reward)
            self.store_episode(reflection, None)
        return