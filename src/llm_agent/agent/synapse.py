from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action

class Synapse(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        # Agent config should control the behavior here, reflect all algorithms we want to encompass
        in_context_data = self.get_in_context_data(key_type="goal", key=self.goal, value_type=["observation", "reasoning", "action"], outcome="winning", k=2) # key=repr(obs.structured)
        if not self.plan:
            await self.create_plan(obs, valid_actions, in_context_data) # Re-planning based off reflection can go in here
        reasoning = await self.reason(obs, valid_actions, in_context_data)
        print("Reasoning", reasoning)
        input()
        action = await self.act(obs, valid_actions, reasoning) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        