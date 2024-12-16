from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action

class Synapse(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        # Create observation from obs string
        obs = Observation(structured=obs)
        # Create action objects from valid action strings
        valid_actions = [Action(text=action) for action in valid_actions]
        if self.config.get('use_summarization', False):
            obs = await self.summarize(obs) # Create_conversation can pull in the trajectory
        # Agent config should control the behavior here, reflect all algorithms we want to encompass
        in_context_data = self.get_in_context_data(key_type="environment_id", key=self.environment_id, value_type=["observation", "action"], outcome="losing", k=1) # key=repr(obs.structured)
        if not self.plan:
            await self.create_plan(obs, valid_actions, in_context_data) # Re-planning based off reflection can go in here
            print("Plan", self.plan)
            input()
        reasoning = await self.reason(obs, valid_actions, in_context_data)
        print("Reasoning", reasoning)
        input()
        action = await self.act(obs, valid_actions, reasoning) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        if done and reward == 1:
            # We need to add to the database here
            await self.store_episode(None, None)
        