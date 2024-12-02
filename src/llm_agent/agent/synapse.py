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
        await self.create_plan(obs, valid_actions) # Re-planning based off reflexion can go in here
        reasoning = await self.reason(obs, valid_actions)
        in_context_data = self.get_in_context_data(key_type="state", key=repr(obs.structured), value_type=["state", "action"])
        action = await self.act(obs, valid_actions, reasoning, in_context_data) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        new_obs = Observation(structured=new_obs)
        reflexion = None
        summary = None
        if done and reward == 1:
            # We need to add to the database here
            await self.store_episode(reflexion, summary)
        