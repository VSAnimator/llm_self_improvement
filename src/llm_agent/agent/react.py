from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action

class React(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        # Use the previos reflections as in-context data
        # Create observation from obs string
        obs = Observation(structured=obs)
        # Create action objects from valid action strings
        valid_actions = [Action(text=action) for action in valid_actions]
        #if self.config.get('use_summarization', False):
        #    obs = await self.summarize(goal, obs) # Create_conversation can pull in the trajectory
        # Agent config should control the behavior here, reflect all algorithms we want to encompass
        await self.create_plan(obs, valid_actions) # Re-planning based off reflexion can go in here
        reasoning = await self.reason(obs, valid_actions)
        action = await self.act(obs, valid_actions, reasoning) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        return