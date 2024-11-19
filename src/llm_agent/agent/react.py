from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action

class React(BaseAgent):
    def __init__(self, config, llm):
        super().__init__(config, llm)

    async def choose_action(self, goal, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        # Use the previos reflections as in-context data
        # Create observation from obs string
        obs = Observation(structured=obs)
        # Create action objects from valid action strings
        valid_actions = [Action(text=action) for action in valid_actions]
        if self.config.get('use_summarization', False):
            obs = await self.summarize(goal, obs) # Create_conversation can pull in the trajectory
        # Agent config should control the behavior here, reflect all algorithms we want to encompass
        await self.create_plan(goal, obs, valid_actions) # Re-planning based off reflexion can go in here
        reasoning = await self.reason(goal, obs, valid_actions)
        action = await self.act(goal, obs, valid_actions, reasoning) 
        return action
    
    async def process_feedback(self, goal, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        return