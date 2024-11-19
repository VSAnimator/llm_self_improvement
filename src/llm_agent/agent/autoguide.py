from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action

class AutoGuide(BaseAgent):
    def __init__(self, config, llm):
        super().__init__(config, llm)

    async def choose_action(self, goal, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        in_context_low = await self.get_in_context_data(key_type="goal", key=goal, value_type="trajectory")
        in_context_high = await self.get_in_context_data(key_type="goal", key=goal, value_type="reflexions")
        # Combine the two in-context data dictionaries
        in_context_data = {**in_context_low, **in_context_high}
        # Create observation from obs string
        obs = Observation(structured=obs)
        # Create action objects from valid action strings
        valid_actions = [Action(text=action) for action in valid_actions]
        if self.config.get('use_summarization', False):
            obs = await self.summarize(goal, obs) # Create_conversation can pull in the trajectory
        # Agent config should control the behavior here, reflect all algorithms we want to encompass
        await self.create_plan(goal, obs, valid_actions, in_context_data) # Re-planning based off reflexion can go in here
        reasoning = await self.reason(goal, obs, valid_actions, in_context_data)
        action = await self.act(goal, obs, valid_actions, reasoning, in_context_data) 
        return action
    
    async def process_feedback(self, goal, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        new_obs = Observation(structured=new_obs)
        reflexion = None
        summary = None
        if done:
            # We need to add to the database here
            reflexion = await self.reflect(goal, new_obs)
            await self.store_episode(goal, self.observation_history, self.reasoning_history, self.action_history, self.reward_history, self.plan, self.in_context_data, reflexion, summary)
        