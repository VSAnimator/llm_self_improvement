from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action
from ..in_context.alfworld_fewshots import get_fewshots_for_goal

class ExpelTrain(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        # Use the previos reflections as in-context data
        # Create observation from obs string
        obs = Observation(structured=obs)
        # Create action objects from valid action strings
        valid_actions = [Action(text=action) for action in valid_actions]
        if self.config.get('use_summarization', False):
            obs = await self.summarize(obs) # Create_conversation can pull in the trajectory
        expel_in_context = get_fewshots_for_goal(self.goal)
        expel_in_context = [repr(entry) for entry in expel_in_context]
        # Add to reflexion in_context_data
        in_context_data = self.get_in_context_data(key_type="environment_id", key=self.environment_id, value_type="reflexion", outcome="losing")
        in_context_data['low_level'] = (True, expel_in_context)
        # Also get in_context_data from the previous episodes
        print("In context:", in_context_data)
        # Agent config should control the behavior here, reflect all algorithms we want to encompass
        if not self.plan:
            await self.create_plan(obs, valid_actions, in_context_data) # Re-planning based off reflexion can go in here
        reasoning = await self.reason(obs, valid_actions, in_context_data)
        action = await self.act(obs, valid_actions, reasoning, in_context_data) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        new_obs = Observation(structured=new_obs)
        summary = None
        if done:
            reflexion = None
            if reward < 1:
                reflexion = await self.reflect(new_obs)
            self.store_episode(reflexion, summary)
        return