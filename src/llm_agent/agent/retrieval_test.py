from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action

class RetrievalTest(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        if not self.plan:   
            in_context_data = self.get_in_context_data(key_type="environment_id", key=self.environment_id, value_type=["plan"], outcome="winning", k=5) # key=repr(obs.structured) ["observation", "action"]
            self.plan = await self.create_plan(obs, valid_actions, in_context_data)
        in_context_data = self.get_in_context_data(key_type="environment_id", key=self.environment_id, value_type=["observation", "action"], outcome="winning", k=1) # key=repr(obs.structured)
        reasoning = await self.reason(obs, valid_actions, in_context_data)
        print("Reasoning", reasoning)
        action = await self.act(obs, valid_actions, reasoning) 
        print("Action", action)
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        '''
        new_obs = Observation(structured=new_obs)
        reflection = None
        summary = None
        if done and reward == 1:
            # We need to add to the database here
            await self.store_episode(reflection, summary)
        '''
        