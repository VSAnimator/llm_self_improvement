from llm_agent.agent.base_agent_v2 import BaseAgent
from ...env.base_env import Observation, Action

class RAPFlex(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions):
        if not self.plan:
            plan_data = self.get_trajectory_data(key_types=["goal", "category"], keys=[self.goal, self.category], value_types=["goal", "plan"], outcome="winning", k=self.num_ic)
            await self.create_plan(obs, valid_actions, in_context_data=plan_data) 
        if self.in_context_data is None:
            self.in_context_data = self.get_state_data(trajectory_key_types=["goal", "category", "plan"], trajectory_keys=[self.goal, self.category, self.plan], state_key_types=["observation"], state_keys=[obs.structured], value_types=["goal", "observation", "reasoning", "action"], outcome="winning", k=self.num_ic, window=5)
        reasoning = await self.reason(obs, valid_actions, in_context_data=self.in_context_data)
        self.in_context_data = self.get_state_data(trajectory_key_types=["goal", "category", "plan"], trajectory_keys=[self.goal, self.category, self.plan], state_key_types=["reasoning"], state_keys=[reasoning], value_types=["goal", "plan", "observation", "reasoning", "action"], outcome="winning", k=self.num_ic, window=5)
        action = await self.act(obs, valid_actions, reasoning, in_context_data=self.in_context_data) 
        return action
    
    async def analyze_episode(self):
        """Process feedback from the environment"""
        nothing_indices = [i for i in range(len(self.observation_history)) if "Nothing happens" in self.observation_history[i].structured]
        
        for idx in sorted(nothing_indices, reverse=True):
            del self.observation_history[idx]
            if idx > 0:
                if idx-1 < len(self.reasoning_history):
                    del self.reasoning_history[idx-1]
                if idx-1 < len(self.action_history):
                    del self.action_history[idx-1]
                if idx-1 < len(self.reward_history):
                    del self.reward_history[idx-1]
