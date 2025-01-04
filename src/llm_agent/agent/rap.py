from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action

class RAP(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        if not self.plan:
            # Retrieve with task and category as key--we don't really have a notion of category built into this db but we can add it for this I guess...
            # Works in general for workloads that function in the same type of environment (alfworld, browser, etc.) but have different categories of requirements
            plan_data = self.get_in_context_data(key_type=["goal", "category"], key=[self.goal, self.category], value_type=["goal", "plan"], outcome="winning", k=3)
            await self.create_plan(obs, valid_actions, in_context_data=plan_data) 
            print("plan", self.plan)
        if self.in_context_data is None:
            self.in_context_data = self.get_in_context_data(key_type=["goal", "category", "plan", "observation"], key=[self.goal, self.category, self.plan, obs.structured], value_type=["goal", "observation", "reasoning", "action"], outcome="winning", k=3, window=20)
        reasoning = await self.reason(obs, valid_actions, in_context_data=self.in_context_data)
        self.in_context_data = self.get_in_context_data(key_type=["goal", "category", "plan", "reasoning"], key=[self.goal, self.category, self.plan, reasoning], value_type=["goal", "plan", "observation", "reasoning", "action"], outcome="winning", k=4, window=5)
        action = await self.act(obs, valid_actions, reasoning, in_context_data=self.in_context_data) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        