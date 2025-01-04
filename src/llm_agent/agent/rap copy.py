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
        reasoning_data = self.get_in_context_data(key_type=["goal", "category", "plan", "observation"], key=[self.goal, self.category, self.plan, obs.structured], value_type=["goal", "observation", "reasoning", "action"], k=3)
        reasoning = await self.reason(obs, valid_actions, in_context_data=reasoning_data)
        # Theoretically this should have in-context data but we already provide valid actions so i don't think this is needed...the problem of mapping reasoning to action text is made easier when the valid actions are provided
        action_data = self.get_in_context_data(key_type=["goal", "category", "plan", "observation", "reasoning"], key=[self.goal, self.category, self.plan, obs.structured, reasoning], value_type=["goal", "plan", "observation", "reasoning", "action"], k=3)
        print("action_data", action_data)
        print("keys", [self.goal, self.category, self.plan, obs.structured, reasoning])
        input()
        action = await self.act(obs, valid_actions, reasoning, in_context_data=action_data) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        