from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action

class RAP(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        test = self.get_in_context_data(key_type=["category"], key=[self.category], value_type=["category"], outcome="winning", k=3)
        print(test)
        print("keys", [self.category])
        input()
        # Do a second test--search just by goal, return just goal
        test2 = self.get_in_context_data(key_type=["goal"], key=[self.goal], value_type=["goal"], outcome="winning", k=3)
        print(test2)
        print("keys", [self.goal])
        input()
        # Do a third test--search by goal and observation, return goal, observation, and action
        '''
        test3 = self.get_in_context_data(key_type=["goal", "observation"], key=[self.goal, obs.structured], value_type=["goal", "observation", "action"], outcome="winning", k=3)
        print(test3)
        print("keys", [self.goal, obs.structured])
        input()
        '''
        # A fourth test--search by goal and category, return goal, category, and plan
        test4 = self.get_in_context_data(key_type=["goal", "category"], key=[self.goal, self.category], value_type=["goal", "category", "plan"], outcome="winning", k=3)
        print(test4)
        print("keys", [self.goal, self.category])
        input()
        if not self.plan:
            # Retrieve with task and category as key--we don't really have a notion of category built into this db but we can add it for this I guess...
            # Works in general for workloads that function in the same type of environment (alfworld, browser, etc.) but have different categories of requirements
            plan_data = self.get_in_context_data(key_type=["goal", "category"], key=[self.goal, self.category], value_type=["goal", "plan"], outcome="winning", k=3)
            print(plan_data)
            input()
            await self.create_plan(obs, valid_actions, in_context_data=plan_data) 
        reasoning_data = self.get_in_context_data(key_type=["goal", "category", "plan", "observation"], key=[self.goal, self.category, self.plan, obs], value_type=["observation", "action"])
        reasoning = await self.reason(obs, valid_actions, in_context_data=reasoning_data)
        # Theoretically this should have in-context data but we already provide valid actions so i don't think this is needed...the problem of mapping reasoning to action text is made easier when the valid actions are provided
        action = await self.act(obs, valid_actions, reasoning) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        