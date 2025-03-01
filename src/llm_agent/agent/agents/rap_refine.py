from llm_agent.agent.base_agent_v2 import BaseAgent
from ...env.base_env import Observation, Action

class RAPRefine(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions):
        reflection_data = self.get_trajectory_data(key_types=["environment_id"], keys=[self.environment_id], value_types=["reflection"], outcome=None, k=10) # Up to 10 reflections
        if len(reflection_data[0]) > 0:
            # We have at least one positive reflection
            reflection_data = (reflection_data[0], [])
        if not self.plan:
            plan_data = self.get_trajectory_data(key_types=["goal", "category"], keys=[self.goal, self.category], value_types=["goal", "plan"], outcome="winning", k=3)
            if len(reflection_data[0]) == 0 and len(reflection_data[1]) == 0:
                in_context_data = plan_data
            else:
                in_context_data = {"reflection": reflection_data, "exemplar": plan_data}
            print(f"Reflection data: {reflection_data}")
            print(f"Exemplar data: {plan_data}")
            await self.create_plan(obs, valid_actions, in_context_data=in_context_data) 
        if self.in_context_data is None:
            self.in_context_data = self.get_state_data(trajectory_key_types=["goal", "category", "plan"], trajectory_keys=[self.goal, self.category, self.plan], state_key_types=["observation"], state_keys=[obs.structured], value_types=["goal", "observation", "reasoning", "action"], outcome="winning", k=4, window=5)
            if not (len(reflection_data[0]) == 0 and len(reflection_data[1]) == 0):
                self.in_context_data = {"reflection": reflection_data, "exemplar": self.in_context_data}
        reasoning = await self.reason(obs, valid_actions, in_context_data=self.in_context_data)
        self.in_context_data = self.get_state_data(trajectory_key_types=["goal", "category", "plan"], trajectory_keys=[self.goal, self.category, self.plan], state_key_types=["reasoning"], state_keys=[reasoning], value_types=["goal", "plan", "observation", "reasoning", "action"], outcome="winning", k=4, window=5)
        if not (len(reflection_data[0]) == 0 and len(reflection_data[1]) == 0):
            self.in_context_data = {"reflection": reflection_data, "exemplar": self.in_context_data}
        action = await self.act(obs, valid_actions, reasoning, in_context_data=self.in_context_data) 
        return action
    
    async def analyze_episode(self):
        """Process feedback from the environment"""
        in_context_data = self.get_trajectory_data(key_types=["goal"], keys=[self.goal], value_types=["goal", "plan", "observation", "reasoning", "action"], outcome="winning", k=2)
        await self.reflect(in_context_data)

        # Remove invalid actions
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
