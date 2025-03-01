from llm_agent.agent.base_agent_v2 import BaseAgent
from ...env.base_env import Observation, Action

class ReflexionRefine(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions):
        """Choose an action from available actions given the current observation"""
        # Use the previos reflections as in-context data
        reflection_data = self.get_trajectory_data(key_types=["environment_id"], keys=[self.environment_id], value_types=["reflection"], outcome=None, k=10) # Up to 10 reflections
        exemplar_data = self.get_trajectory_data(key_types=["goal"], keys=[self.goal], value_types=["goal", "plan", "observation", "reasoning", "action"], outcome="winning", k=2)
        if len(reflection_data[0]) == 0 and len(reflection_data[1]) == 0:
            in_context_data = exemplar_data
        else:
            in_context_data = {"reflection": reflection_data, "exemplar": exemplar_data}
        print(f"Reflection data: {reflection_data}")
        print(f"Exemplar data: {exemplar_data}")
        # Agent config should control the behavior here, reflect all algorithms we want to encompass
        if not self.plan:
            await self.create_plan(obs, valid_actions, in_context_data) # Re-planning based off reflection can go in here
        reasoning = await self.reason(obs, valid_actions, in_context_data)
        action = await self.act(obs, valid_actions, reasoning, in_context_data) 
        return action
    
    async def analyze_episode(self):
        """Process feedback from the environment"""
        # Reflect regardless of success or failure
        in_context_data = self.get_trajectory_data(key_types=["goal"], keys=[self.goal], value_types=["goal", "plan", "observation", "reasoning", "action"], outcome="winning", k=2)
        await self.reflect(in_context_data)
        