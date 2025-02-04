from llm_agent.agent.base_agent_v2 import BaseAgent
from ...env.base_env import Observation, Action
from ...in_context.alfworld_fewshots import get_fewshots_for_goal

class ReAct(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions):
        """Choose an action from available actions given the current observation"""
        data = self.get_trajectory_data(key_types=["goal"], keys=[self.goal], value_types=["goal", "plan", "observation", "reasoning", "action"], outcome="winning", k=2)
        if not self.plan:
            await self.create_plan(obs, valid_actions, in_context_data=data)
        reasoning = await self.reason(obs, valid_actions, in_context_data=data)
        action = await self.act(obs, valid_actions, reasoning, in_context_data=data) 
        return action