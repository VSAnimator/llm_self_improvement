from llm_agent.agent.base_agent_v2 import BaseAgent

class Reflexion(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions):
        """Choose an action from available actions given the current observation"""
        # Use the previos reflections as in-context data
        in_context_data = self.get_trajectory_data(key_types=["environment_id"], keys=[self.environment_id], value_types=["reflection"], outcome="losing", k=5)
        # Agent config should control the behavior here, reflect all algorithms we want to encompass
        if not self.plan:
            await self.create_plan(obs, valid_actions, in_context_data) # Re-planning based off reflection can go in here
        reasoning = await self.reason(obs, valid_actions, in_context_data)
        action = await self.act(obs, valid_actions, reasoning, in_context_data) 
        return action
    
    async def analyze_episode(self):
        """Process feedback from the environment"""
        if self.reward_history[-1] < 1:
            await self.reflect()
        self.clean_history()
        