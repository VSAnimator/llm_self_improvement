from llm_agent.agent.base_agent_v2 import BaseAgent

class ZeroShotReact(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions):
        """Choose an action from available actions given the current observation"""
        if not self.plan:
            await self.create_plan(obs, valid_actions) 
        reasoning = await self.reason(obs, valid_actions)
        action = await self.act(obs, valid_actions, reasoning) 
        return action
    
    async def analyze_episode(self):
        self.clean_history()