from llm_agent.agent.base_agent_v2 import BaseAgent
from ...env.base_env import Observation, Action
from ...in_context.alfworld_fewshots import get_fewshots_for_goal

class AutoGuide(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions):
        """Choose an action from available actions given the current observation"""
        # Use the previos reflections as in-context data
        example_data = self.get_trajectory_data(key_types=["category"], keys=[self.category], value_types=["goal", "observation", "reasoning", "action"], outcome="winning", k=2) # TODO: revisit keys for this call...
        rule_data = self.get_rule_data(trajectory_key_types=["category", "goal"], trajectory_keys=[self.category, self.goal], state_key_types=["observation"], state_keys=[obs.structured], value_types=["rules"], outcome=None, k=3, window=20) # Rules come from successes and failures # Should bring back the actual example data for positive rules, not sure what to do when fetching both positive and negative rules
        print("Example data: ", example_data)
        print("Rule data: ", rule_data)
        input()
        in_context_data = [example_data, rule_data]
        if not self.plan:
            await self.create_plan(obs, valid_actions, in_context_data)
        reasoning = await self.reason(obs, valid_actions, in_context_data)
        action = await self.act(obs, valid_actions, reasoning, in_context_data) 
        return action
    
    async def batch_analyze_episodes(self):
        """For AutoGuide, this is where we generate context-aware guidelines"""
        await self.generate_rules(mode="pair", environment_id="all")
        await self.consolidate_rules() # Basically identifies shared contexts/names and consolidates the lists. Store under the same context/name if similar enough, and can modify existing names/contexts to match if needed. 
        pass