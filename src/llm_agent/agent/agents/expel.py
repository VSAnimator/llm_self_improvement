from llm_agent.agent.base_agent_v2 import BaseAgent
from ...env.base_env import Observation, Action
from ...in_context.alfworld_fewshots import get_fewshots_for_goal

class Expel(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions):
        """Choose an action from available actions given the current observation"""
        # Use the previos reflections as in-context data
        example_data = self.get_trajectory_data(key_types=["category"], keys=[self.category], value_types=["goal", "observation", "reasoning", "action"], outcome="winning", k=2) # TODO: revisit keys for this call...
        rule_data = self.get_rule_data(key_types=None, keys=None, value_types=["rule_content"], k=3)
        in_context_data = [example_data, rule_data]
        if not self.plan:
            await self.create_plan(obs, valid_actions, in_context_data)
        reasoning = await self.reason(obs, valid_actions, in_context_data)
        action = await self.act(obs, valid_actions, reasoning, in_context_data) 
        return action
    
    async def batch_analyze_episodes(self):
        """For Expel, leverage both contrastive pairs and similar sets"""
        # We loop through contrastive pairs of episodes, and generate rules for each pair
        '''
        contrastive_pairs = self.get_contrastive_pairs()
        similar_sets = self.get_similar_sets(n = len(contrastive_pairs), k = 3)
        for pair in contrastive_pairs:
            self.generate_rule(pair, update_system="vote") # Add/update/delete rule
        for set in similar_sets:
            self.generate_rule(set, update_system="vote") # Add/update/delete rule
        '''
        #await self.generate_rules(mode="pair", environment_id="all")
        await self.generate_rules(mode="similar", environment_id="all")
        await self.consolidate_rules()
    pass