from llm_agent.agent.base_agent_v2 import BaseAgent
from ...env.base_env import Observation, Action
from ...in_context.alfworld_fewshots import get_fewshots_for_goal

class AutoManual(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        # Use the previos reflections as in-context data
        '''
        example_data = self.get_trajectory_data(key_types=["category"], keys=[self.category], value_types=["goal", "observation", "reasoning", "action"], outcome="winning", k=2) # TODO: revisit keys for this call...
        summary_data = self.get_rule_data(key_types=["observation"], keys=[obs.structured], value_types=["context", "observation","reasoning", "action"], outcome="winning", k=3) # Get a rule corresponding to a state similar to the current state
        curr_context = self.summarize(obs, summary_data)
        rule_data = self.get_rule_data(key_types=["context"], keys=[curr_context], value_types=["name", "context", "rule_content", "example_data"], outcome="winning", k=3)
        '''
        example_data = self.get_trajectory_data(key_types=["category"], keys=[self.category], value_types=["goal", "observation", "reasoning", "action"], outcome="winning", k=2) # TODO: revisit keys for this call...
        rule_data = self.get_rule_data(trajectory_key_types=["category", "goal"], trajectory_keys=[self.category, self.goal], state_key_types=["observation"], state_keys=[obs.structured], value_types=["rules"], outcome=None, k=3, window=20) # Rules come from successes and failures # Should bring back the actual example data for positive rules, not sure what to do when fetching both positive and negative rules
        print("Example data: ", example_data)
        print("Rule data: ", rule_data)
        input()
        in_context_data = [example_data, rule_data] # TODO: revisit this, should be a single variable with the rule data matching the example data
        if not self.plan:
            await self.create_plan(obs, valid_actions, in_context_data)
        reasoning = await self.reason(obs, valid_actions, in_context_data)
        action = await self.act(obs, valid_actions, reasoning, in_context_data) 
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        if done:
            reflection = self.reflect(new_obs, reward)
            self.store_episode(reflection, None)
        return
    
    async def update_rules_online(self, environment_id):
        """ For AutoManual, this is the builder"""
        #relevant_rules = self.get_rule_data(key_types=["trajectory"], keys=[self.trajectory], value_types=["name", "context", "rule_content"], outcome="winning", k=2)
        await self.generate_rules(mode="vanilla", environment_id=environment_id) # Either an update or a new rule
        return
        
    async def update_rules_offline(self):
        """For AutoManual, this is the consolidator"""
        await self.consolidate_rules()
