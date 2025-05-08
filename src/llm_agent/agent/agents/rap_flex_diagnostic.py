from llm_agent.agent.base_agent_v2 import BaseAgent
from ...env.base_env import Observation, Action

class RAPFlexDiagnostic(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions):
        if not self.plan:
            plan_data = self.get_trajectory_data(key_types=["goal", "category"], keys=[self.goal, self.category], value_types=["goal", "plan"], outcome="winning", k=self.num_ic)
            # Get diagnostic data with different k values
            plan_data_diagnostic_1x = self.get_trajectory_data(key_types=["goal", "category"], keys=[self.goal, self.category], value_types=["rewards"], outcome=None, k=self.num_ic) # 1x examples
            self.f.write(f"Probability of success 1x: {len(plan_data_diagnostic_1x[0])/(len(plan_data_diagnostic_1x[0])+len(plan_data_diagnostic_1x[1]))}\n")
            
            plan_data_diagnostic_2x = self.get_trajectory_data(key_types=["goal", "category"], keys=[self.goal, self.category], value_types=["rewards"], outcome=None, k=2*self.num_ic) # 2x examples
            self.f.write(f"Probability of success 2x: {len(plan_data_diagnostic_2x[0])/(len(plan_data_diagnostic_2x[0])+len(plan_data_diagnostic_2x[1]))}\n")
            
            plan_data_diagnostic_3x = self.get_trajectory_data(key_types=["goal", "category"], keys=[self.goal, self.category], value_types=["rewards"], outcome=None, k=3*self.num_ic) # 3x examples
            self.f.write(f"Probability of success 3x: {len(plan_data_diagnostic_3x[0])/(len(plan_data_diagnostic_3x[0])+len(plan_data_diagnostic_3x[1]))}\n")
            
            self.f.flush()
            await self.create_plan(obs, valid_actions, in_context_data=plan_data) 
        if self.in_context_data is None:
            self.in_context_data = self.get_state_data(trajectory_key_types=["goal", "category", "plan"], trajectory_keys=[self.goal, self.category, self.plan], state_key_types=["observation"], state_keys=[obs.structured], value_types=["goal", "observation", "reasoning", "action"], outcome="winning", k=self.num_ic, window=5)
        reasoning = await self.reason(obs, valid_actions, in_context_data=self.in_context_data)
        self.in_context_data = self.get_state_data(trajectory_key_types=["goal", "category", "plan"], trajectory_keys=[self.goal, self.category, self.plan], state_key_types=["reasoning"], state_keys=[reasoning], value_types=["goal", "plan", "observation", "reasoning", "action"], outcome="winning", k=self.num_ic, window=5)
        # Need to figure out how many steps are left for the task
        steps_left = 30 - len(self.action_history)
        self.f.write(f"Steps left: {steps_left}\n")
        self.f.flush()
        diagnostic_data = self.get_state_data(trajectory_key_types=["goal", "category", "plan"], trajectory_keys=[self.goal, self.category, self.plan], state_key_types=["reasoning"], state_keys=[reasoning], value_types=["rewards"], outcome=None, k=self.num_ic*3, window=steps_left)
        # For the winning trajectories, see if 1 is present in the rewards
        winning_count = 0
        for trajectory in diagnostic_data[0]:
            print(f"Trajectory: {trajectory}")
            if 1 in trajectory["rewards"]:
                winning_count += 1
        self.f.write(f"Win probability: {winning_count/(len(diagnostic_data[0])+len(diagnostic_data[1]))}\n")
        self.f.flush()
        action = await self.act(obs, valid_actions, reasoning, in_context_data=self.in_context_data) 
        return action
    
    async def analyze_episode(self):
        """Process feedback from the environment"""
        self.clean_history()
