from agent_algo_bench.src.llm_agent.agent.base_agent import BaseAgent
from ...env.base_env import Observation, Action

class TrajBSNoPlan(BaseAgent):
    """
    TrajBS variant that skips the planning phase.
    """
    
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions):
        """
        Choose an action using the TrajBS-NoPlan approach.
        
        Key differences from other agents:
        - Skips the planning phase entirely
        - Uses goal+observation keys for initial retrieval
        - Updates in-context examples based on reasoning
        """
        if self.in_context_data is None:
            # Key decision: Initial retrieval based on goal and observation
            # This is different from TrajBS-Flex which uses goal+category for planning
            # and then goal+category+plan+observation for reasoning
            self.in_context_data = self.get_trajectory_data(
                key_types=["goal", "observation"],  # Match examples with similar goals and observations
                keys=[self.goal, obs.structured],
                value_types=["goal", "observation", "reasoning", "action"],  # Extract full state information
                outcome="winning",  # Only use successful examples
                k=self.num_ic
            )
        
        # Generate reasoning using the current in-context data
        reasoning = await self.reason(obs, valid_actions, in_context_data=self.in_context_data)
        
        # Key decision: Update in-context data based on the current reasoning
        # Similar to TrajBS-Flex but uses trajectory-level retrieval instead of state-level
        self.in_context_data = self.get_state_data(
            trajectory_key_types=["goal", "reasoning"],  # Find trajectories with similar goals and reasoning
            trajectory_keys=[self.goal, reasoning],
            state_key_types=["reasoning"],  # Find states with similar reasoning within those trajectories
            state_keys=[reasoning],
            value_types=["goal", "observation", "reasoning", "action"],
            outcome="winning",
            k=self.num_ic,
            window=5
        )
        
        # Select action using the updated in-context data
        action = await self.act(obs, valid_actions, reasoning, in_context_data=self.in_context_data) 
        return action
    
    async def analyze_episode(self):
        """Clean up the agent's history after an episode"""
        self.clean_history()
        
