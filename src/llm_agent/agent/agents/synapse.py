from llm_agent.agent.base_agent import BaseAgent

class Synapse(BaseAgent):
    """
    Synapse agent that focuses on providing in-context examples for reasoning and action.
    
    This agent skips the planning phase and directly retrieves examples
    based on the goal to inform reasoning and action selection.
    """
    
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions):
        """
        Choose an action using the Synapse approach.
        
        Key differences from other agents:
        - No explicit planning phase
        - Uses the same in-context data for both reasoning and action
        """
        # Key decision: Retrieve examples based on goal similarity
        # but don't use them for planning (no planning phase)
        in_context_data = self.retrieve_trajectory_data(
            key_types=["goal"],
            keys=[self.goal],
            value_types=["goal","observation", "reasoning", "action"],
            outcome="winning",
            k=2
        )
        
        # Key decision: Skip the planning phase entirely
        reasoning = await self.reason(obs, valid_actions, in_context_data)
        # Key decision: Use in-context examples for action but not the ones from reasoning
        action = await self.act(obs, valid_actions, reasoning) 
        return action

    async def analyze_episode(self):
        self.clean_history()
