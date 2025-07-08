from llm_agent.agent.base_agent import BaseAgent

class ReAct(BaseAgent):
    """
    ReAct agent that enhances the ReAct paradigm with in-context examples.
    
    This agent retrieves successful trajectories with similar goals and uses them
    to inform planning, reasoning, and action selection.
    """
    
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions):
        """
        Choose an action using the ReAct approach with in-context examples.
        
        Key differences from other agents:
        - Uses goal-based retrieval for in-context examples
        - Same in-context data is used for all three steps (plan, reason, act)
        - Focuses on trajectory-level retrieval
        """
        # Key decision: Retrieve examples based on goal similarity only
        # This is simpler than RAP which uses goal+category
        data = self.retrieve_trajectory_data(
            key_types=["goal"],  # Match examples with similar goals
            keys=[self.goal],
            value_types=["goal", "plan", "observation", "reasoning", "action"],
            outcome="winning",  # Only use successful examples
            k=2
        )
        
        if not self.plan:
            await self.create_plan(obs, valid_actions, in_context_data=data)
        reasoning = await self.reason(obs, valid_actions, in_context_data=data)
        action = await self.act(obs, valid_actions, reasoning, in_context_data=data) 
        return action
    
    async def analyze_episode(self):
        self.clean_history()
