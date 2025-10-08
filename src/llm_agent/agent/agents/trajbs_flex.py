from llm_agent.agent.base_agent import BaseAgent


class TrajBSFlex(BaseAgent):
    """
    Default TrajBS implementation that combines trajectory-level and state-level retrieval.

    This agent dynamically updates in-context examples based on the current reasoning state.
    """

    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions):
        """
        Choose an action using the TrajBS-Flex approach.

        Key differences from other agents:
        - Uses trajectory-level retrieval for planning (goal+category keys)
        - Uses state-level retrieval for reasoning and action
        - Dynamically updates in-context examples based on current reasoning
        - Combines trajectory and state-level retrieval for more targeted examples
        """
        if not self.plan:
            # Key decision: Use trajectory-level retrieval for planning
            # Retrieve examples based on goal and category similarity
            plan_data = self.retrieve_trajectory_data(
                key_types=[
                    "goal",
                    "category",
                ],  # Match examples with similar goals and categories
                keys=[self.goal, self.category],
                value_types=["goal", "plan"],  # Only need goal and plan for planning
                outcome="winning",  # Only use successful examples
                k=self.num_ic,
            )
            await self.create_plan(obs, valid_actions, in_context_data=plan_data)

        if self.in_context_data is None:
            # Key decision: Initial state-level retrieval based on observation
            # This combines trajectory-level keys (goal, category, plan) with state-level keys (observation)
            self.in_context_data = self.retrieve_state_data(
                trajectory_key_types=[
                    "goal",
                    "category",
                    "plan",
                ],  # First find trajectories with similar goals, categories, and plans
                trajectory_keys=[self.goal, self.category, self.plan],
                state_key_types=[
                    "observation"
                ],  # Then find states with similar observations within those trajectories
                state_keys=[obs.structured],
                value_types=[
                    "goal",
                    "observation",
                    "reasoning",
                    "action",
                ],  # Extract these values from the matched states
                outcome="winning",
                k=self.num_ic,
                window=5,  # Include 5 states before and after the matched state
            )

        # Generate reasoning using the current in-context data
        reasoning = await self.reason(
            obs, valid_actions, in_context_data=self.in_context_data
        )

        # Key decision: Update in-context data based on the current reasoning
        # This is the most distinctive feature of TrajBS-Flex - dynamically updating context
        self.in_context_data = self.retrieve_state_data(
            trajectory_key_types=["goal", "category", "plan"],
            trajectory_keys=[self.goal, self.category, self.plan],
            state_key_types=["reasoning"],  # Now find states with similar reasoning
            state_keys=[reasoning],
            value_types=["goal", "plan", "observation", "reasoning", "action"],
            outcome="winning",
            k=self.num_ic,
            window=5,
        )

        # Select action using the updated in-context data
        action = await self.act(
            obs, valid_actions, reasoning, in_context_data=self.in_context_data
        )
        return action

    async def analyze_episode(self):
        """Clean up the agent's history after an episode"""
        self.clean_history()
