from llm_agent.agent.base_agent import BaseAgent


class ZeroShotReact(BaseAgent):
    """
    Basic ReAct agent that uses no in-context examples.

    This agent relies solely on the LLM's capabilities without retrieving
    any examples from the database. It follows the standard ReAct pattern
    of planning, reasoning, and then acting.
    """

    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions):
        """
        Choose an action using the zero-shot ReAct approach.

        Key differences from other agents:
        - No in-context examples are used (pure zero-shot)
        - Still follows plan → reason → act pattern
        """
        if not self.plan:
            # Create plan without any in-context examples
            await self.create_plan(obs, valid_actions)
        # Generate reasoning without any in-context examples
        reasoning = await self.reason(obs, valid_actions)
        # Select action without any in-context examples
        action = await self.act(obs, valid_actions, reasoning)
        return action

    async def analyze_episode(self):
        self.clean_history()
