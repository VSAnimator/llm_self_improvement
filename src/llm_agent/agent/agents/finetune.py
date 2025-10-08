from typing import Any, Dict, List, Optional, Tuple

from ...env.base_env import Action, Observation
from ..base_agent import BaseAgent


class FinetuneAgent(BaseAgent):
    """
    Agent that uses a fine-tuned LLM to select actions.

    This agent bypasses the standard reasoning and planning steps,
    directly mapping observations to actions using a fine-tuned model.
    It requires no in-context examples, relying instead on knowledge
    embedded in the model weights.
    """

    def __init__(self, llm, db, env, config):
        """Initialize the fine-tuned agent"""
        super().__init__(llm, db, env, config)

    async def choose_action(
        self, obs: Observation, valid_actions
    ) -> Tuple[Action, List[Dict]]:
        """
        Choose an action using the fine-tuned model approach.

        Key differences from other agents:
        - No planning or explicit reasoning steps
        - No in-context examples used
        - Directly maps observations to actions using fine-tuned model
        - Both reasoning and action are extracted from a single model call
        """
        # Key decision: Store the observation but don't use it for retrieval
        self.observation_history.append(obs)

        # Key decision: Use the specialized act_finetune method instead of the standard
        # create_plan -> reason -> act pipeline
        action = await self.act_finetune(obs)
        return action

    async def analyze_episode(self):
        """Clean up the agent's history after an episode"""
        self.clean_history()
