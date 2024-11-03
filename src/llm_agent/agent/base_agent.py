import os
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from logging import getLogger

from ..env.base_env import Observation, Action
from ..llm.lite_llm import LiteLLMWrapper
from ..agent.generate_plan import generate_plan
from ..agent.choose_action import select_action, reason
from ..agent.trajectory_reflexion import trajectory_reflexion

logger = getLogger(__name__)

class BaseAgent:
    """Base agent class that uses an LLM to select actions"""
    
    def __init__(self, llm: LiteLLMWrapper, config: Dict):
        """Initialize the agent
        
        Args:
            llm: LLM instance to use for decision making
            config: Configuration dictionary containing agent parameters
        """
        self.llm = llm
        self.config = config
        self.max_retries = config.get('max_retries', 3)
        self.memory_size = config.get('memory_size', 10)
        self.temperature = config.get('temperature', 0.7)
        
        # Keep track of recent observations and actions
        self.observation_history: List[Observation] = []
        self.action_history: List[Action] = []
        self.plan: Optional[str] = None
        self.reflexions: Optional[List[str]] = None

    def create_conversation(self, conversation: List[Dict], goal: str, observation: Observation, available_actions: List[Action], reasoning: Union[str, None] = None) -> List[Dict]:
        """Create a conversation with the observation and action history"""
        for i in range(len(self.observation_history)):
            conversation.append({"role": "user", "content": f"Observation {i+1}: " + repr(self.observation_history[i].structured)})
            conversation.append({"role": "assistant", "content": f"Action {i+1}: " + repr(self.action_history[i].text)})
        curr_prompt = f"Current observation: " + repr(observation.structured) + "\nAvailable actions:\n"
        for i, action in enumerate(available_actions):
            curr_prompt += f"{i+1}. {action.text}\n"
        if reasoning:
            curr_prompt += "\nReasoning: " + reasoning
        conversation.append({"role": "user", "content": curr_prompt})
        if True:
            # Collapse everything after the system prompt into a single user message
            for i in range(len(conversation)-2, -1, -1):
                if conversation[i]['role'] == 'system':
                    break
                else:
                    conversation[i]['content'] += "\n" + conversation[i+1]['content']
                    del conversation[i+1]
        return conversation

    async def reason(self, goal: str, observation: Observation, available_actions) -> List[Dict]:
        """Reason about the conversation and observation"""
        conversation = []
        # Add system prompt
        conversation.append({"role": "system", "content": f"You are an agent in an environment. Given the current observation, you must reason about the most appropriate action to take towards achieving the goal: {goal}"})
        # Add conversation history
        conversation = self.create_conversation(conversation, goal, observation, available_actions)
        reasoning = await reason(conversation, observation, available_actions, self.llm, self.config)
        return reasoning

    async def act(self, goal: str, observation: Observation, available_actions: List[Action], reasoning: Union[str, None] = None) -> Tuple[Action, List[Dict]]:
        """Select an action from available actions given the current observation"""
        # Generate plan
        plan = []
        if self.config.get('generate_plan', False) and len(self.observation_history) == 0:
            plan = await generate_plan(goal, observation, self.llm)

        # Create a conversation with observations and actions so far
        conversation = []
        # Want the system prompt to be standardized. Should have environment and goal info, as well as observation and action format. 
        conversation.append({"role": "system", "content": f"You are an agent in an environment. Given the current observation, you must select an action to take towards achieving the goal: {goal}"})
        # Add plan to conversation if it exists
        if plan:
            conversation.append({"role": "assistant", "content": "Plan of action: " + plan})
        conversation = self.create_conversation(conversation, goal, observation, available_actions, reasoning)
        
        # Select action
        action = await select_action(conversation, observation, available_actions, self.llm, self.config)

        # Append observation, action to history
        self.observation_history.append(observation)
        self.action_history.append(action)

        # Enforce memory size limit
        self.observation_history = self.observation_history[-self.memory_size:]
        self.action_history = self.action_history[-self.memory_size:]

        return action
    
    async def reflect(self, goal: str, conversation: List[Dict], observation: Observation) -> List[Dict]:
        """Reflect on the conversation and observation"""
        trajectory = [(observation, action) for action in self.action_history]
        reflexion = await trajectory_reflexion(goal, trajectory, self.llm)
        if self.reflexions is None:
            self.reflexions = [reflexion]
        else:
            self.reflexions.append(reflexion)
        return reflexion
        
    def reset(self):
        """Reset agent state between episodes"""
        self.observation_history = []
        self.action_history = []
        self.plan = None
        self.reflexions = None