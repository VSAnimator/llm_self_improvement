import os
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from logging import getLogger

from ..env.base_env import State, Action
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
        
        # Keep track of recent states and actions
        self.state_history: List[State] = []
        self.action_history: List[Action] = []
        self.plan: Optional[str] = None
        self.reflexions: Optional[List[str]] = None

    def create_conversation(self, conversation: List[Dict], goal: str, state: State, available_actions: List[Action]) -> List[Dict]:
        """Create a conversation with the state and available actions"""
        conversation.append({"role": "user", "content": "State: " + repr(state)})
        conversation.append({"role": "assistant", "content": "Available actions:\n"})
        for i, action in enumerate(available_actions):
            conversation.append({"role": "assistant", "content": f"{i+1}. {action.text}\n"})
        return conversation

    async def reason(self, goal: str, state: State, available_actions) -> List[Dict]:
        """Reason about the conversation and state"""
        conversation = []
        # Add system prompt
        conversation.append({"role": "system", "content": f"You are an agent in an environment. Given the current state, you must reason about the most appropriate action to take towards achieving the goal: {goal}"})
        # Add conversation history
        conversation = self.create_conversation(conversation, goal, state, available_actions)
        reasoning = await reason(conversation, state, available_actions, self.llm, self.config)
        return reasoning

    async def act(self, goal: str, state: State, available_actions: List[Action]) -> Tuple[Action, List[Dict]]:
        """Select an action from available actions given the current state"""
        # Generate plan
        plan = []
        if self.config.get('generate_plan', False) and len(self.state_history) == 0:
            plan = await generate_plan(goal, state, self.llm)

        # Create a conversation with states and actions so far
        conversation = []
        # Want the system prompt to be standardized. Should have environment and goal info, as well as state and action format. 
        conversation.append({"role": "system", "content": f"You are an agent in an environment. Given the current state, you must select an action to take towards achieving the goal: {goal}"})
        # Add plan to conversation if it exists
        if plan:
            conversation.append({"role": "assistant", "content": "Plan of action: " + plan})
        conversation = self.create_conversation(conversation, goal, state, available_actions)
        
        # Select action
        action = await select_action(conversation, state, available_actions, self.llm, self.config)

        # Append state, action to history
        self.state_history.append(state)
        self.action_history.append(action)

        # Enforce memory size limit
        self.state_history = self.state_history[-self.memory_size:]
        self.action_history = self.action_history[-self.memory_size:]

        return action
    
    async def reflect(self, goal: str, conversation: List[Dict], state: State) -> List[Dict]:
        """Reflect on the conversation and state"""
        trajectory = [(state, action) for action in self.action_history]
        reflexion = await trajectory_reflexion(goal, trajectory, self.llm)
        if self.reflexions is None:
            self.reflexions = [reflexion]
        else:
            self.reflexions.append(reflexion)
        return reflexion
        
    def reset(self):
        """Reset agent state between episodes"""
        self.state_history = []
        self.action_history = []
        self.plan = None
        self.reflexions = None