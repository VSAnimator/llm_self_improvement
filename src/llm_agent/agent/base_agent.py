import os
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from logging import getLogger

from ..env.base_env import Observation, Action
from ..llm.lite_llm import LiteLLMWrapper
from ..agent.generate_plan import generate_plan
from ..agent.choose_action import select_action, reason
from ..agent.trajectory_reflexion import trajectory_reflexion, conversation_reflexion
from ..in_context.alfworld_fewshots import get_fewshots_for_goal

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
        self.reasoning_history: List[str] = []
        self.action_history: List[Action] = []
        self.plan: Optional[str] = None
        self.reflexions: Optional[List[str]] = None

    def create_conversation(self, conversation: List[Dict], goal: str, observation: Observation, available_actions: List[Action], reasoning: Union[str, None] = None, fewshots: bool = True) -> List[Dict]:
        """Create a conversation with the observation and action history"""
        if fewshots and False:
            # Add in fewshots
            fewshots = get_fewshots_for_goal(goal)[0:1] # Only use one fewshot for now
            for fewshot in fewshots:
                observations, actions = fewshot
                # Add observation and action history
                fewshot_prompt = "Here is an example of a goal and a sequence of observations and actions that achieves a similar goal.\n"
                for i in range(len(observations)):
                    if i == 0:
                        fewshot_prompt += f"Goal: " + repr(observations[i]) + "\n"
                    else:
                        fewshot_prompt += f"Observation {i+1}: " + repr(observations[i]) + "\n"
                    if i < len(actions):
                        fewshot_prompt += f"Action {i+1}: " + repr(actions[i]) + "\n"
                conversation.append({"role": "user", "content": fewshot_prompt})
        # Add on reflexions from previous episodes
        if self.reflexions:
            for i, reflexion in enumerate(self.reflexions):
                conversation.append({"role": "user", "content": "Reflexion on attempt " + str(i+1) + ": " + reflexion})
        # Add on plan
        if self.plan:
            conversation.append({"role": "user", "content": "Plan of action: " + self.plan})
        for i in range(len(self.observation_history)):
            conversation.append({"role": "user", "content": f"Observation {i+1}: " + repr(self.observation_history[i].structured)}) # Also have image observation option
            #if self.reasoning_history and len(self.reasoning_history) > i:
            #    conversation.append({"role": "user", "content": f"Reasoning {i+1}: " + self.reasoning_history[i]}) # Unclear if reasoning should be included in the conversation history
            conversation.append({"role": "assistant", "content": f"Action {i+1}: " + repr(self.action_history[i].text)})
        curr_prompt = f"Current observation: " + repr(observation.structured) # Also have image observation option
        if available_actions and len(available_actions) > 0:
            curr_prompt += "\nAvailable actions:\n"
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

    async def create_plan(self, goal: str, observation: Observation, available_actions: List[Action]) -> str:
        """Generate a plan for the agent to follow"""
        conversation = []
        conversation.append({"role": "system", "content": f"You are an expert at generating high-level plans of actions to achieve a goal."})
        conversation = self.create_conversation(conversation, goal, observation, available_actions, fewshots=False)
        plan = await generate_plan(conversation, goal, observation, self.llm)
        self.plan = plan
        return plan

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
        # Create a conversation with observations and actions so far
        conversation = []
        # Want the system prompt to be standardized. Should have environment and goal info, as well as observation and action format. 
        conversation.append({"role": "system", "content": f"You are an agent in an environment. Given the current observation, you must select an action to take towards achieving the goal: {goal}"})

        conversation = self.create_conversation(conversation, goal, observation, available_actions, reasoning)
        
        # Select action
        action = await select_action(conversation, observation, available_actions, self.llm, self.config)

        # Append observation, action to history
        self.observation_history.append(observation)
        if reasoning:
            print("Appending reasoning")
            self.reasoning_history.append(reasoning)
        else:
            print("Not appending reasoning because", self.reasoning_history, reasoning)
        self.action_history.append(action)

        # Enforce memory size limit
        self.observation_history = self.observation_history[-self.memory_size:]
        self.action_history = self.action_history[-self.memory_size:]

        return action
    
    async def reflect(self, goal: str, observation: Observation) -> List[Dict]:
        """Reflect on the conversation and observation"""
        conversation = []
        conversation = self.create_conversation(conversation, goal, observation, [])
        reflexion = await conversation_reflexion(goal, conversation, self.llm)
        if self.reflexions is None:
            self.reflexions = [reflexion]
        else:
            self.reflexions.append(reflexion)
        return reflexion
        
    def clear_history(self):
        """Clear the agent's history"""
        self.observation_history = []
        self.action_history = []
        self.reasoning_history = []
        self.plan = None

    def reset(self):
        """Reset agent state between episodes"""
        self.observation_history = []
        self.action_history = []
        self.reasoning_history = []
        self.plan = None
        self.reflexions = None