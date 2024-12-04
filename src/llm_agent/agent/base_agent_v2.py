import os
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from logging import getLogger

from ..env.base_env import Observation, Action
from ..llm.lite_llm import LiteLLMWrapper
from ..agent.generate_plan import generate_plan
from ..agent.choose_action import select_action, reason
from ..agent.trajectory_reflexion import trajectory_reflexion, conversation_reflexion, trajectory_summary, observation_summary
from ..in_context.alfworld_fewshots import get_fewshots_for_goal

logger = getLogger(__name__)

class BaseAgent:
    """Base agent class that uses an LLM to select actions"""
    
    def __init__(self, llm, db, env, config):
        """Initialize the agent
        
        Args:
            llm: LLM instance to use for decision making
            config: Configuration dictionary containing agent parameters
        """
        self.config = config

        # LLM
        self.llm = llm
        self.max_retries = config.get('max_retries', 3)
        self.memory_size = config.get('memory_size', 20)
        self.temperature = config.get('temperature', 0.7)
        
        # Trajectory history
        self.observation_history: List[Observation] = []
        self.reasoning_history: List[str] = []
        self.action_history: List[Action] = []
        self.reward_history: List[float] = []
        self.plan: Optional[str] = None
        self.reflexions: Optional[List[str]] = None
        
        # Environment info
        self.environment_id: Optional[str] = env.id
        self.goal: Optional[str] = env.goal

        # Database
        self.db = db

    def get_in_context_data(self, key_type, key, value_type) -> List[Dict]:
        """Retrieve in context examples from the database"""
        similar_entries = self.db.get_similar_entries(key_type, key)
        # Now figure out which part of the examples to return in-context
        # Options:
        # 1. Return the entire trajectory
        # 2. Return the summary
        # 3. Return the action and observation pairs
        # 4. Return the reflexion
        in_context_examples = []
        print("Similar entries", similar_entries)
        if isinstance(value_type, list): # Check that this is a list, not a string
            in_context_examples = [entry[value_type[i]] for entry in similar_entries for i in range(len(value_type))]
        elif value_type == 'trajectory':
            in_context_examples = [entry['trajectory'] for entry in similar_entries]
        elif value_type == 'summary':
            in_context_examples = [entry['summary'] for entry in similar_entries]
        elif value_type == 'action':
            in_context_examples = [[entry['observation'], entry['action']] for entry in similar_entries]
        elif value_type == 'reflexion':
            in_context_examples = [entry['reflexion'] for entry in similar_entries]
        return in_context_examples
    
    def store_episode(self, reflexion, summary):
        """Store an episode in the database"""
        self.db.store_episode(self.environment_id, self.goal, self.observation_history, self.reasoning_history, self.action_history, self.reward_history, self.plan, reflexion, summary)

    def create_conversation(self, conversation: List[Dict], observation: Observation, available_actions: List[Action], reasoning: Union[str, None] = None, in_context: bool = False) -> List[Dict]:
        """Create a conversation with the observation and action history"""
        if in_context:
            # Add in fewshots
            fewshots = self.retrieve_in_context(self.goal, observation, available_actions, reasoning)
            for i, elem in enumerate(fewshots):
                # Have to have a prompt for the type of fewshot we are using
                fewshot_config = self.config.get('in_context_queries', [])[i]
                fewshot_type = fewshot_config.get('in_context_type', 'trajectory')
                for fewshot in elem:
                    observations, actions = fewshot
                    # Add observation and action history
                    fewshot_prompt = f"Here is an example of a goal and a sequence of observations and actions that achieves a similar goal.\nType of example: {fewshot_type}\n"
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
        if observation is not None:
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

    async def create_plan(self, observation: Observation, available_actions: List[Action], in_context_data = None) -> str:
        """Generate a plan for the agent to follow"""
        conversation = []
        conversation.append({"role": "system", "content": f"You are an expert at generating high-level plans of actions to achieve a goal."})
        print("Observation", observation)
        conversation = self.create_conversation(conversation, observation, available_actions, in_context=False)
        current_reflection = None
        if self.config.get('always_reflect', False) and len(self.reflexions) > 0: # If we always refelct, re-generate plan
            current_reflection = self.reflexions[-1]
        plan = await generate_plan(conversation, self.goal, observation, self.llm, self.plan, current_reflection)
        self.plan = plan
        return plan

    async def reason(self, observation: Observation, available_actions, in_context_data = None) -> List[Dict]:
        """Reason about the conversation and observation"""
        conversation = []
        # Add system prompt
        conversation.append({"role": "system", "content": f"You are an agent in an environment. Given the current observation, you must reason about the most appropriate action to take towards achieving the goal: {self.goal}"})
        # Add conversation history
        conversation = self.create_conversation(conversation, observation, available_actions)
        reasoning = await reason(conversation, observation, available_actions, self.llm, self.config)
        return reasoning

    async def act(self, observation: Observation, available_actions: List[Action], reasoning: Union[str, None] = None, in_context_data= None) -> Tuple[Action, List[Dict]]:
        """Select an action from available actions given the current observation"""
        # Create a conversation with observations and actions so far
        conversation = []
        # Want the system prompt to be standardized. Should have environment and goal info, as well as observation and action format. 
        conversation.append({"role": "system", "content": f"You are an agent in an environment. Given the current observation, you must select an action to take towards achieving the goal: {self.goal}"})

        conversation = self.create_conversation(conversation, observation, available_actions, reasoning)
        
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
    
    async def reflect(self, observation: Observation, in_context_data = None) -> List[Dict]:
        """Reflect on the conversation and observation"""
        conversation = []
        conversation = self.create_conversation(conversation, observation, [])
        reflexion = await conversation_reflexion(self.goal, conversation, self.llm)
        if self.reflexions is None:
            self.reflexions = [reflexion]
        else:
            self.reflexions.append(reflexion)
        return reflexion
    
    async def summarize(self, obs=None, in_context_data = None) -> str:
        """Summarize the conversation and observation"""
        conversation = []
        conversation = self.create_conversation(conversation, None, None, None)
        if obs is None:
            summary = await trajectory_summary(self.goal, conversation, self.llm)
        else:
            summary = await observation_summary(self.goal, obs, conversation, self.llm)
        return summary
    
    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        # Create observation from obs string
        obs = Observation(structured=obs)
        # Create action objects from valid action strings
        valid_actions = [Action(text=action) for action in valid_actions]
        if self.config.get('use_summarization', False):
            original_obs = obs
            obs = await self.summarize(self.goal, obs) # Create_conversation can pull in the trajectory
        # Agent config should control the behavior here, reflect all algorithms we want to encompass
        if self.config.get('use_plan', True):
            await self.create_plan(self.goal, obs, valid_actions) # Re-planning based off reflexion can go in here
        if self.config.get('use_reasoning', True):
            reasoning = await self.reason(self.goal, obs, valid_actions)
        action = await self.act(self.goal, obs, valid_actions, reasoning) # This is where we store everything too.
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        new_obs = Observation(structured=new_obs)
        reflexion = None
        summary = None
        if self.config.get('use_reflexion', True) and ((done and reward < 1) or self.config.get('always_reflect', False)):
            reflexion = await self.reflect(self.goal, new_obs) # Handle this separately if in progress or not. In theory this could take more inputs from the reasoning chain...
            if self.config.get('always_reflect', False) and (not done and len(self.reflexions) > 0):
                # We should override the existing reflexion
                self.reflexions[-1] = reflexion
            else:
                self.reflexions.append(reflexion)
        if self.config.get('use_summarization', True) and done:
            summary = await self.summarize(self.goal, new_obs)
            print("Summary", summary)
        if self.config.get('use_memory', True) and done and reward == 1:
            # We need to add to the database here
            await self.store_episode(self.goal, self.observation_history, self.reasoning_history, self.action_history, self.reward_history, self.plan, self.reflexions, reflexion, summary)
        
    def clear_history(self):
        """Clear the agent's history"""
        self.observation_history = []
        self.action_history = []
        self.reasoning_history = []
        self.reward_history = []
        self.plan = None

    def reset(self):
        """Reset agent state between episodes"""
        self.observation_history = []
        self.action_history = []
        self.reasoning_history = []
        self.reward_history = []
        self.plan = None
        self.reflexions = None