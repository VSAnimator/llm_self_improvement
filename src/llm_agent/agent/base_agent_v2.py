import os
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from logging import getLogger
import re
from ..env.base_env import Observation, Action
from ..llm.lite_llm import LiteLLMWrapper
from ..agent.generate_plan import generate_plan
from ..agent.choose_action import select_action, reason
from ..agent.trajectory_reflection import trajectory_reflection, conversation_reflection, trajectory_summary, observation_summary
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
        self.reflections: Optional[List[str]] = None
        
        # Environment info
        self.environment_id: Optional[str] = env.id
        self.goal: Optional[str] = env.goal

        # Database
        self.db = db

    def create_in_context_string(self, in_context_data, value_type):
        """Create a string for the in context data"""
        #print("In context data", in_context_data)
        #print("Value type", value_type)
        # So we basically want to unroll the examples into a string
        if len(value_type) == 1:
            return value_type[0] + ": " + repr(in_context_data[value_type[0]])
        else:
            # If there are multiple value types, we want to interleave them. Ex. Obs 1, Action 1, Obs 2, Action 2, etc.
            in_context_string = ""
            for i in range(len(in_context_data[value_type[0]])):
                for j in range(len(value_type)):
                    in_context_string += value_type[j] + " " + str(i+1) + ": " + repr(in_context_data[value_type[j]][i]) + ", "
            return in_context_string

    def get_in_context_data(self, key_type, key, value_type, outcome="winning", k=5) -> List[Dict]:
        """Retrieve in context examples from the database"""
        success_entries, failure_entries = self.db.get_similar_entries(key_type, key, outcome=outcome, k=k)
        # Now figure out which part of the examples to return in-context
        if not isinstance(value_type, list): # Check that this is a list, not a string
            value_type = [value_type]
        in_context_examples_success = [self.create_in_context_string(entry, value_type) for entry in success_entries]
        in_context_examples_failure = [self.create_in_context_string(entry, value_type) for entry in failure_entries]
        # Now we return a dict where the keys are low/high level, then the values have success labels with list of examples
        in_context_data = {}
        value_abstraction = "low_level" if "state" in value_type or "action" in value_type else "high_level"
        if len(success_entries) > 0:
            in_context_data[value_abstraction] = (True, in_context_examples_success)
        if len(failure_entries) > 0:
            in_context_data[value_abstraction] = (False, in_context_examples_failure)
        return in_context_data
    
    def store_episode(self, reflection, summary):
        """Store an episode in the database"""
        self.db.store_episode(self.environment_id, self.goal, self.observation_history, self.reasoning_history, self.action_history, self.reward_history, self.plan, reflection, summary)

    def create_conversation(self, conversation: List[Dict], observation: Observation, available_actions: List[Action], reasoning: Union[str, None] = None) -> List[Dict]:
        """Create a conversation with the observation and action history"""
        # Add on reflections from previous episodes
        if self.reflections:
            for i, reflection in enumerate(self.reflections):
                conversation.append({"role": "user", "content": "Reflection on attempt " + str(i+1) + ": " + reflection})
        # Add on plan
        if self.plan:
            conversation.append({"role": "user", "content": "Plan of action: " + self.plan})
        for i in range(len(self.observation_history)):
            conversation.append({"role": "user", "content": f"Observation {i+1}: " + repr(self.observation_history[i].structured)}) # Also have image observation option
            #if self.reasoning_history and len(self.reasoning_history) > i:
            #    conversation.append({"role": "user", "content": f"Reasoning {i+1}: " + self.reasoning_history[i]}) # Unclear if reasoning should be included in the conversation history
            conversation.append({"role": "assistant", "content": f"Action {i+1}: " + repr(self.action_history[i].text)})
        if observation is not None:
            curr_prompt = f"Observation {len(self.observation_history)+1}: " + repr(observation.structured) # Also have image observation option
            if available_actions and len(available_actions) > 0:
                curr_prompt += "\n Available actions: \n"
                for i, action in enumerate(available_actions):
                    curr_prompt += f"{i+1}. {action.text}, "
            if reasoning:
                curr_prompt += "\n Follow this reasoning: " + reasoning
            conversation.append({"role": "user", "content": curr_prompt})
        if True:
            # Collapse everything after the system prompt into a single user message
            for i in range(len(conversation)-2, -1, -1):
                if conversation[i]['role'] == 'system':
                    break
                else:
                    conversation[i]['content'] += ". " + conversation[i+1]['content']
                    del conversation[i+1]
        return conversation
    
    def in_context_prompt(self, system_prompt, in_context_data): # We should integrate the value type into the prompt
        """Create a prompt for the in context data"""
        if "low_level" in in_context_data:
            success, data = in_context_data["low_level"]
            system_prompt += f"\nHere are some low-level examples from episodes that {'successfully achieved' if success else 'failed to achieve'} similar goals:"
            for i, elem in enumerate(data):
                system_prompt += f"\nExample {i+1}:\n" + elem
        if "high_level" in in_context_data:
            success, data = in_context_data["high_level"]
            system_prompt += f"\nHere is some higher-level analysis examples that {'successfully achieved' if success else 'failed to achieve'} similar goals:"
            for i, elem in enumerate(data):
                system_prompt += f"\nExample {i+1}:\n" + elem
        return system_prompt
    
    async def create_plan(self, observation: Observation, available_actions: List[Action], in_context_data = None) -> str:
        """Generate a plan for the agent to follow"""
        conversation = []
        system_prompt = f"You are an expert at generating high-level plans of actions to achieve a goal."
        if in_context_data:
            system_prompt = self.in_context_prompt(system_prompt, in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
        #print("Observation", observation)
        conversation = self.create_conversation(conversation, observation, available_actions)
        current_reflection = None
        if self.config.get('always_reflect', False) and len(self.reflections) > 0: # If we always refelct, re-generate plan
            current_reflection = self.reflections[-1]
        plan = await generate_plan(conversation, self.goal, observation, self.llm, self.plan, current_reflection)
        self.plan = plan
        return plan

    async def reason(self, observation: Observation, available_actions, in_context_data = None) -> List[Dict]:
        """Reason about the conversation and observation"""
        conversation = []
        # Add system prompt
        system_prompt = f"You are an agent in an environment. Given the current observation, you must reason about the most appropriate action to take towards achieving the goal: {self.goal}"
        if in_context_data:
            system_prompt = self.in_context_prompt(system_prompt, in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
        # Add conversation history
        conversation = self.create_conversation(conversation, observation, available_actions)
        reasoning = await reason(conversation, observation, available_actions, self.llm, self.config)
        return reasoning

    async def act(self, observation: Observation, available_actions: List[Action], reasoning: Union[str, None] = None, in_context_data= None) -> Tuple[Action, List[Dict]]:
        """Select an action from available actions given the current observation"""
        # If the model is gemini, just get the action from the reasoning string
        if "gemini" in self.llm.model.lower():
            # Look for a number followed by a period
            action = re.search(r'\d+\.', reasoning)
            for op in available_actions:
                if op.text in reasoning:
                    action = op
                    break
            if action is None:
                raise ValueError(f"No action found in reasoning: {reasoning}")
            #print("Action", action)
            #input("waiting")
            return action

        # Create a conversation with observations and actions so far
        conversation = []
        # Want the system prompt to be standardized. Should have environment and goal info, as well as observation and action format. 
        system_prompt = f"You are an agent in an environment. Given the current observation, you must select an action to take towards achieving the goal: {self.goal}"
        if in_context_data:
            system_prompt = self.in_context_prompt(system_prompt, in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
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
        reflection = await conversation_reflection(self.goal, conversation, self.llm)
        if self.reflections is None:
            self.reflections = [reflection]
        else:
            self.reflections.append(reflection)
        return reflection
    
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
            await self.create_plan(self.goal, obs, valid_actions) # Re-planning based off reflection can go in here
        if self.config.get('use_reasoning', True):
            reasoning = await self.reason(self.goal, obs, valid_actions)
        action = await self.act(self.goal, obs, valid_actions, reasoning) # This is where we store everything too.
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        new_obs = Observation(structured=new_obs)
        reflection = None
        summary = None
        if self.config.get('use_reflection', True) and ((done and reward < 1) or self.config.get('always_reflect', False)):
            reflection = await self.reflect(self.goal, new_obs) # Handle this separately if in progress or not. In theory this could take more inputs from the reasoning chain...
            if self.config.get('always_reflect', False) and (not done and len(self.reflections) > 0):
                # We should override the existing reflection
                self.reflections[-1] = reflection
            else:
                self.reflections.append(reflection)
        if self.config.get('use_summarization', True) and done:
            summary = await self.summarize(self.goal, new_obs)
            print("Summary", summary)
        if self.config.get('use_memory', True) and done and reward == 1:
            # We need to add to the database here
            await self.store_episode(self.goal, self.observation_history, self.reasoning_history, self.action_history, self.reward_history, self.plan, self.reflections, reflection, summary)
        
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
        self.reflections = None