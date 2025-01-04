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
        self.in_context_data: Optional[Dict] = None
        
        # Environment info
        self.environment_id: Optional[str] = env.id
        self.goal: Optional[str] = env.goal
        self.category: Optional[str] = env.category if hasattr(env, 'category') else None

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
            # For the things that are full-trajectory, stick them on first
            interleaved_values = ["state", "reasoning", "action", "next_state"]
            # Loop through value_types, first add all non-interleaved values
            for value in value_type:
                if value not in interleaved_values:
                    in_context_string += value + ": " + repr(in_context_data[value]) + ", "
            # Now add the interleaved values
            if "state" in value_type:
                for i in range(len(in_context_data["state"])):
                    for j in range(len(value_type)):
                        if len(in_context_data[value_type[j]]) > i:
                            in_context_string += value_type[j] + " " + str(i+1) + ": " + repr(in_context_data[value_type[j]][i]) + ", "
            return in_context_string

    def get_in_context_data(self, key_type, key, value_type, outcome="winning", k=5) -> List[Dict]:
        """Retrieve in context examples from the database"""
        success_entries, failure_entries = self.db.get_similar_entries(key_type, key, outcome=outcome, k=k)
        # Now figure out which part of the examples to return in-context
        if not isinstance(value_type, list): # Check that this is a list, not a string
            value_type = [value_type]
        # Filter out the keys that are not in the value_type
        success_entries = [{k: v for k, v in entry.items() if k in value_type} for entry in success_entries]
        failure_entries = [{k: v for k, v in entry.items() if k in value_type} for entry in failure_entries]
        return success_entries, failure_entries
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
        self.db.store_episode(self.environment_id, self.goal, self.category,self.observation_history, self.reasoning_history, self.action_history, self.reward_history, self.plan, reflection, summary)

    def create_conversation_old(self, conversation: List[Dict], observation: Observation, available_actions: List[Action], reasoning: Union[str, None] = None) -> List[Dict]:
        """Create a conversation with the observation and action history"""
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

    # User message reflecting the current trajectory (system reflects goal and in-context data)
    def create_conversation(self, keys: List[str], observation: Observation, available_actions: List[Action], reasoning: Union[str, None] = None) -> List[Dict]:
        """Create a conversation with the observation and action history"""
        # First create a dictionary of the current trajectory. Valid keys are "plan", "observation", "reasoning", "action"
        trajectory_dict = {}
        for key in keys:
            if key == "goal":
                trajectory_dict[key] = self.goal
            elif key == "plan":
                trajectory_dict[key] = self.plan
            elif key == "observation":
                trajectory_dict[key] = [self.observation_history[i].structured for i in range(len(self.observation_history))]
                if observation is not None:
                    trajectory_dict[key].append(observation.structured)
            elif key == "reasoning":
                trajectory_dict[key] = [self.reasoning_history[i] for i in range(len(self.reasoning_history))]
                if reasoning is not None:
                    trajectory_dict[key].append(reasoning)
            elif key == "action":
                trajectory_dict[key] = [self.action_history[i].text for i in range(len(self.action_history))]
                #if available_actions is not None:
                #    trajectory_dict[key].append(repr([f"{i+1}. {a.text}" for i, a in enumerate(available_actions)]))
            elif key == "available_actions":
                trajectory_dict[key] = repr([f"{i+1}. {a.text}" for i, a in enumerate(available_actions)])
        # Create a string of the current trajectory by rolling up the trajectory_dict
        trajectory_dict = self.roll_up_trajectory([trajectory_dict])[0]
        # Create a string of the current trajectory
        trajectory_string = "Current trajectory:\n"
        for key in trajectory_dict:
            trajectory_string += f"{key}: {trajectory_dict[key]}\n"
        return trajectory_string
    
    def in_context_prompt(self, system_prompt, in_context_data): # We should integrate the value type into the prompt
        """Create a prompt for the in context data"""
        if "low_level" in in_context_data:
            success, data = in_context_data["low_level"]
            system_prompt += f"\nHere are some low-level examples from episodes that {'successfully achieved' if success else 'failed to achieve'} similar goals:"
            for i, elem in enumerate(data):
                system_prompt += f"\nExample {i+1}:\n" + elem
        if "high_level" in in_context_data:
            success, data = in_context_data["high_level"]
            system_prompt += f"\nHere is some higher-level analysis examples from episodes that {'successfully achieved' if success else 'failed to achieve'} similar goals:"
            for i, elem in enumerate(data):
                system_prompt += f"\nExample {i+1}:\n" + elem
        return system_prompt
    
    def roll_up_trajectory(self, entries):
        interleaved_keys = ["observation", "reasoning", "action"]
        """Roll up the trajectory into a single string"""
        value_type = list(entries[0].keys())
        if any(k in value_type for k in interleaved_keys):
            for i in range(len(entries)):
                trajectory_string = "\n"
                for j in range(len(entries[i][interleaved_keys[0]])):
                    for k in interleaved_keys:
                        if len(entries[i][k]) > j and len(entries[i][k][j]) > 0:
                            trajectory_string += f"{k}: {entries[i][k][j]}\n"
                entries[i]["trajectory"] = trajectory_string
                # Delete the interleaved keys from the dictionary
                for k in interleaved_keys:
                    del entries[i][k]
        return entries
    
    def data_driven_in_context_prompt(self, in_context_data):
        """Create a system prompt containing in-context examples from similar episodes"""
        # Determine which set of entries to use
        success_entries, failure_entries = in_context_data[0], in_context_data[1]
        entries = success_entries if len(success_entries) > 0 else failure_entries
        outcome = "successfully achieved" if len(success_entries) > 0 else "failed to achieve"

        if len(entries) == 0:
            return ""
        
        # So if there are trajectory type keys, we want to interleave them with the other keys
        # If there are any interleaved keys, generate a new key called "trajectory" that contains a string generated from interleaving the elements of the lists corresponding to the interleaved keys
        entries = self.roll_up_trajectory(entries)
        value_type = list(entries[0].keys()) # Keys from entries
        
        #in_context_inputs = [k for k in value_type if k != output_type]
        #in_context_outputs = [output_type]

        # Create description of what we're showing
        #system_prompt = f"Given {', '.join(in_context_inputs)}, generate {', '.join(in_context_outputs)}."
        #system_prompt += f"\nHere are some examples of {', '.join(in_context_inputs)} and corresponding {', '.join(in_context_outputs)} from episodes that {outcome} similar goals:\n"
        system_prompt = f"\nHere are some examples of {', '.join(value_type)} from episodes that {outcome} similar goals:\n"

        # Add each example
        for i, entry in enumerate(entries):
            system_prompt += f"\nExample {i+1}:\n"
            # Add keys
            for k_type in value_type:
                system_prompt += f"{k_type}: {entry[k_type]}\n"

        system_prompt += f"\nFollow the provided examples closely."

        return system_prompt
    
    async def create_plan_old(self, observation: Observation, available_actions: List[Action], in_context_data = None) -> str:
        """Generate a plan for the agent to follow"""
        conversation = []
        system_prompt = f"You are an expert at generating high-level plans of actions to achieve a goal."
        if in_context_data:
            system_prompt = self.in_context_prompt(system_prompt, in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
        #print("Observation", observation)
        conversation = self.create_conversation(conversation, observation, available_actions)
        plan = await generate_plan(conversation, self.goal, observation, self.llm)
        self.plan = plan
        return plan
    
    async def create_plan(self, observation: Observation, available_actions: List[Action], in_context_data = None) -> str:
        """Generate a plan for the agent to follow"""
        conversation = []
        system_prompt = f"You are an expert at generating high-level plans of actions to achieve a goal. "
        if in_context_data:
            system_prompt += self.data_driven_in_context_prompt(in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
        #print("Observation", observation)
        #conversation = self.create_conversation(conversation, observation, available_actions)
        plan = await generate_plan(conversation, self.goal, observation, self.llm)
        self.plan = plan
        return plan
    
    async def reason_old(self, observation: Observation, available_actions, in_context_data = None) -> List[Dict]:
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

    async def reason(self, observation: Observation, available_actions, in_context_data = None) -> List[Dict]:
        """Reason about the conversation and observation"""
        conversation = []
        # Add system prompt
        system_prompt = f"You are an expert at reasoning about the most appropriate action to take towards achieving a goal. "
        if in_context_data:
            system_prompt += self.data_driven_in_context_prompt(in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
        # Add conversation history
        conversation.append({"role": "user", "content": self.create_conversation(["goal", "plan", "observation", "reasoning", "action"], observation, available_actions, None) + "reasoning: "})
        reasoning = await reason(conversation, observation, available_actions, self.llm, self.config)
        return reasoning

    async def act(self, observation: Observation, available_actions: List[Action], reasoning: Union[str, None] = None, in_context_data= None) -> Tuple[Action, List[Dict]]:
        """Select an action from available actions given the current observation"""
        # If the model is gemini, just get the action from the reasoning string
        action = None
        '''
        if True or ("gemini" in self.llm.model.lower() or "together" in self.llm.model.lower()):
            match = re.search(r"action:\s*(.*)", reasoning, re.IGNORECASE)
            if match:
                action = match.group(1).strip()
                action = Action(text=action)
        '''
        if action is None:
            # Create a conversation with observations and actions so far
            conversation = []
            # Want the system prompt to be standardized. Should have environment and goal info, as well as observation and action format. 
            system_prompt = f"""You are an agent in an environment. Given the current observation, you must select an action to take towards achieving the goal: {self.goal}."""
            if in_context_data:
                system_prompt += self.data_driven_in_context_prompt(in_context_data)
            conversation.append({"role": "system", "content": system_prompt})
            conversation.append({"role": "user", "content": self.create_conversation(["goal", "plan", "observation", "reasoning", "action"], observation, available_actions, reasoning) + "action: "})
            
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

    async def act_old(self, observation: Observation, available_actions: List[Action], reasoning: Union[str, None] = None, in_context_data= None) -> Tuple[Action, List[Dict]]:
        """Select an action from available actions given the current observation"""
        # If the model is gemini, just get the action from the reasoning string
        if ("gemini" in self.llm.model.lower() or "together" in self.llm.model.lower()):
            # Look for a number followed by a period
            action = None #re.search(r'\d+\.', reasoning)
            for op in available_actions:
                if op.text in reasoning:
                    action = op
                    break
            if action is None:
                raise ValueError(f"No action found in reasoning: {reasoning}")
            return action

        # Create a conversation with observations and actions so far
        conversation = []
        # Want the system prompt to be standardized. Should have environment and goal info, as well as observation and action format. 
        system_prompt = "You are an expert at selecting actions to achieve a goal. "
        if in_context_data:
            system_prompt += self.data_driven_in_context_prompt(in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": self.create_conversation(["goal", "plan", "observation", "reasoning", "action", "available_actions"], observation, available_actions, reasoning) + "Action: "})
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
    
    async def reflect(self, observation: Observation, reward: float) -> List[Dict]:
        """Reflect on the conversation and observation"""
        conversation = []
        conversation = self.create_conversation(conversation, observation, [])
        reflection = await conversation_reflection(self.goal, conversation, self.llm, reward)
        return reflection
    
    async def summarize(self, obs=None) -> str:
        """Summarize the conversation and observation"""
        conversation = []
        conversation = self.create_conversation(conversation, None, None, None)
        if obs is None:
            summary = await trajectory_summary(self.goal, conversation, self.llm)
        else:
            prev_obs = self.observation_history[-1] if len(self.observation_history) > 0 else Observation(None)
            summary = await observation_summary(self.goal, obs, prev_obs, self.llm)
        # Create an observation object again
        summary = Observation(summary)
        return summary
    
    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        pass
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        pass
        
    def clear_history(self):
        """Clear the agent's history"""
        self.observation_history = []
        self.action_history = []
        self.reasoning_history = []
        self.reward_history = []
        self.plan = None
        self.in_context_data = None
        
    def reset(self):
        """Reset agent state between episodes"""
        self.observation_history = []
        self.action_history = []
        self.reasoning_history = []
        self.reward_history = []
        self.plan = None
        self.reflections = None
        self.in_context_data = None