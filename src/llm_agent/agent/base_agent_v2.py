import os
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from logging import getLogger
from ..env.base_env import Observation, Action
from ..agent.generate_plan import generate_plan
from ..agent.trajectory_reflection import trajectory_summary, observation_summary

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
        self.memory_size = config.get('memory_size', 50)
        self.temperature = config.get('temperature', 0.7)
        self.num_ic = config.get('num_ic', None)
        
        # Trajectory history
        self.observation_history: List[Observation] = []
        self.reasoning_history: List[str] = []
        self.action_history: List[Action] = []
        self.reward_history: List[float] = []
        self.plan: Optional[str] = None
        self.in_context_data: Optional[Dict] = None
        self.reflection: Optional[str] = None
        self.summary: Optional[str] = None
        
        # Environment info
        self.environment_id: Optional[str] = env.id
        self.goal: Optional[str] = env.goal if hasattr(env, 'goal') else None
        self.category: Optional[str] = env.category if hasattr(env, 'category') else None
        self.action_space: Optional[str] = env.get_action_space() if hasattr(env, 'get_action_space') else None
        # Database
        self.db = db

        # File to log to
        self.f: None

    """ Helper functions """

    def _trajectory_to_string(self, trajectory):
        """Convert a trajectory to a string"""
        # First roll up the trajectory
        trajectory = self._roll_up_trajectory([trajectory])[0]
        key_order = ["goal", "plan", "trajectory", "available_actions"]
        # Now convert to a string following key order
        trajectory_string = "\n"
        for key in key_order:
            if key in trajectory:
                trajectory_string += f"{key}: {trajectory[key]}\n"
        return trajectory_string

    def _create_conversation(self, keys: List[str], available_actions: List[Action]) -> List[Dict]:
        """Create a conversation with the observation and action history"""
        # First create a dictionary of the current trajectory. Valid keys are "plan", "observation", "reasoning", "action"
        trajectory_dict = {}
        # Sort keys according to the following order: goal, plan, observation, reasoning, action, available_actions
        key_order = ["goal", "plan", "observation", "reasoning", "action", "available_actions"]
        keys = [key for key in key_order if key in keys]
        for key in keys:
            if key == "goal":
                trajectory_dict[key] = self.goal
            elif key == "plan":
                trajectory_dict[key] = self.plan
            elif key == "observation":
                trajectory_dict[key] = [self.observation_history[i].structured for i in range(len(self.observation_history))]
            elif key == "reasoning":
                trajectory_dict[key] = [self.reasoning_history[i] for i in range(len(self.reasoning_history))]
            elif key == "action":
                trajectory_dict[key] = [self.action_history[i].text for i in range(len(self.action_history))]
            elif key == "available_actions":
                trajectory_dict[key] = repr([f"{i+1}. {a.text}" for i, a in enumerate(available_actions)])
        return self._trajectory_to_string(trajectory_dict)
    
    def _create_conversation_for_finetune(self) -> List[Dict]:
        """Create a conversation in the format used for fine-tuning
        
        Returns:
            List[Dict]: List of messages in the format used by OpenAI's fine-tuning API
        """
        messages = []
        
        # Add system message
        system_prompt = """You are a ReAct agent that helps users accomplish tasks. 
Given a goal, you will receive observations about the environment and respond with your reasoning and actions.
For each observation, first think through the problem step by step (Thought), then decide on an action (Action).
Your actions should be clear, concise, and directly executable in the environment."""
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add goal and initial observation
        if self.goal:
            initial_obs = self.observation_history[0].structured if self.observation_history else ""
            messages.append({
                "role": "user",
                "content": f"Goal: {self.goal}\nInitial observation: {initial_obs}"
            })
        
        # Process each step in the trajectory
        for i in range(len(self.reasoning_history)):
            # Add assistant's response with reasoning and action
            if i < len(self.reasoning_history) and i < len(self.action_history):
                assistant_content = f"Thought: {self.reasoning_history[i]}\nAction: {self.action_history[i].text}"
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
            
            # Add next observation if available
            if i + 1 < len(self.observation_history):
                messages.append({
                    "role": "user",
                    "content": f"Observation: {self.observation_history[i+1].structured}"
                })
        return messages
    
    def _roll_up_trajectory(self, entries):
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
                # If the category is "intercode_sql", then add the second-to-last action text to the goal
                if self.category == "sql" and "action" in entries[i] and len(entries[i]["action"]) >= 2:
                    if "SELECT" in entries[i]["action"][-2]:
                        entries[i]["goal"] += f"\nSolution query: {entries[i]['action'][-2]}"
                # Delete the interleaved keys from the dictionary
                for k in interleaved_keys:
                    del entries[i][k]
        return entries
    
    # Level of abstraction of the in-context data should come in here
    def _in_context_prompt(self, in_context_data):
        """Create a system prompt containing in-context examples from similar episodes"""
        in_context_list = []
        # If this is a dictionary, we need to unpack it
        if isinstance(in_context_data, dict):
            for key, value in in_context_data.items():
                in_context_list.append(value)
        else:
            in_context_list = [in_context_data]
        
        for in_context in in_context_list:
            # Determine which set of entries to use
            success_entries, failure_entries = in_context[0], in_context[1]
            entries = success_entries if len(success_entries) > 0 else failure_entries
            outcome = "successfully achieved" if len(success_entries) > 0 else "failed to achieve"

            if len(entries) == 0:
                return ""
            
            entries = self._roll_up_trajectory(entries)
            value_type = list(entries[0].keys()) # Keys from entries

            system_prompt = f"\nHere are some examples of {', '.join(value_type)} from episodes that {outcome} similar goals:\n"

            # Add each example
            for i, entry in enumerate(entries):
                system_prompt += f"\nExample {i+1}:\n"
                # Add keys
                for k_type in value_type:
                    system_prompt += f"{k_type}: {entry[k_type]}\n"

        if self.config.get("diversity_mode", False):
            system_prompt += f"\nFollow the provided examples closely." # However, make sure to solve the goal while taking at least one action that is different from the provided examples."
        else:
            system_prompt += f"\nFollow the provided examples closely."

        return system_prompt
    
    def _get_in_context_data(self, key_type, key, value_type, outcome="winning", k=5, window=20, filtered_environment_id=None) -> List[Dict]:
        """Retrieve in context examples from the database"""
        success_entries, failure_entries = self.db.get_similar_entries(key_type, key, outcome=outcome, k=k, window=window, filtered_environment_id=filtered_environment_id)
        if self.f:
            success_entry_ids = [entry['id'] for entry in success_entries]
            failure_entry_ids = [entry['id'] for entry in failure_entries]
            self.f.write(f"Success entry ids: {success_entry_ids}\n")
            self.f.write(f"Failure entry ids: {failure_entry_ids}\n")
        # Now figure out which part of the examples to return in-context
        if not isinstance(value_type, list): # Check that this is a list, not a string
            value_type = [value_type]
        # Filter out the keys that are not in the value_type
        success_entries = [{k: v for k, v in entry.items() if k in value_type} for entry in success_entries]
        failure_entries = [{k: v for k, v in entry.items() if k in value_type} for entry in failure_entries]
        return success_entries, failure_entries
    
    """ Main components used by agent's choose_action function """

    # A wrapper function for the get_in_context_data function when getting state-level data with a window
    def get_state_data(self, trajectory_key_types, trajectory_keys, state_key_types, state_keys, value_types, outcome, k, window, filtered_environment_id="self") -> List[Dict]:
        # Combine the trajectory and state keys for now since the wrapped function will split them back out
        if filtered_environment_id == "self":
            filtered_environment_id = self.environment_id
        key = trajectory_keys + state_keys
        key_type = trajectory_key_types + state_key_types
        """Retrieve state-level in-context examples from the database"""
        return self._get_in_context_data(key_type, key, value_types, outcome, k, window, filtered_environment_id=filtered_environment_id)
    
    # A wrapper function for the get_in_context_data function when getting trajectory-level data
    def get_trajectory_data(self, key_types, keys, value_types, outcome, k, filtered_environment_id="self") -> List[Dict]:
        """Retrieve trajectory-level in-context examples from the database"""
        if filtered_environment_id == "self":
            filtered_environment_id = self.environment_id
        return self._get_in_context_data(key_types, keys, value_types, outcome, k, filtered_environment_id=filtered_environment_id)
    
    # A wrapper function for the get_in_context_data function when getting rule-level data
    def get_rule_data(self, trajectory_key_types, trajectory_keys, state_key_types, state_keys, value_types, outcome, k, window, filtered_environment_id="self") -> List[Dict]:
        """Retrieve rule-level in-context examples from the database"""
        # Rules can apply to any state, so we need to combine the trajectory and state keys
        if filtered_environment_id == "self":
            filtered_environment_id = self.environment_id
        key = trajectory_keys + state_keys
        key_type = trajectory_key_types + state_key_types
        return self._get_in_context_data(key_type, key, value_types, outcome, k, filtered_environment_id=filtered_environment_id)

    async def create_plan(self, observation: Observation, available_actions: List[Action], in_context_data = None) -> str:
        """Generate a plan for the agent to follow"""
        conversation = []
        system_prompt = f"You are an expert at generating high-level plans of actions to achieve a goal. "
        if self.config.get("diversity_mode", False):
            system_prompt += f"\nMake sure to solve the goal while taking at least one action that is different from the provided examples. This plan must be a single line without any line breaks."
            print("Diversity mode enabled")
        if in_context_data:
            system_prompt += self._in_context_prompt(in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
        plan = await generate_plan(conversation, self.goal, self.llm)
        self.plan = plan
        return plan

    async def reason(self, observation: Observation, available_actions, in_context_data = None) -> List[Dict]:
        """Reason about the conversation and observation"""
        self.observation_history.append(observation)
        conversation = []
        # Add system prompt
        system_prompt = f"You are an expert at reasoning about the most appropriate action to take towards achieving a goal. "
        if self.config.get("give_action_space", False):
            system_prompt += "\nHere is your action space:\n" + self.action_space['description']
        if in_context_data:
            system_prompt += self._in_context_prompt(in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
        # Add conversation history
        conversation.append({"role": "user", "content": self._create_conversation(["goal", "plan", "observation", "reasoning", "action"], available_actions) + "reasoning: "})
        stop = None if self.config.get("multiline_reasoning", False) else ["\n"]
        reasoning = await self.llm.generate_chat(conversation, stop=stop)
        self.reasoning_history.append(reasoning)
        return reasoning

    async def act(self, observation: Observation, available_actions: List[Action], reasoning: Union[str, None] = None, in_context_data= None) -> Tuple[Action, List[Dict]]:
        """Select an action from available actions given the current observation"""
        # Create a conversation with observations and actions so far
        conversation = []
        # Want the system prompt to be standardized. Should have environment and goal info, as well as observation and action format. 
        system_prompt = f"""You are an agent in an environment. Given the current observation, you must select an action to take towards achieving the goal: {self.goal}."""
        # If this is a TRAD agent, we want to add the action space to the system prompt
        #print("Action space", self.action_space)
        if self.config.get("give_action_space", False):
            system_prompt += "\nHere is your action space:\n" + self.action_space['description']
        if in_context_data:
            system_prompt += self._in_context_prompt(in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": self._create_conversation(["goal", "plan", "observation", "reasoning", "action"], available_actions) + "action: "})
        
        # Select action
        stop = None if self.config.get("multiline_action", False) else ["\n"]
        action = await self.llm.generate_chat(conversation, stop=stop)
        action = Action(text=action)
        self.action_history.append(action)

        # Enforce memory size limit # VISHNU: Turn this off for now
        # self.observation_history = self.observation_history[-self.memory_size:]
        # self.action_history = self.action_history[-self.memory_size:]

        return action
    
    async def act_finetune(self, observation: Observation) -> Tuple[Action, List[Dict]]:
        """Select an action from available actions given the current observation"""
        conversation = self._create_conversation_for_finetune()
        response = await self.llm.generate_chat(conversation, stop=None)
        # Strip the "Action: " prefix
        # Also get the reasoning
        action = response.split("Action:")[1].strip()
        reasoning = response.split("Thought:")[1].split("Action:")[0].strip()
        action = Action(text=action)
        self.action_history.append(action)
        self.reasoning_history.append(reasoning)
        return action
    
    """ Main components used by agent's process_feedback function """
    
    async def reflect(self, in_context_data = None) -> List[Dict]:
        """Reflect on the conversation and observation"""
        user_prompt = self._create_conversation(["goal", "plan", "observation", "reasoning", "action"], [])
        # Identify success or failure
        reward = self.reward_history[-1]
        if reward == 1:
            success = True
        else:
            success = False
        # Create success vs failure reflection prompt
        if success: 
            #reflection_prompt = f"Your are provided with an ultimately successful trajectory of observations and actions taken. Taking into consideration the entire planning, reasoning, and execution, outline the key aspects of the trajectory that you would repeat in the future, and key mistakes you made that you would avoid in the future/any areas for improvement. Generate a response that is optimally tailored to improve the performance of agents' future attempts at this goal."
            reflection_prompt = "You are provided with an ultimately successful trajectory of observations and actions taken."
        else:
            #reflection_prompt = f"Your are provided with an ultimately failed trajectory of observations and actions taken. Taking into consideration the entire planning, reasoning, and execution, outline the key aspects of the trajectory that you would repeat in the future, and key mistakes you made that you would avoid in the future/any areas for improvement. Generate a response that is optimally tailored to improve the performance of agents' future attempts at this goal."
            reflection_prompt = "You are provided with an ultimately failed trajectory of observations and actions taken."
        #reflection_prompt += " What are up to three key facts about this task in this environment that you would like to remember for future attempts at this goal?"
        reflection_prompt += "You will have to solve this exact goal again in the future. Write down three key facts about this task in this environment that will be provided to you in the future."
        system_prompt = f"You are an agent in an environment. Given the goal: {self.goal}, {reflection_prompt}"
        if in_context_data:
            system_prompt += self._in_context_prompt(in_context_data)
        response = await self.llm.generate_chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], stop=None) # Reflection can be multiline
        self.reflection = response
        return response
    
    async def summarize(self, obs=None) -> str:
        """Summarize the conversation and observation"""
        conversation = []
        conversation = self._create_conversation(conversation, None, None, None)
        if obs is None:
            summary = await trajectory_summary(self.goal, conversation, self.llm)
        else:
            prev_obs = self.observation_history[-1] if len(self.observation_history) > 0 else Observation(None)
            summary = await observation_summary(self.goal, obs, prev_obs, self.llm)
        # Create an observation object again
        self.summary = summary
        summary = Observation(summary)
        return summary
    
    def store_episode(self):
        """Store an episode in the database"""
        # Add one more state to the observation history
        # If success, add a "Task complete" state
        if self.reward_history[-1] == 1:
            self.observation_history.append(Observation("Task complete"))
        else:
            self.observation_history.append(Observation("Task failed"))
        self.db.store_episode(self.environment_id, self.goal, self.category,self.observation_history, self.reasoning_history, self.action_history, self.reward_history, self.plan, self.reflection, self.summary)

    """ For updating rules offline """

    def _get_contrastive_pairs(self):
        """Get contrastive pairs of episodes"""
        # Fetch contrastive pairs from database and return
        contrastive_pairs = self.db.get_contrastive_pairs()
        return contrastive_pairs
    
    def _get_similar_sets(self, n, k):
        """Get similar sets of episodes"""
        # Fetch similar sets from database and return
        similar_sets = self.db.get_similar_sets(n, k)
        return similar_sets
    
    async def _generate_rule(self, data, mode, success = None):
        """Generate a rule from the given data"""
        if mode == "pair":
            sys_prompt = f"You are an evaluator that generates rules on how to successfully achieve a goal. You are given trajectories from both a successful attempt and a failed attempt at the same task. Given the key divergence in the actions taken between the two trajectories, create a rule that helps the agent successfully achieve similar goals in the future."
            user_prompt = f"Successful trajectory: {data[0]}\nFailed trajectory: {data[1]}\n Rule: "
        elif mode == "similar":
            sys_prompt = f"You are an evaluator that generates rules on how to successfully achieve a goal. You are given trajectories from agents that successfully achieved a set of similar goals. Create a rule that generalizes the shared strategies used to achieve these goals, helping the agent successfully achieve similar goals in the future. Make the rule as specific as possible to the set of trajectories provided." # Can ask for it to be general if we want...
            user_prompt = f"Successful trajectories: "
            for i, elem in enumerate(data):
                user_prompt += f"Example {i+1}: {elem}\n"
            user_prompt += "Rule: " # Can ask for this to be single line if we want...
        elif mode == "vanilla":
            success_string = "successfully achieved" if success else "failed to achieve"
            sys_prompt = f"You are an evaluator that generates rules on how to successfully achieve a goal. You are given a trajectory from an agent that {success_string} a goal, as well as a set of existing rules that may apply to this trajectory. Would you like to update an existing rule or create a new one?"
            user_prompt = f"Trajectory: {data[0]}\nExisting rules: {data[1]}\n Update or create a new rule? (update/create): "
            response = await self.llm.generate_chat([{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}])
            if response == "update":
                # Update the rule
                sys_prompt = f"You are an evaluator that updates rules on how to successfully achieve a goal. You are given a trajectory from an agent that {success_string} a goal, as well as a set of existing rules that may apply to this trajectory. You must update the rule that is most relevant to the trajectory, refining the rule by correcting any errors or adding any missing information."
                user_prompt = f"Trajectory: {data[0]}\nExisting rules: {data[1]}\n Update the rule that best matches the trajectory (rule_id: rule_content): "
            elif response == "create":
                # Create a new rule
                sys_prompt = f"You are an evaluator that generates rules on how to successfully achieve a goal. You are given a trajectory from an agent that {success_string} a goal, as well as a set of existing rules that may apply to this trajectory. Create a new rule that helps the agent successfully achieve similar goals in the future."
                user_prompt = f"Trajectory: {data[0]}\nExisting rules: {data[1]}\n Create a new rule that helps the agent successfully achieve similar goals in the future. Rule: "
        response = await self.llm.generate_chat([{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}], stop=None)
        print(response)
        return response
    
    async def generate_rules(self, mode, environment_id):
        if mode == "pair":
            contrastive_pairs = self._get_contrastive_pairs()
            for pair in contrastive_pairs:
                pair[0] = self._trajectory_to_string(pair[0])
                pair[1] = self._trajectory_to_string(pair[1])
                self._generate_rule(pair, mode)
        elif mode == "similar":
            contrastive_pairs = self._get_contrastive_pairs()
            similar_sets = self._get_similar_sets(n = 5, k = 3)
            print("Similar sets", similar_sets)
            for set in similar_sets:
                for i in range(len(set)):
                    set[i] = self._trajectory_to_string(set[i])
                await self._generate_rule(set, mode)
        elif mode == "vanilla":
            # What rules apply to this trajectory? Do match on goal/category/observation/reasoning? 
            # Let's assume we can return rules in the "value_types"
            # Check if success or failure by seeing if 1 in reward_history
            success = 1 in self.reward_history
            current_trajectory = self._create_conversation(keys=["goal", "plan", "observation", "reasoning", "action"], available_actions=[])
            relevant_rules = self.get_trajectory_data(key_types=["goal", "category"], keys=[self.goal, self.category], value_types=["rule_content"], k=3)
            self._generate_rule([current_trajectory, relevant_rules], mode, success) # Either an update or a new rule

    async def consolidate_rules(self):
        """Consolidate rules"""
        pass

    """ Placeholder functions for agent's choose_action and process_feedback functions """
    
    async def choose_action(self, obs, valid_actions):
        """Choose an action from available actions given the current observation"""
        pass
    
    async def analyze_episode(self):
        """Process feedback from the environment"""
        pass

    """ Rule generation functions """

    async def batch_analyze_episodes(self):
        """Update rules offline"""
        pass

    """ Between-episode functions for the outer loop """
        
    def clear_history(self):
        """Clear the agent's history"""
        self.observation_history = []
        self.action_history = []
        self.reasoning_history = []
        self.reward_history = []
        self.plan = None
        self.in_context_data = None
        self.reflection = None
        self.summary = None

    def reset(self):
        """Reset agent state between episodes"""
        self.observation_history = []
        self.action_history = []
        self.reasoning_history = []
        self.reward_history = []
        self.plan = None
        self.reflection = None
        self.summary = None
        self.in_context_data = None

    def clean_history(self):
        """Clean the agent's history"""
        nothing_indices = [i for i in range(len(self.observation_history)) if "Nothing happens" in self.observation_history[i].structured]
        
        for idx in sorted(nothing_indices, reverse=True):
            del self.observation_history[idx]
            if idx > 0:
                if idx-1 < len(self.reasoning_history):
                    del self.reasoning_history[idx-1]
                if idx-1 < len(self.action_history):
                    del self.action_history[idx-1]
                if idx-1 < len(self.reward_history):
                    del self.reward_history[idx-1]