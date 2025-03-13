import os
from enum import IntEnum

import numpy as np
import gym
from gym.utils import seeding

from llm_agent.env.envs.wordcraft_recipebook import Recipe, RecipeBook

from llm_agent.env.base_env import BaseEnv, Observation, Action
from typing import Tuple, Dict, Optional, List

NO_RECIPE_PENALTY = 0.0
IRRELEVANT_RECIPE_PENALTY = 0.0
GOAL_REWARD = 1.0
SUBGOAL_REWARD = 0.0

class WordCraftBaseEnv(gym.Env):
    """
    Simple text-only RL environment for crafting multi-step recipes.

    At a high level, the state consists of a goal, the inventory, and the current selection.
    """
    def __init__(
        self,
        data_path='src/llm_agent/env/envs/alchemy2.json',
        recipe_book_path=None,
        max_depth=1,
        split='by_task',
        train_ratio=0.9,
        num_distractors=0,
        uniform_distractors=False,
        max_mix_steps=1,
        subgoal_rewards=True,
        seed=None,
    ):
        super().__init__()

        self.eval_mode = False

        if seed is None:
            seed = int.from_bytes(os.urandom(4), byteorder="little")

        if recipe_book_path is not None:
            self.recipe_book = RecipeBook.load(recipe_book_path)
            self.recipe_book.set_seed(seed)
            max_depth = self.recipe_book.max_depth
        else:
            self.recipe_book = RecipeBook(
                data_path=data_path, max_depth=max_depth, split=split, train_ratio=train_ratio, seed=seed)

        self.set_seed(seed)

        self.max_selection_size = self.recipe_book.max_recipe_size
        self.max_mix_steps = max(max_mix_steps or max_depth, max_depth)
        self.max_steps = self.max_selection_size*self.max_mix_steps

        self.sample_depth = max_depth

        self.subgoal_rewards = subgoal_rewards
        self.max_depth = max_depth
        self.num_distractors = num_distractors
        self.uniform_distractors = uniform_distractors

        self.max_table_size = 2**max_depth + num_distractors + self.max_mix_steps

        self.task = None
        self.distractors = []

        self._reset_table()
        self._reset_selection()
        self._reset_history()

        self.episode_step = 0
        self.episode_mix_steps = 0
        self.episode_reward = 0
        self.done = False

        obs = self.reset()
        num_entities = len(self.recipe_book.entities)
        dspaces = {
            'goal_index': gym.spaces.MultiDiscrete([num_entities]),
            'table_index': gym.spaces.MultiDiscrete(self.max_table_size*[num_entities]),
            'selection_index': gym.spaces.MultiDiscrete(self.max_selection_size*[num_entities]),
        }
        self.observation_space = gym.spaces.Dict(dspaces)
        self.action_space = gym.spaces.Discrete(self.max_table_size) # Actions correspond to choosing an entity in a table position

    def reset(self):
        self.episode_step = 0
        self.episode_mix_steps = 0
        self.episode_reward = 0
        self.done = False

        self.task = self.recipe_book.sample_task(depth=self.sample_depth)
        print("Sampled task", self.task)
        self.distractors = self.recipe_book.sample_distractors(self.task, self.num_distractors, uniform=self.uniform_distractors)
        self._reset_selection()
        self._reset_table()
        self._reset_history()

        return self._get_observation()

    def eval(self, split='test'):
        self.eval_mode = True
        self.recipe_book.test_mode = (split == 'test')

    def train(self):
        self.eval_mode = False
        self.recipe_book.test_mode = False

    def set_seed(self, seed):
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        self.recipe_book.set_seed(seed)

    def sample_depth(self, depth):
        self.sample_depth = depth

    def __max_table_size_for_depth(self, depth):
        return 2**depth - 1

    def _reset_table(self):
        if self.task:
            self.table = list(self.task.base_entities + self.distractors)
            self.np_random.shuffle(self.table)
        else:
            self.table = []
        self.table_index = -np.ones(self.max_table_size, dtype=int)

        num_start_items = len(self.table)
        self.table_index[:num_start_items] = \
            np.array([self.recipe_book.entity2index[e] for e in self.table], dtype=int)

    def _reset_selection(self):
        self.selection = []
        self.selection_index = -np.ones(self.max_selection_size, dtype=int)
        
    def _reset_history(self):
        self.subgoal_history = set()

    def _get_observation(self):
        """
        Note, includes indices for each inventory and selection item,
        since torchbeast stores actions in a shared_memory tensor shared among actor processes
        """
        return {
            'goal_index': [self.recipe_book.entity2index[self.task.goal]],
            'table_index': self.table_index,
            'selection_index': self.selection_index,
        }

    def step(self, action):
        reward = 0
        if self.done: # no-op if env is done
            return self._get_observation(), reward, self.done, {}

        # Handle invalid actions
        invalid_action = not (0 <= action < self.max_table_size)
        if invalid_action:
            self.episode_step += 1
            if self.episode_step >= self.max_steps:
                self.done = True

        i = self.table_index[action]
        e = self.recipe_book.entities[i]

        selection_size = len(self.selection)
        if selection_size < self.max_selection_size:
            # Update selection
            self.selection.append(e)
            self.selection_index[selection_size] = i
            selection_size = len(self.selection)

        if selection_size == self.max_selection_size:
            self.episode_mix_steps += 1

            # Evaluate selection
            recipe = Recipe(self.selection)
            result = self.recipe_book.evaluate_recipe(recipe)

            if result is None:
                reward = NO_RECIPE_PENALTY if not self.eval_mode else 0

            elif result == self.task.goal:
                reward = GOAL_REWARD
                self.done = True
            elif result in self.task.intermediate_entities:
                reward = 0
                if result not in self.subgoal_history:
                    self.subgoal_history.add(result)
                    reward = SUBGOAL_REWARD if self.subgoal_rewards and not self.eval_mode else 0
            else:
                reward = IRRELEVANT_RECIPE_PENALTY if not self.eval_mode else 0
            self.episode_reward += reward

            if result:
                result_i = self.recipe_book.entity2index[result]
                table_size = len(self.table)
                self.table.append(result)
                self.table_index[table_size] = result_i

            # Clear selection
            self._reset_selection()

        self.episode_step += 1
        if self.episode_mix_steps >= self.max_mix_steps or self.episode_step >= self.max_steps:
            self.done = True

        obs = self._get_observation()

        return obs, reward, self.done, {}

    def _display_ascii(self, mode='human'):
        """
        Render the env state as ascii:

        Combine the ingredients to make *torch*

        -------------------------------------------------------
        1:fire, 2:wind, 3:sand, 4:star, 5:wood, 6:stick, 7:coal
        -------------------------------------------------------

        (on hand): stick

        Subgoal rewards: 0
        """
        goal_str = f'Combine the ingredients to make *{self.task.goal}*'
        if mode == 'human':
            table_str = f"{', '.join([f'{i+1}:{e}' for i, e in enumerate(self.table)])}"
        else:
            table_str = f"{', '.join(self.table)}"
        selection_str = f"(on hand): {', '.join(self.selection)}"
        hr = ''.join(['-']*50)

        # output = f'\n{goal_str}\n\n{hr}\n{table_str}\n{hr}\n\n{selection_str}\n\nSubgoal rewards: {self.episode_reward}\n'
        output = f'\n{goal_str}\n\n{hr}\n{table_str}\n{hr}\n\n{selection_str}\n\n'

        return output

    def render(self, mode='human'):
        return self._display_ascii(mode)

gym.envs.registration.register(
    id='wordcraft-multistep-goal-v0',
    entry_point=f"{__name__}:WordCraftEnv",
)

'''
if __name__ == "__main__":
    env = WordCraftEnv(max_depth=2, num_distractors=2)
    env.reset()
    for i in range(10):
        env.render()
        action = int(input("Enter an action: "))
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        env.render()
        if done:
            env.reset()
'''

class WordCraftEnv(BaseEnv):
    def __init__(self, config: Dict):
        """Initialize WordCraft environment
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.max_steps = config.get('max_steps', 100)
        self._observation = None
        self.steps = 0
        self.id = config.get('problem_id', 0)
        self.category = "wordcraft"
        self.env = WordCraftBaseEnv(max_depth=2, num_distractors=0, seed=self.id, max_mix_steps=4)
        # If we are in test, need to set the test mode
        if self.config.get('split', 'train') == 'test':
            self.env.eval('test')
        
    def clean_obs(self, obs: str) -> str:
        """Clean the observation
        
        Args:
            obs: Observation to clean
        """
        # Remove "(on hand)"
        obs = obs.strip()
        obs = obs.replace("(on hand):", "")
        # Remove the line with "Combine the ingredients to make *"
        obs = obs.split("\n")[2:]
        obs = " ".join(obs)
        # Remove trailing spaces and newlines
        obs = obs.strip()
        # Remove all the dashes
        obs = obs.replace("--", "")
        # Add a bit saying "ingredients available"
        obs = "Ingredients available: " + obs
        return obs
        
    def reset(self) -> Tuple[Observation, Dict]:
        """Reset environment to initial state
        
        Returns:
            Tuple[Observation, Dict]: Initial observation and info
        """
        x = self.env.reset()
        obs = self.env.render()
        obs = obs.strip()
        self.goal = obs.split('\n')[0]
        self.goal += ". You may only combine two entities at a time. You can take only two steps to make the final product."
        obs = self.clean_obs(obs)
        return obs, {}
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, Dict]:
        """
        Take a step in the environment
        """
        
        action_list = []
        valid_options = [f'{e}' for e in self.env.table]
        for option in valid_options:
            if option in action:
                # Get the index of the option
                action_list.append(valid_options.index(option))
                # If it appears twice, we need to add it twice
                if action.count(option) == 2:
                    action_list.append(valid_options.index(option))
        if len(action_list) != 2:
            return None, 0, False, False, {}
        toreturn = None
        for action in action_list:
            toreturn = self.env.step(action)
        obs = self.env.render()
        obs = self.clean_obs(obs)
        return obs, toreturn[1], toreturn[2], toreturn[3]
    
    def get_action_space(self) -> Dict:
        """Get JSON schema describing valid action format
        
        Returns:
            JSON schema for text commands
        """
        # Output strings with the names of the two entities we would like to combine
        return {
            "type": "string",
            "description": "Output strings with the names of the two entities we would like to combine in this step."
        }
        
# Let's test the environment
'''
if __name__ == "__main__":
    env = WordCraftEnv({'max_steps': 100, 'problem_id': 0})
    #env.reset()
    #env.env.set_seed(0)
    #env.reset()
    for i in range(10):
        env.env.set_seed(i)
        env.reset()
        env.env.render()
        continue
        action = input("Enter an action: ")
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        env.env.render()
        if done:
            env.reset()
'''