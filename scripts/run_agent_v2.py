import asyncio

import yaml
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from llm_agent.agent.base_agent_v2 import BaseAgent
from llm_agent.agent.agents import (
    ReAct,
    Reflexion,
    RAP,
    RAPNoPlan,
    Synapse,
    Expel,
    AutoGuide,
    AutoManual,
    TRAD
)
from llm_agent.env.base_env import Observation, Action
from llm_agent.llm.lite_llm import LiteLLMWrapper
from llm_agent.database.learning_db import LearningDB
import random
import argparse
import concurrent.futures
from tqdm import tqdm
import multiprocessing
import time
import threading

def config(env, gym_env_name):
    # Load default config first
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with alfworld-specific config
    with open('config/benchmark/alfworld.yaml', 'r') as f:
        env_config = yaml.safe_load(f)
        
    # Ensure required alfworld configuration exists
    if env == 'alfworld_test':
        env_config.update({
            'type': 'AlfredTWEnv',  # Required env type for alfworld
            'split': 'eval_out_of_distribution',#'eval_out_of_distribution'  # Required split parameter # eval_out_of_distribution
            'name': 'alfworld_test'
        })
    elif env == 'alfworld':
        env_config.update({
            'type': 'AlfredTWEnv',  # Required env type for alfworld
            'split': 'eval_in_distribution',#'eval_out_of_distribution'  # Required split parameter # eval_out_of_distribution
            'name': 'alfworld'
        })
    elif env == 'webshop':
        env_config.update({
            'type': 'WebShopEnv',  # Required env type for alfworld
            'name': 'webshop',
            'max_steps': 10,
        })
    elif env == 'gymnasium':
        env_config.update({
            'type': 'GymEnv',
            'name': gym_env_name,
            'max_steps': 10,
        })
    else:
        raise ValueError(f"Invalid environment name: {env}")
    
    # Update with benchmark config
    config['benchmark'] = env_config
    
    return config

use_gym = False

def env(config):
    if use_gym:
        from llm_agent.env.envs.gym_env import GymEnv
        env_config = {"env_name": "CartPole-v1"}
        return GymEnv(env_config)
    elif config['benchmark']['name'] == 'webshop':
        from llm_agent.env.envs.webshop_site_env import WebShopEnv
        return WebShopEnv(config['benchmark'])
    elif config['benchmark']['name'] == 'alfworld':
        from llm_agent.env.envs.alfworld_train_env import AlfWorldTrainEnv
        return AlfWorldTrainEnv(config['benchmark'])
    elif config['benchmark']['name'] == 'alfworld_test':
        from llm_agent.env.envs.alfworld_env import AlfWorldEnv
        return AlfWorldEnv(config['benchmark'])
    else:
        raise ValueError(f"Invalid environment name: {config['benchmark']['name']}")

def real_llm(config):
    return LiteLLMWrapper(config)

def test_config(agent_type):
    return {
        "max_retries": 1,
        "memory_size": 50,
        "temperature": 0.1,
        "agent_type": agent_type
    }

def test_agent(real_llm, db, env, test_config):
    if test_config.get('agent_type', 'react') == 'react':
        return ReAct(real_llm, db, env, test_config)
    elif test_config.get('agent_type', 'react') == 'reflexion':
        return Reflexion(real_llm, db, env, test_config)
    elif test_config.get('agent_type', 'react') == 'rap':
        return RAP(real_llm, db, env, test_config)
    elif test_config.get('agent_type', 'react') == 'rap_noplan':
        return RAPNoPlan(real_llm, db, env, test_config)
    elif test_config.get('agent_type', 'react') == 'synapse':
        return Synapse(real_llm, db, env, test_config)
    elif test_config.get('agent_type', 'react') == 'expel':
        return Expel(real_llm, db, env, test_config)
    elif test_config.get('agent_type', 'react') == 'trad':
        test_config['give_action_space'] = True
        return TRAD(real_llm, db, env, test_config)
    elif test_config.get('agent_type', 'react') == 'autoguide':
        return AutoGuide(real_llm, db, env, test_config)
    elif test_config.get('agent_type', 'react') == 'automanual':
        return AutoManual(real_llm, db, env, test_config)
    else:
        raise ValueError(f"Invalid agent type: {test_config.get('agent_type', 'react')}")

def db(db_path):
    return LearningDB(db_path)

async def run_env(agent, env, log_file, num_attempts):
    # Goal: run the agent on the environment and log the results
    attempt_count = 0
    print("Num attempts", num_attempts)
    with open(log_file, "a" if os.path.exists(log_file) else "w") as f:
        for attempt in range(num_attempts):
            # Initial reset
            obs, info = env.reset()
            # Set agent goal
            agent.goal = env.goal
            agent.category = env.category
            agent.environment_id = env.id
            # Continue to logging
            f.write(f"Initial observation: {obs}\n")
            obs = Observation(structured=obs)
            done = False
            steps = 0

            while not done and steps < env.max_steps:
                # Get valid actions
                # Check if env has get_available_actions
                if hasattr(env, 'get_available_actions'):
                    valid_actions = env.get_available_actions(info)
                    valid_actions = [Action(text=action) for action in valid_actions]
                    f.write(f"Valid actions: {valid_actions}\n")
                else:
                    valid_actions = None
                # Choose action
                selected_action = await agent.choose_action(obs, valid_actions)
                f.write(f"Reasoning: {agent.reasoning_history[-1]}\n")
                f.write(f"Selected action: {selected_action}\n")
                # Take step in env
                obs, reward, done, info = env.step(selected_action.text) # This feedback needs to be looped anyways
                f.write(f"Obs: {obs}, Reward: {reward}\n")
                obs = Observation(structured=obs)
                # Pass feedback to agent
                agent.reward_history.append(reward) # Add to reward history
                # Increment step count
                steps += 1
                f.write(f"Step {steps} of {env.max_steps}\n")
                # Flush the file to ensure the log is written
                f.flush()
            
            if not done:
                f.write("\nEpisode timed out after reaching max steps\n")
                f.flush()

            # Analyze episode
            await agent.analyze_episode()

            # Store episode if store_episodes is True
            if agent.config.get("store_episodes", False):
                agent.store_episode()

            if reward < 1:
                attempt_count += 1 
                agent.clear_history()
            else:
                agent.clear_history()
                break

# Run the environment
async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', required=True, help='LLM model to use')
    parser.add_argument('--backend', default='litellm', help='Backend to use for LLM')
    parser.add_argument('--db_path', help='Optional custom path for learning database')
    parser.add_argument('--db_name', help='Optional custom name for learning database')
    parser.add_argument('--agent_type', required=True, help='Type of agent to use')
    parser.add_argument('--num_passes', type=int, default=1, help='Number of passes to run')
    parser.add_argument('--num_attempts', type=int, default=1)
    parser.add_argument('--run_offline_rules', action='store_true', help='Run offline rules')
    parser.add_argument('--store_episodes', action='store_true', help='Store episodes')
    parser.add_argument('--env', default='alfworld_test', help='Environment to use (alfworld, alfworld_test, webshop, gymnasium)')
    parser.add_argument('--gym_env_name', help='Name of gymnasium environment if using gymnasium')
    parser.add_argument('--num_tasks', type=int, default=134, help='Number of tasks to run')
    parser.add_argument('--parallel', type=int, default=1, help='Number of threads to use')
    args = parser.parse_args()

    async def process_task(i, args, environment=None):
        cfg = config(args.env, args.gym_env_name)
        cfg["benchmark"]["problem_id"] = i
        cfg["llm"]["model"] = args.llm
        cfg["llm"]["backend"] = args.backend

        agent_config = test_config(agent_type=args.agent_type)
        agent_config["store_episodes"] = args.store_episodes
        db_name = args.db_name if args.db_name else "default"
        log_dir = (
            Path("logs/episodes")
            / f"{cfg['benchmark']['name']}/{cfg['benchmark']['split']}/{args.agent_type}/{args.llm}/{db_name}"
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{i}.txt"
        if environment is None or cfg["benchmark"]["name"] != "alfworld":
            environment = env(cfg)
        llm = real_llm(cfg)
        default_db_path = f"{log_dir}/learning.db"
        db_path = args.db_path if args.db_path else default_db_path
        learning_db = db(db_path=db_path)
        agent = test_agent(llm, learning_db, environment, agent_config)
        if args.run_offline_rules:
            # Only need to run offline rules once
            await agent.update_rules_offline()
            return
        await run_env(agent, environment, log_file, args.num_attempts)

    def worker(task_queue, args):
        """Worker function that fetches tasks from the queue and runs them sequentially."""
        while not task_queue.empty():
            try:
                i = task_queue.get_nowait()  # Non-blocking get to prevent hanging
            except multiprocessing.queues.Empty:
                break
            try:
                asyncio.run(process_task(i, args))  # Run the async function
            except Exception as e:
                print(f"Error processing task {i}: {e}")

    def monitor_progress(task_queue, total_tasks):
        """Thread function to update tqdm every second."""
        with tqdm(total=total_tasks, desc="Processing tasks", unit="task") as pbar:
            while not task_queue.empty():
                remaining = task_queue.qsize()
                pbar.n = total_tasks - remaining  # Update the progress bar
                pbar.refresh()
                time.sleep(1)  # Update every second

    for _ in range(args.num_passes):
        with multiprocessing.Manager() as manager:
            task_queue = manager.Queue()
            for i in range(args.num_tasks):
                task_queue.put(i)

            processes = []                
            for _ in range(args.parallel):
                p = multiprocessing.Process(target=worker, args=(task_queue, args))
                processes.append(p)
                p.start()

            # Start progress monitoring thread
            progress_thread = threading.Thread(target=monitor_progress, args=(task_queue, args.num_tasks), daemon=True)
            progress_thread.start()

            for p in processes:
                p.join()  # Ensure all processes complete execution
            progress_thread.join()  # Ensure progress thread terminates


if __name__ == "__main__":
    asyncio.run(main())