import asyncio

import yaml
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from llm_agent.agent.base_agent_v2 import BaseAgent
from llm_agent.agent.agents.react import ReAct
from llm_agent.agent.agents.reflexion import Reflexion
from llm_agent.agent.agents.rap import RAP
from llm_agent.agent.agents.rap_noplan import RAPNoPlan
from llm_agent.agent.agents.synapse import Synapse
from llm_agent.agent.agents.expel_train import ExpelTrain
from llm_agent.agent.agents.trad import TRAD
from llm_agent.env.base_env import Observation, Action
from llm_agent.llm.lite_llm import LiteLLMWrapper
from llm_agent.env.alfworld_env import AlfWorldEnv
from llm_agent.env.webshop_site_env import WebShopEnv
from llm_agent.env.gym_env import GymEnv
from llm_agent.database.learning_db import LearningDB
import random
import argparse
import json

def config():
    # Load default config first
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with alfworld-specific config
    with open('config/benchmark/alfworld.yaml', 'r') as f:
        env_config = yaml.safe_load(f)
        
    # Ensure required alfworld configuration exists
    env_config.update({
        'type': 'AlfredTWEnv',  # Required env type for alfworld
        'split': 'eval_out_of_distribution'  # Required split parameter # eval_out_of_distribution
    })
    '''
    env_config.update({
        'type': 'WebShopEnv',  # Required env type for alfworld
        'name': 'webshop',
        'max_steps': 10,
    })
    '''
    
    # Update with benchmark config
    config['benchmark'] = env_config
    
    return config

use_gym = False

def env(config):
    if use_gym:
        env_config = {"env_name": "CartPole-v1"}
        return GymEnv(env_config)
    elif config['benchmark']['name'] == 'webshop':
        return WebShopEnv(config['benchmark'])
    else:
        return AlfWorldEnv(config['benchmark'])

def real_llm(config):
    return LiteLLMWrapper(config)

def test_config(agent_type):
    return {
        "max_retries": 1,
        "memory_size": 50,
        "temperature": 0.7,
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
    elif test_config.get('agent_type', 'react') == 'expel_train':
        return ExpelTrain(real_llm, db, env, test_config)
    elif test_config.get('agent_type', 'react') == 'trad':
        return TRAD(real_llm, db, env, test_config)
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
                selected_action = await agent.choose_action(obs, valid_actions, log_file)
                f.write(f"Selected action: {selected_action}\n")
                # Take step in env
                obs, reward, done, info = env.step(selected_action.text) # This feedback needs to be looped anyways
                f.write(f"Obs: {obs}, Reward: {reward}\n")
                obs = Observation(structured=obs)
                # Pass feedback to agent
                await agent.process_feedback(obs, reward, done, log_file)
                # Increment step count
                steps += 1
                f.write(f"Step {steps} of {env.max_steps}\n")
                # Flush the file to ensure the log is written
                f.flush()
            
            if not done:
                f.write("\nEpisode timed out after reaching max steps\n")
                f.flush()

            if reward < 1:
                attempt_count += 1 
                agent.clear_history()
            else:
                agent.clear_history()
                break
        await agent.update_rules_online(env.id)

# Run the environment
async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', required=True, help='LLM model to use')
    parser.add_argument('--db_path', help='Optional custom path for learning database')
    parser.add_argument('--db_name', help='Optional custom name for learning database')
    parser.add_argument('--agent_type', required=True, help='Type of agent to use')
    parser.add_argument('--num_passes', type=int, default=1, help='Number of passes to run')
    parser.add_argument('--num_attempts', type=int, default=1)
    parser.add_argument('--run_offline_rules', action='store_true', help='Run offline rules')
    args = parser.parse_args()

    for j in range(args.num_passes):
        for i in range(27,134,1):
            cfg = config()
            cfg['benchmark']['problem_id'] = i
            cfg['llm']['model'] = args.llm

            agent_config = test_config(agent_type=args.agent_type)
            db_name = args.db_name if args.db_name else "default"
            log_dir = Path("logs/episodes") / f"{cfg['benchmark']['name']}/{cfg['benchmark']['split']}/{args.agent_type}/{args.llm}/{db_name}"
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / f"{i}.txt"
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

if __name__ == "__main__":
    asyncio.run(main())