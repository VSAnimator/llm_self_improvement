import time
from llm_agent.env.envs.miniwob_env import MiniWoBEnv
import ast
import json
from miniwob.action import ActionTypes

# Create environment
env = MiniWoBEnv("click-test-2-v1")

try:
    # Reset environment
    obs, info = env.reset()
    print("Obs:", obs)
    
    # Get action space
    action_space = env.get_action_space(obs)
    
    # Find element with text "ONE" 
    target_ref = None
    for ref in action_space:
        #print("Obs:", obs)
        for elem in obs["dom_elements"]:
            if str(elem["ref"]) == ref and elem["text"] == "ONE":
                target_ref = ref
                break
                
    if target_ref is None:
        print("Could not find target element")
        exit(1)

    print("Target ref:", target_ref)
        
    # Take action
    obs, reward, done, info = env.step(env.env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=target_ref))
    
    # Print results
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    time.sleep(2)  # Allow time to view result

finally:
    env.close()
