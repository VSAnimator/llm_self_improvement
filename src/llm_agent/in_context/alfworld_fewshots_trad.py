from llm_agent.in_context.alfworld_3prompts import react_prompt_inference

#print(react_prompt_inference)

def get_task_type(task_desc):
    """Map task description to task type based on key verbs.
    
    Args:
        task_desc: String containing task description
        
    Returns:
        task_type: String indicating task type (pick_and_place, pick_clean_then_place, etc)
    """
    task_desc = task_desc.lower()
    
    if "clean" in task_desc:
        return "pick_clean_then_place"
    elif "heat" in task_desc or "hot" in task_desc:
        return "pick_heat_then_place"
    elif "cool" in task_desc:
        return "pick_cool_then_place" 
    elif "look at" in task_desc or "examine" in task_desc:
        return "look_at_obj"
    elif "two" in task_desc or any(w in task_desc for w in ["put two", "put 2"]):
        return "pick_two_obj"
    elif "put" in task_desc:
        return "pick_and_place"
    else:
        raise ValueError(f"Could not determine task type from description: {task_desc}")

def parse_fewshot_trad(fewshot):
    """Parse a fewshot example into observations and actions"""
    lines = fewshot.split('\n')
    
    current_obs = ""
    current_act = ""
    current_thought = ""
    plan = ""

    goal = lines[1]
    del lines[1]
    obs_list = []
    thought_list = []
    act_list = []

    phase = "act"
    
    for line in lines:
        if line.startswith('> '):
            # Parse the action
            act = line[2:] # Remove '> ' prefix
            if act.startswith('think:'):
                phase = "think"
                # Add the thought
                current_thought = act[7:].strip()
            else:
                phase = "act"
                current_act = act[5:].strip()
        else:
            if phase == "think":
                #if "To solve the task" in line and plan == "":
                #    plan = line.split(".")[0].strip()
                #    line = line[len(plan):].strip()
                current_thought += (". " if current_thought[-1] != ":" else " ") + line.strip()
            else:
                if len(current_obs) > 0:
                    # New obs--meaning add the previous obs, thought, and act to the lists
                    obs_list.append(current_obs)
                    thought_list.append(current_thought)
                    act_list.append(current_act)
                current_obs = line.strip()
                current_thought = ""
                current_act = ""

    del act_list[-1]
    del thought_list[-1]
            
    # Add final observation if exists
    if len(current_obs) > 0:
        obs_list.append(current_obs)

    # Get task type
    task_type = get_task_type(goal)
        
    return goal, task_type, obs_list, thought_list, act_list

def parse_alfworld_fewshots():
    all_parsed = []
    for elem in react_prompt_inference.values():
        parsed = parse_fewshot_trad(elem)
        all_parsed.append(parsed)
    return all_parsed

FEWSHOTS_TRAD = parse_alfworld_fewshots()
#print(FEWSHOTS_TRAD)