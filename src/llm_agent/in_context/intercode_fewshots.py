from llm_agent.in_context.intercode_fewshot_file import DEMO_MAP_REACT, DEMO_MAP_PLAN

def parse_fewshot(react_demo, plan_demo):
    # Return goal, plan, obs_list, thought_list, act_list
    # We may need to assume that the same index in demo_map_react and demo_map_plan correspond to the same fewshot
    lines = react_demo.split('\n')
    
    # Extract goal from first line
    goal = lines[0].replace('Question: ', '')
    
    obs_list = [goal]
    thought_list = []
    act_list = []

    # Extract plan from plan demo
    plan = ", ".join(plan_demo.split('Plan:')[1:]).strip()
    
    current_thought = ""
    
    for line in lines[1:]:
        line = line.strip()
        if line.startswith('Thought'):
            # Extract thought number and content
            thought_num = line.split(':')[0].replace('Thought ', '')
            thought_content = ':'.join(line.split(':')[1:]).strip()
            current_thought = thought_content
            
        elif line.startswith('Action'):
            # Add the thought and action
            thought_list.append(current_thought)
            act = line.split(':')[1].strip()
            act_list.append(act)
            current_thought = ""
            
        elif line.startswith('Observation'):
            # Add observation
            obs = line.split(':')[1].strip()
            obs_list.append(obs)
            
    return goal, plan, obs_list, thought_list, act_list

def parse_all_fewshots(language):
    # Split the entries in demo_map_react and demo_map_plan into lists of fewshots
    # Split by the phrase "Question: "
    react_examples = DEMO_MAP_REACT[language]
    plan_examples = DEMO_MAP_PLAN[language]
    # Convert strings to lists
    react_examples = react_examples.split("Question: ")[1:]
    plan_examples = plan_examples.split("Question: ")[1:]
    # Now loop through each fewshot and parse it
    return_list = []
    for i in range(len(react_examples)):
        goal, plan, obs_list, thought_list, act_list = parse_fewshot(react_examples[i], "")
        return_list.append((goal, plan, obs_list, thought_list, act_list))
    return return_list
