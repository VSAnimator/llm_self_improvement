import json

with open('/Users/sarukkai/Documents/agent_algo_bench/data/miniwob-plusplus-demos/clean-demos/click-checkboxes-large/click-checkboxes-large_1026090635.json', 'r') as f:
    data = json.load(f)

def parse_miniwob_fewshots(data):
    goal = data['taskName'] + ": " + data['utterance']
    obs_list = []
    act_list = []
    for state in data['states']:
        obs_list.append(state['dom'])
        if state['action'] and len(state['action']) > 0:
            act_list.append(state['action'])

    return goal, obs_list, act_list

goal, obs_list, act_list = parse_miniwob_fewshots(data)
print(goal)
print(obs_list)
print(act_list)
print(len(obs_list))
print(len(act_list))