PROMPT1 = """Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]
"""

def get_webshop_fewshots():
    # Parse the prompt into actions, observations and thoughts
    actions = []
    observations = []
    thoughts = []
    
    lines = PROMPT1.split('\n')
    
    current_observation = []
    last_action_was_think = False
    
    for line in lines:
        if line.startswith('Action: '):
            # Store previous observation if exists
            if current_observation:
                obs_text = '\n'.join(current_observation).strip()
                if obs_text != 'OK.':
                    observations.append(obs_text)
                current_observation = []
            
            # Parse action
            action_text = line[len('Action: '):]
            if action_text.startswith('think['):
                thoughts.append(action_text[len('think['):-1])
                last_action_was_think = True
            else:
                # Add empty thought if previous action wasn't a think
                if not last_action_was_think:
                    thoughts.append("")
                actions.append(action_text)
                last_action_was_think = False
                
        elif line.startswith('Observation: '):
            current_observation = [line[len('Observation: '):]]
        elif current_observation is not None:
            current_observation.append(line)

    # Goal is in first obs
    goal = observations[0].split('Instruction: ')[1].split('[')[0].strip()
            
    return [(goal, observations, thoughts, actions)]

WEBSHOP_FEWSHOTS = get_webshop_fewshots()

'''
goal, observations, thoughts, actions = get_webshop_fewshots()
print(len(actions))
print(len(observations))
print(len(thoughts))
# Print the 3 interleaved
for i in range(len(observations)):
    print(f"Observation: {observations[i]}")
    print(f"Thought: {thoughts[i]}")
    print(f"Action: {actions[i]}")
    print("\n")
print(goal)
'''