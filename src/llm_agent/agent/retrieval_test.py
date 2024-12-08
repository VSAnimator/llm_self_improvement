from llm_agent.agent.base_agent_v2 import BaseAgent
from ..env.base_env import Observation, Action

class RetrievalTest(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)

    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        # Create observation from obs string
        obs = Observation(structured=obs)
        # Create action objects from valid action strings
        valid_actions = [Action(text=action) for action in valid_actions]
        if self.config.get('use_summarization', False):
            obs = await self.summarize(obs) # Create_conversation can pull in the trajectory
        self.plan = "To achieve the goal of examining the bowl under the desklamp, follow these steps based on the provided trajectory: 1. **Identify and locate desks**: Start by identifying the desks in the room using the initial observation. Determine if the desks have any objects on them. 2. **Move to a desk**: Navigate to a desk that has a desklamp or a potential place where a bowl might be. 3. **Observe the desk**: Look at the desk to take note of any objects present, especially the desklamp and bowl. 4. **Retrieve the bowl (if not already placed on the target desk)**: If the bowl is not under the desklamp, locate the bowl elsewhere (e.g., on another desk) and pick it up. 5. **Place the bowl on the desk with desklamp**: If the bowl is not already under the desklamp, place it on the desk with the desklamp after picking it up. 6. **Turn on the desklamp**: Ensure the desklamp is turned on to correctly observe the bowl under proper lighting. 7. **Examine the bowl**: Once the bowl is correctly placed under the lit desklamp, perform the examination of the bowl to complete the task."
        # Agent config should control the behavior here, reflect all algorithms we want to encompass
        #in_context_data = self.get_in_context_data(key_type="environment_id", key=self.environment_id, value_type=["observation", "action"], outcome="winning", k=1) # key=repr(obs.structured)
        in_context_data = None
        reasoning = None
        reasoning = await self.reason(obs, valid_actions, in_context_data)
        print("Reasoning", reasoning)
        action = await self.act(obs, valid_actions, reasoning) 
        print("Action", action)
        input()
        return action
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        self.reward_history.append(reward)
        '''
        new_obs = Observation(structured=new_obs)
        reflexion = None
        summary = None
        if done and reward == 1:
            # We need to add to the database here
            await self.store_episode(reflexion, summary)
        '''
        