import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from urllib.parse import quote
from ..base_env import BaseEnv, Observation, Action

WEBSHOP_URL = f"http://localhost:3000/"

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )

def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
      url = (
          f'{WEBSHOP_URL}/{session}'
      )
    if page_type == 'search':
      url = (
          f'{WEBSHOP_URL}/search_results/{session}/'
          f'{query_string}/{page_num}'
      )
    elif page_type == 'item':
      url = (
          f'{WEBSHOP_URL}/item_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{options}'
      )
    elif page_type == 'item_sub':
      url = (
          f'{WEBSHOP_URL}/item_sub_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
      )
    elif page_type == 'end':
      url = (
          f'{WEBSHOP_URL}/done/{session}/'
          f'{asin}/{options}'
      )
    html = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))
    # visible_texts = [str(text).strip().strip('\\n') for text in visible_texts]
    # if page_type == 'end': import pdb; pdb.set_trace()
    if False:
        # For `simple` mode, return just [SEP] separators
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        # Otherwise, return an observation with tags mapped to specific, unique separators
        observation = ''
        option_type = ''
        options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
            # if t.startswith('Instruction:') and page_type != 'init': continue
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                    # observation = f'You have clicked {t}.\n' + observation
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
                # options[option_type] = options.get(option_type, []) + [str(t)]
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                if prod_cnt >= 3:
                  processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t =  '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        info = {}
        if options:
          info['option_types'] = options
        if asins:
          info['asins'] = asins
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
          idx = visible_texts.index('Your score (min 0.0, max 1.0)')
          info['reward'] = float(visible_texts[idx + 1])
          observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        # Retrieve images available on webpage
        if page_type == 'search' or page_type == 'item':
          info['img'] = list(filter(tag_visible, html_obj.findAll(lambda tag: (tag.name == 'img' and tag.has_attr('src')))))
        # Get starting instruction text
        instruction = html_obj.find(id='instruction-text')
        if instruction is not None:
          instruction = instruction.h4
          if instruction is not None:
            instruction = instruction.text
        else:
          instruction = html_obj.find(id='goal-instruction-text')
          if instruction is not None:
            instruction = instruction.pre
            if instruction is not None:
              instruction = instruction.text
        info['instruction'] = instruction #if instruction is not None else ''
        query = html_obj.find(id='goal-query')
        if query is not None:
            query = query.pre
            if query is not None:
              query = query.text
        info['query'] = query if query is not None else ''
        category = html_obj.find(id='goal-category')
        if category is not None:
            category = category.pre
            if category is not None:
              category = category.text
        info['category'] = category if category is not None else ''
        return clean_str(observation), info
    
class WebShopEnv(BaseEnv):
  def __init__(self, config):
    super().__init__(config)
    self.sessions = {}
    self.id = config['problem_id']
    # Run reset once to get the goal
    obs, info = self.reset()
    self.goal = obs.split('Instruction: ')[1].split('[')[0].strip()
    self.max_steps = config['max_steps']

  # In this case reset just calls 
  def reset(self):
    obs, _, _, info = self.step('reset')
    return obs, info
  
  def step(self, action):
    done = False
    observation_ = None
    try:
      if action == 'reset':
        self.sessions[self.id] = {'session': self.id, 'page_type': 'init'}
      elif action.startswith('search['):
        assert self.sessions[self.id]['page_type'] == 'init'
        query = action[7:-1]
        self.sessions[self.id] = {'session': self.id, 'page_type': 'search',
                                  'query_string': query, 'page_num': 1}
      elif action.startswith('click['):
        button = action[6:-1]
        if button == 'Buy Now':
          assert self.sessions[self.id]['page_type'] == 'item'
          # Help URI Encoding, as WSGI error thrown when option has '#'
          if 'options' in self.sessions[self.id]:
              for option_type in self.sessions[self.id]['options']:
                  self.sessions[self.id]['options'][option_type] = quote(self.sessions[self.id]['options'][option_type])
          self.sessions[self.id]['page_type'] = 'end'
          done = True
        elif button == 'Back to Search':
          assert self.sessions[self.id]['page_type'] in ['search', 'item_sub', 'item']
          self.sessions[self.id] = {'session': self.id, 'page_type': 'init'}
        elif button == 'Next >':
          assert False # ad hoc page limitation
          assert self.sessions[self.id]['page_type'] == 'search'
          self.sessions[self.id]['page_num'] += 1
        elif button == '< Prev':
          assert self.sessions[self.id]['page_type'] in ['search', 'item_sub', 'item']
          if self.sessions[self.id]['page_type'] == 'search':
            assert False
            self.sessions[self.id]['page_num'] -= 1
          elif self.sessions[self.id]['page_type'] == 'item_sub':
            self.sessions[self.id]['page_type'] = 'item'
          elif self.sessions[self.id]['page_type'] == 'item':
            self.sessions[self.id]['page_type'] = 'search'
            self.sessions[self.id]['options'] = {}
        elif button in ACTION_TO_TEMPLATE:
          assert self.sessions[self.id]['page_type'] == 'item'
          self.sessions[self.id]['page_type'] = 'item_sub'
          self.sessions[self.id]['subpage'] = button
        else:
          if self.sessions[self.id]['page_type'] == 'search':
            assert button in self.sessions[self.id].get('asins', [])  # must be asins
            print("All asins:", self.sessions[self.id]['asins'])
            self.sessions[self.id]['page_type'] = 'item'
            self.sessions[self.id]['asin'] = button
          elif self.sessions[self.id]['page_type'] == 'item':
            assert 'option_types' in self.sessions[self.id]
            assert button in self.sessions[self.id]['option_types'], (button, self.sessions[self.id]['option_types'])  # must be options
            print("All options:", self.sessions[self.id]['option_types'])
            option_type = self.sessions[self.id]['option_types'][button]
            if not 'options' in self.sessions[self.id]:
              self.sessions[self.id]['options'] = {}
            self.sessions[self.id]['options'][option_type] = button
            observation_ = f'You have clicked {button}.'
          # Vishnu modification--not covering the else situation
          else:
            # This is invalid from item_sub...
            assert False
      else:
        assert False
      observation, info = webshop_text(**self.sessions[self.id])
      if observation_:
        observation = observation_
      self.sessions[self.id].update(info)
      reward = info.get('reward', 0.0)
    except Exception as e:
      print(e)
      # Return "Invalid action!" as observation
      observation = "Invalid action!"
      reward = 0.0
      done = False
      info = {}

    print("Page type:", self.sessions[self.id]['page_type'])
    
    return observation, reward, done, info
'''
env = webshopEnv()
x = env.step('1', 'reset')
print(x)
'''