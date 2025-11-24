import json
from scienceai.reasoning import *

with open('test_o1.json', 'r') as json_file:
    context = json.load(json_file)

result = add_reasoning_to_context(context)

print(result)
