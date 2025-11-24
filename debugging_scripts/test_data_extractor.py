from scienceai.data_extractor import *

# go up two directories to get to the root of the project

with open('test_summaries.json', 'r') as json_file:
    papers = json.load(json_file)

summaries = ""
for paper in papers:
    summaries += paper["title"] + "\n\nSummary: " + paper["summary"] + "\n\n\n"

with open('test_paper.txt', 'r') as file:
    cleaned_text = file.read()

collection_name = "SampleSizeCollection"
collection_goal = ("Collect detailed information on sample sizes from all the papers, "
                   "including any variations or patterns observed.")

schema = generate_schema(summaries, goal=collection_name+" - "+collection_goal)
tools = schema_to_tool(schema)
from pprint import pprint as pp
pp(tools)

results = extract_data(tools, cleaned_text)
pp(results)
