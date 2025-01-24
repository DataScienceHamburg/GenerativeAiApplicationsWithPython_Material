#%% packages
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv('.env')
from pprint import pprint

#%% fetch prompt
prompt = hub.pull("hardkothari/prompt-maker")

#%% get input variables
prompt.input_variables

# %% model
model = ChatOpenAI(model="gpt-4o-mini", 
                   temperature=0)

# %% chain
chain = prompt | model | StrOutputParser()

# %% invoke chain
lazy_prompt = "summer, vacation, beach"
task = "Shakespeare poem"
improved_prompt = chain.invoke({"lazy_prompt": lazy_prompt, "task": task})
# %%
print(improved_prompt)

# %% run model with improved prompt
res = model.invoke(improved_prompt)
print(res.content)

# %%
res = model.invoke("summer, vacation, beach, Shakespeare poem")
print(res.content)
# %%
