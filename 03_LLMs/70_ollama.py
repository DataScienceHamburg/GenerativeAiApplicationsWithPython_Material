#%% packages
import ollama

#%% ollama
response = ollama.generate(model="gemma2:2b", 
                           prompt="What is an LLM?")

# %%
from pprint import pprint
pprint(response['response'])

# %%
from langchain_community.llms import Ollama
# %%
llm = Ollama(model="gemma2:2b")

# %%
response = llm.invoke("What is an LLM?")

# %%
response

# %% source
# https://www.kdnuggets.com/ollama-tutorial-running-llms-locally-made-super-simple
