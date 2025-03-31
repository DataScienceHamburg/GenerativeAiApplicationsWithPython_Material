#%% packages
from agents import Agent, Runner
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

# %%
agent = Agent(name="my_first_agent",
              instructions="You are a helpful assistant that can answer questions and help with tasks.")

# %% run the agent
response = await Runner.run(agent, 
                      input="Hello, what is OpenAI Agents?")

# %% model output
response.final_output
# %%
