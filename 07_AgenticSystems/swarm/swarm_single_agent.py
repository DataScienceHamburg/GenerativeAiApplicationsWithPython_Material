#%% packages
from swarm import Swarm, Agent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

# %%
client = Swarm()

agent = Agent(name="my_first_agent",
              instructions="You are a helpful assistant that can answer questions and help with tasks.")

# %% run the agent
messages = [
    {"role": "user", "content": "Hello, what is OpenAI Swarm?"},
]
response = client.run(agent=agent, 
                      messages=messages
                      )

# %% get the last message
response.messages[-1]['content']

# %%
response.model_dump()

# %%

# %%
