#%% packages
from swarm import Swarm, Agent
# %%
client = Swarm()

#%% define the functions
def transfer_to_german_agent():
    """Transfer to the German Agent."""
    return german_agent

def transfer_to_english_agent():
    """Transfer to the English Agent."""
    return english_agent

#%% define the agents
english_agent = Agent(
    name="English Agent",
    instructions="You are a helpful agent and only speak in English.",
    functions=[transfer_to_german_agent],
)

german_agent = Agent(
    name="German Agent",
    instructions="You are a helpful agent and only speak in German.",
    functions=[transfer_to_english_agent],
)
# %% run the swarm
response = client.run(
    agent=english_agent,
    messages=[{"role": "user", "content": "Ich brauche Hilfe mit meiner Buchung."}],
)

print(response.messages[-1]["content"])
# %%
response.model_dump()
# %%
