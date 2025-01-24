#%% packages
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

#%% Create the agent
memory = MemorySaver()
model = ChatGroq(model_name="llama-3.1-70b-versatile")
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model=model, 
                                    tools=tools, 
                                    checkpointer=memory)

#%% Use the agent
config = {"configurable": {"thread_id": "abcd123"}}

#%%
agent_executor.invoke(
    {"messages": [("user", "My name is Bert Gollnick, I am a trainer and data scientist. I live in Hamburg")]}, config
)

#%% function for extracting the last message from the memory
def get_last_message(memory, config):
    return memory.get_tuple(config=config).checkpoint['channel_values']['messages'][-1].model_dump()['content']

#%% check whether the model can remember me
agent_executor.invoke(
    {"messages": ("user", "What is my name and in which country do I live?")}, config
)
get_last_message(memory, config)
#%% check if it is possible to find me in the internet
agent_executor.invoke(
    {"messages": ("user", "What can you find about me in the internet")},
    config
)
get_last_message(memory, config)

# %%
list(memory.list(config=config))
# %% extract the last message from the memory

# %%
get_last_message(memory, config)
