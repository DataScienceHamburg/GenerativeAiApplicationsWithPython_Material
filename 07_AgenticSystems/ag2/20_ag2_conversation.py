
#%% packages
from autogen import ConversableAgent
from dotenv import load_dotenv, find_dotenv
import os
#%% load the environment variables
load_dotenv(find_dotenv(usecwd=True))

#%% LLM config
llm_config = {"config_list": [
    {"model": "gpt-4o-mini", 
     "temperature": 0.9, 
     "api_key": os.environ.get("OPENAI_API_KEY")}]}

#%% set up the agent: Jack, the flat earther
jack_flat_earther = ConversableAgent(
    name="jack",
    system_message="""
    You believe that the earth is flat. 
    You try to convince others of this. 
    With every answer, you are more frustrated and angry that they don't see it.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER", 
)

#%% set up the agent: Alice, the scientist
alice_scientist = ConversableAgent(
    name="alice",
    system_message="""
    You are a scientist who believes that the earth is round. 
    Answer very polite, short and concise.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",  
)

# %% start the conversation
result = jack_flat_earther.initiate_chat(
    recipient=alice_scientist, 
    message="Hello, how can you not see that the earth is flat?", 
    max_turns=3)
# %%
result.chat_history
# %% 
