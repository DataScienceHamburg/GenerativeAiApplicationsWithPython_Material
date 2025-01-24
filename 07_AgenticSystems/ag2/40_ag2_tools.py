#%% packages
from typing import Annotated, Literal
import datetime
from autogen import ConversableAgent, UserProxyAgent
from dotenv import load_dotenv, find_dotenv
import os
#  load the environment variables
load_dotenv(find_dotenv(usecwd=True))
# %% llm config_list
config_list = {"config_list": [
    {"model": "gpt-4o-mini", 
     "temperature": 0.9, 
     "api_key": os.environ.get("OPENAI_API_KEY")}]}

#%% tool function
def get_current_date() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d")

# %% create an agent with a tool
my_assistant = ConversableAgent(
    name="my_assistant",
    system_message="""
    You are a helpful AI assistant.
    You can get the current date.
    Return 'TASK COMPLETED' when the task is done.
    """,
    llm_config=config_list,
    # Add human_input_mode to handle tool responses
    human_input_mode="NEVER"
)

# register the tool signature at agent level
my_assistant.register_for_llm(
    name="get_current_date", 
    description="Returns the current date in the format YYYY-MM-DD."
)(get_current_date)

# register the tool function at execution level
# my_assistant.register_for_execution(name="get_current_date")(get_current_date)

# %% create a user proxy to handle the conversation
user_proxy = ConversableAgent(
    name="user_proxy",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TASK COMPLETED" in msg["content"],
    human_input_mode="NEVER"
)
#%% register the tool signature at user proxy level
#%% register the tool function at execution level
user_proxy.register_for_execution(name="get_current_date")(get_current_date)

# %% using the tool through user proxy
result = user_proxy.initiate_chat(
    my_assistant,
    message="What is the current date?"
)
# %% 
print(result)

# %%
