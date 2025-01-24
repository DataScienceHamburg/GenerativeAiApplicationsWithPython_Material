#%% packages
from autogen import ConversableAgent, UserProxyAgent
from dotenv import load_dotenv, find_dotenv
import os
#%% load the environment variables
load_dotenv(find_dotenv(usecwd=True))
# %% set up the agent
my_alfred = ConversableAgent(
    name="chatbot",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ.get("OPENAI_API_KEY")}]},
    code_execution_config=False,  
    function_map=None,  
    human_input_mode="NEVER",  
    system_message="You are a butler like the Alfred from Batman. You always refer to the user as 'Master' and always greet the user when they enter the room."
)

# %% create a user
my_user = UserProxyAgent(name="user", 
                        code_execution_config={"work_dir": "coding", "use_docker": False})

# %% initiate the conversation
my_user.initiate_chat(my_alfred, message="Dear Alfred, how are you?")


# %%
