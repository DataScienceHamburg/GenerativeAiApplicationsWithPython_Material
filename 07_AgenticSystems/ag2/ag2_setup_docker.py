#%% packages
from pathlib import Path
from autogen import UserProxyAgent
from autogen.coding import DockerCommandLineCodeExecutor
from autogen import AssistantAgent
from dotenv import load_dotenv, find_dotenv
import os
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# and OAI_CONFIG_LIST_sample
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
#%% load the environment variables
load_dotenv(find_dotenv(usecwd=True))
#%% set up the work directory
work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)

#%% set up the code executor
with DockerCommandLineCodeExecutor(work_dir=work_dir) as code_executor:
    assistant = AssistantAgent(name="assistant", 
                           llm_config={"config_list": config_list})
    user_proxy = UserProxyAgent(name="user_proxy", 
                                code_execution_config={"work_dir": "coding", "use_docker": True}) # IMPORTANT: set to True to run code in docker, recommended
    user_proxy.initiate_chat(assistant, message="Plot a chart of ETH and SOL stock price change YTD.")
# %%
