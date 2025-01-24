#%% packages
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
# Load LLM inference endpoints from an env variable or a file
# and OAI_CONFIG_LIST_sample
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")


#%% set up the agents
assistant = AssistantAgent(name="assistant", 
                           llm_config={"config_list": config_list})
user_proxy = UserProxyAgent(name="user_proxy", 
                            code_execution_config={"work_dir": "coding", "use_docker": False}) # IMPORTANT: set to True to run code in docker, recommended
user_proxy.initiate_chat(assistant, message="Plot a chart of ETH and SOL stock price change YTD.")
# This initiates an automated chat between the two agents to solve the task
# %% set up Docker to run the code directly on the machine
# docker build -f .devcontainer/Dockerfile -t ag2_base_img https://github.com/ag2ai/ag2.git#main
