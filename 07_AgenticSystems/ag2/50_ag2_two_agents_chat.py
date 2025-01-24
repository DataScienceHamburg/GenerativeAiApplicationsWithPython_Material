#%% packages
import os
from autogen import ConversableAgent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# %% llm config_list
config_list = {"config_list": [
    {"model": "gpt-4o-mini", 
     "temperature": 0.9, 
     "api_key": os.environ.get("OPENAI_API_KEY")}]}



student_agent = ConversableAgent(
    name="Student_Agent",
    system_message="You are a student willing to learn.",
    llm_config=config_list,
)
teacher_agent = ConversableAgent(
    name="Teacher_Agent",
    system_message="You are a math teacher.",
    llm_config=config_list,
)
#%% initiate chat
chat_result = student_agent.initiate_chat(
    teacher_agent,
    message="What is triangle inequality?",
    summary_method="reflection_with_llm",
    max_turns=2,
)
# %%
print(chat_result.summary)

# %%
ConversableAgent.DEFAULT_SUMMARY_PROMPT
