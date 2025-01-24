#%% packages
import os
from dotenv import load_dotenv, find_dotenv
from autogen import ConversableAgent

#%% load the environment variables
load_dotenv(find_dotenv(usecwd=True))

#%% define the function to predict
def predict_conversation(user_prompt: str, number_of_turns: int):
    llm_config = {"config_list": [
    {"model": "gpt-4o-mini", 
     "temperature": 0.9, 
     "api_key": os.environ.get("OPENAI_API_KEY")}]}
    person_a = ConversableAgent(
            name="user",
            system_message=f"""
            You are a person who believes that {user_prompt}. 
            You try to convince others of this. 
            You answer in a friendly way.
            Answer very short and concise.
            """,
            llm_config=llm_config,
            human_input_mode="NEVER", 
        )

    # set up the agent: Alice, the scientist
    person_b = ConversableAgent(
        name="ai",
        system_message="""
        You are a person who believes the opposite of {user_prompt}. 
        You answer in a {style_b} way.
        Answer very short and concise.
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",  
    )

        #  start the conversation
    result = person_a.initiate_chat(
        recipient=person_b, 
        message=user_prompt, 
        max_turns=number_of_turns)
        
    messages = result.chat_history
    return messages