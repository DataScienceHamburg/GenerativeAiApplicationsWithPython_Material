
#%% packages
from autogen import ConversableAgent
from dotenv import load_dotenv, find_dotenv
import os
#%% load the environment variables
load_dotenv(find_dotenv(usecwd=True))
import streamlit as st

#%% LLM config
llm_config = {"config_list": [
    {"model": "gpt-4o-mini", 
     "temperature": 0.9, 
     "api_key": os.environ.get("OPENAI_API_KEY")}]}

st.title("Controversial Debate")

prompt = st.chat_input("Enter a topic to debate about:")
if prompt:
    st.header(f"Topic: {prompt}")

with st.expander("Conversation Settings"):
    number_of_turns = st.slider("Number of turns", min_value=1, max_value=10, value=1)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Style of Person A")
        style_a = st.radio(
            "Choose style for first speaker:",
            ["Friendly", "Neutral", "Unfriendly"],
            key="style_a"
        )

    with col2:
        st.subheader("Style of Person B") 
        style_b = st.radio(
            "Choose style for second speaker:",
            ["Friendly", "Neutral", "Unfriendly"],
            key="style_b"
        )





if prompt:
    #%% set up the agent: Jack, the flat earther
    person_a = ConversableAgent(
        name="user",
        system_message=f"""
        You are a person who believes that {prompt}. 
        You try to convince others of this. 
        You answer in a {style_a} way.
        Answer very short and concise.
        """,
        llm_config=llm_config,
        human_input_mode="NEVER", 
    )

    #%% set up the agent: Alice, the scientist
    person_b = ConversableAgent(
        name="ai",
        system_message="""
        You are a person who believes the opposite of {prompt}. 
        You answer in a {style_b} way.
        Answer very short and concise.
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",  
    )

    # %% start the conversation
    result = person_a.initiate_chat(
        recipient=person_b, 
        message=prompt, 
        max_turns=number_of_turns)
    
    messages = result.chat_history
    for message in messages:
        name = message["name"]
        if name == "user":
            with st.container():
                col1, col2 = st.columns([3, 7])
                with col2:
                    with st.chat_message(name=name):    
                        st.write(message["content"])
        else:
            with st.container():
                col1, col2 = st.columns([7, 3])
                with col1:
                    with st.chat_message(name=name):
                        st.write(message["content"])
