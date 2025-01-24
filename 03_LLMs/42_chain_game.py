#%% Packages
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.console import Console
console = Console()
load_dotenv(".env")

#%% Prepare LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
# %% Session history
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

#%% Begin the story
from langchain.prompts import ChatPromptTemplate

initial_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative storyteller. Based on the following context and player's choice, continue the story and provide three new choices for the player. keep the story extremely short and concise. Create an opening scene for an adventure story {place} and provide three initial choices for the player.")
])

context_chain = initial_prompt | llm

config = {"configurable": {"session_id": "03"}}

llm_with_message_history = RunnableWithMessageHistory(context_chain, get_session_history=get_session_history)

context = llm_with_message_history.invoke({"place": "a dark forest"}, config=config)

# render opening scene as markdown output
console.print(Markdown(context.content))

#%% Function to process player's choice
def process_player_choice(choice):
    response = llm_with_message_history.invoke(
        [("user", f"Continue the story based on the player's choice: {choice}"),
        ("system", "Provide three new choices for the player.")]
        , config=config)
    return response

# %% Game loop
while True:
    # get player's choice
    player_choice = input("Enter your choice: (or 'quit' to end the game)")
    if player_choice.lower() == "quit":
        break
    # continue the story
    context = process_player_choice(player_choice)
    console.print(Markdown(context.content))
# %%
console.print(Markdown(context.content))
