#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))


#%%
model = ChatGroq(model_name='gemma2-9b-it', temperature=0.0)

#%% first run
messages = [
        ("system", "You are an author and write a childs book.respond short and concise. End your answer with a specific question, that provides a new direction for the story."),
        ("user", "A mouse and a cat are best friends."),
    ]
prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | model
output = chain.invoke({})
output.content

# %% next run
messages.append(("ai", output.content))
messages.append(("user", "The dog is running after the cat."))
prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | model
output = chain.invoke({})
output.content

# %%
