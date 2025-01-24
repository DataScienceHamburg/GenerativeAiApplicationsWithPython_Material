#%% packages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv('.env')

#%% Model Instance
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#%% Prepare Prompts
# example: style variations (friendly, polite) vs. (savage, angry)
polite_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Reply in a friendly and polite manner."),
    ("human", "{topic}")
])

savage_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Reply in a savage and angry manner."),
    ("human", "{topic}")
])

#%% Prepare Chains
polite_chain = polite_prompt | llm | StrOutputParser()
savage_chain = savage_prompt | llm | StrOutputParser()


# %% Runnable Parallel
map_chain = RunnableParallel(
    polite=polite_chain,
    savage=savage_chain
)

# %% Invoke
topic = "What is the meaning of life?"
result = map_chain.invoke({"topic": topic})
# %% Print
from pprint import pprint
pprint(result)
# %%
