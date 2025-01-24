#%% packages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv('.env')

#%% set up prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that translates English into another language."),
    ("user", "Translate this sentence: '{input}' into {target_language}"),
])

# %% model
model = ChatOpenAI(model="gpt-4o-mini", 
                   temperature=0)

# %% chain
chain = prompt_template | model | StrOutputParser()

# %% invoke chain
res = chain.invoke({"input": "I love programming.", "target_language": "German"})
res
# %%
