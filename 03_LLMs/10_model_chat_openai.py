#%% packages
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# %%
# %% OpenAI models
# https://platform.openai.com/docs/models/overview

# Model pricing
# https://openai.com/api/pricing/
MODEL_NAME = 'gpt-4o-mini'
model = ChatOpenAI(model_name=MODEL_NAME,
                   temperature=0.5, # controls creativity
                   api_key=os.getenv('OPENAI_API_KEY'))

# %%
res = model.invoke("What is a LangChain?")
# %% find out what is in the result
res.dict()
# %% only print content
print(res.content)