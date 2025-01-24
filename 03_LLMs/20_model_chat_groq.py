#%% packages
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# %%
# Model overview: https://console.groq.com/docs/models
MODEL_NAME = 'llama-3.1-70b-versatile'
model = ChatGroq(model_name=MODEL_NAME,
                   temperature=0.5, # controls creativity
                   api_key=os.getenv('GROQ_API_KEY'))

# %% Run the model
res = model.invoke("What is a Huggingface?")
# %% find out what is in the result
res.dict()
# %% only print content
print(res.content)
# %%
