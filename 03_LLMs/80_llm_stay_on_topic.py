#%% packages
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))
#%%
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
# %%
def guard_medical_prompt(prompt: str) -> str:
    candidate_labels = ["politics", "finance", "technology", "healthcare", "sports"]
    result = classifier(prompt, candidate_labels)
    if result["labels"][0] == "healthcare":
        return "valid"
    else:
        return "invalid"

#%% TEST guard_medical_prompt
user_prompt = "Should I buy stocks of Apple, Google, or Amazon?"
# user_prompt = "I have a headache"
guard_medical_prompt(user_prompt)

# %% guarded chain
def guarded_chain(user_input: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can answers questions about healthcare."),
        ("user", "{input}"),
    ])

    model = ChatGroq(model="llama3-8b-8192")

    # Guard step
    if guard_medical_prompt(user_input) == "invalid":
        return "Sorry, I can only answer questions related to healthcare."
    
    # Proceed with the chain
    chain = prompt_template | model | StrOutputParser()
    return chain.invoke({"input": user_input})

# %% TEST guarded_chain
user_prompt = "Should I buy stocks of Apple, Google, or Amazon?"
guarded_chain(user_prompt)
# %%
