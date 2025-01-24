#%% packages
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from pprint import pprint
load_dotenv(find_dotenv(usecwd=True))

#%% function for Chain-of-Thought Prompting
def chain_of_thought_prompting(prompt: str, model_name: str = "gemma2-9b-it") -> str:
    model = ChatGroq(model_name=model_name)
    prompt = ChatPromptTemplate.from_messages(messages=[
        ("system", "You are a helpful assistant and answer precise and concise."),
        ("user", f"{prompt} \n think step by step")
    ])
    # print(prompt)
    chain = prompt | model
    return chain.invoke({}).content


# %% Self-Consistency CoT
def self_consistency_cot(prompt: str, number_of_runs: int = 3) -> str:
    # run CoT multiple times
    res = []
    for _ in range(number_of_runs):
        current_res = chain_of_thought_prompting(prompt)
        print(current_res)
        res.append(current_res)
    
    # concatenate all results
    res_concat = ";".join(res)
    self_consistency_prompt = f"You will get multiple answers in <<>>, separated by ; <<{res_concat}>> Extract only the final equations and return the most common equation as it was provided originally. If there is no common equation, return the most likely equation."
    self_consistency_prompt_concat = ";".join(self_consistency_prompt)
    messages = [
        ("system", "You are a helpful assistant and answer precise and concise."),
        ("user", f"{self_consistency_prompt_concat}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages=messages)
    model = ChatGroq(model_name="gemma2-9b-it")
    chain = prompt | model
    return chain.invoke({}).content


#%% Test
user_prompt = "The goal of the Game of 24 is to use the four arithmetic operations (addition, subtraction, multiplication, and division) to combine four numbers and get a result of 24. The numbers are 3, 4, 6, and 8. It is mandatory to use all four numbers. Please check the final equation for correctness. Hints: Identify the basic operations, Prioritize multiplication and division, Look for combinations that make numbers divisible by 24, Consider order of operations, Use parentheses strategically, Practice with different number combinations"

# %%
res = chain_of_thought_prompting(prompt=user_prompt)
#%%
res = self_consistency_cot(prompt=user_prompt, number_of_runs=5)
pprint(res)
# %%
from pyperclip import copy 
copy(res)

# %%
