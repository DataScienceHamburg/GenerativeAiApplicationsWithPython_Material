#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

#%%
messages = [
    ("system", "You are a customer service specialist known for empathy, professionalism, and problem-solving. Your responses are warm yet professional, solution-focused, and always end with a concrete next step or resolution. You handle both routine inquiries and escalated issues with the same level of care."),
    ("user", """
     Example 1:
     Customer: I received the wrong size shirt in my order #12345.
     Response: I'm so sorry about the sizing mix-up with your shirt order. That must be disappointing! I can help make this right immediately. You have two options:

     I can send you a return label and ship the correct size right away
     I can process a full refund if you prefer

     Which option works better for you? Once you let me know, I'll take care of it right away.
     Example 2:
     Customer: Your website won't let me update my payment method.
     Response: I understand how frustrating technical issues can be, especially when trying to update something as important as payment information. Let me help you with this step-by-step:
     First, could you try clearing your browser cache and cookies?
     If that doesn't work, I can help you update it directly from my end.
     Could you share your account email address so I can assist you further?
     New Request: {customer_request}
     """
     ),
]
prompt = ChatPromptTemplate.from_messages(messages)
MODEL_NAME = 'gemma2-9b-it'
model = ChatGroq(model_name=MODEL_NAME)
chain = prompt | model
# %%
res = chain.invoke({"customer_request": "I haven't received my refund yet after returning the item 2 weeks ago."})

# %%
res.model_dump()['content']
# %%
from pyperclip import copy
