#%% packages
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv('.env')
#%% Query Expansion Function
def query_expansion(query: str, number: int = 5, model_name: str = "llama3-70b-8192") -> list[str]:
    messages = [
        ("system","""You are part of an information retrieval system. You are given a user query and you need to expand the query to improve the search results. Return ONLY a list of expanded queries. 
        Be concise and focus on synonyms and related concepts.
        Format your response as a Python list of strings.
        The response must:
        1. Start immediately with [
        2. Contain quoted strings
        3. End with ]
        Example correct format:    
        ["alternative query 1", "alternative query 2", "alternative query 3"]
         """),
        ("user", "Please expand the query: '{query}' and return a list of {number} expanded queries.")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | ChatGroq(model_name=model_name)
    res = chain.invoke({"query": query, "number": number})
    return eval(res.content)

#%%
res = query_expansion(query="Albert Einstein", number=3)
res
# %%
