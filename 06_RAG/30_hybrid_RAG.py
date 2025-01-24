#%% packages
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings 
import os
load_dotenv(".env")
# %%
os.getenv("PINECONE_API_KEY")
# %%
# %% connect to Pinecone instance
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "sherlock"
index = pc.Index(name=index_name)
# %%
print(index.describe_index_stats())
#%%
#%% Embedding model
embedding_model = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

#%% embed user query
user_query = "How does the hound look like?"
query_embedding = embedding_model.embed_query(user_query)

#%% search for similar documents
res = index.query(vector=query_embedding, top_k=2, include_metadata=True)

# %%
res
# %%
