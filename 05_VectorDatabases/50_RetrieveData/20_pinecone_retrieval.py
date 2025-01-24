#%% packages
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv(".env")
from langchain_huggingface import HuggingFaceEndpointEmbeddings 
import os

#%% connect to Pinecone instance
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "sherlock"
index = pc.Index(name=index_name)

#%% Embedding model
embedding_model = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

#%% embed user query
user_query = "How does the hound look like?"
query_embedding = embedding_model.embed_query(user_query)

#%% search for similar documents
res = index.query(vector=query_embedding, top_k=2, include_metadata=True)

#%% get the top 3 matches
res["matches"]

#%% get the text metadata for the top 5 matches
for match in res['matches']:
    print(match['metadata']['text'])
    print("---------------")

# %%
