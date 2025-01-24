#%% packages
from dotenv import load_dotenv
import os
load_dotenv(".env")
# %%
os.getenv("PINECONE_API_KEY")
# %% connect to Pinecone instance
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# %% 
index_name = "sherlock"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, 
                    metric="cosine", 
                    dimension=384,
                    spec=ServerlessSpec(
                        cloud = "aws",
                        region="us-east-1"))
# %% Prepare data
from data_prep import create_chunks
chunks = create_chunks("HoundOfBaskerville.txt")

texts = [chunk.page_content for chunk in chunks]


# %% Embedding model
from langchain_huggingface import HuggingFaceEndpointEmbeddings 
embedding_model = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
# %% create all embeddings
embeddings = embedding_model.embed_documents(texts=texts)

# %% create vectors
# {"id": str, "values": List[float], "metadata": Dict[str, str]}
vectors = [{"id": str(i), 
            "values": embeddings[i], 
            "metadata": chunks[i].metadata} 
           for i in range(len(chunks))]
# %%
index = pc.Index(name=index_name)
index.upsert(vectors)

#%% describe index
print(index.describe_index_stats())
