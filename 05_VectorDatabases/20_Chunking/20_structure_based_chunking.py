#%% Packages
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

#%% Path Handling
# Get the current working directory
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)

# Go up one directory level
parent_dir = os.path.dirname(current_dir)
file_path = os.path.join(parent_dir, "data", "HoundOfBaskerville.txt")

#%% load all files in a directory
loader = TextLoader(file_path=file_path, 
                    encoding="utf-8")
docs = loader.load()

# %%
docs

# %% Set up the splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                         chunk_overlap=200,
                                          separators=["\n\n", "\n"," ", ".", ","])

# %% Create the chunks
doc_chunks = splitter.split_documents(docs)
# %% Number of chunks
len(doc_chunks)

#%% 
chroma_path = os.path.join(parent_dir, "db")

# %%
