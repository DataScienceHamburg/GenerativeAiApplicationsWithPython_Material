#%% (1) Packages
import os
from langchain.document_loaders import TextLoader, DirectoryLoader

#%% (2) Path Handling
# Get the current working directory
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)

# Go up one directory level
parent_dir = os.path.dirname(current_dir)
text_files_path = os.path.join(parent_dir, "data")

#%% (3) load all files in a directory
dir_loader = DirectoryLoader(path=text_files_path, 
                             glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'} )
docs = dir_loader.load()

# %%
docs

# %% Splitting text
# Packages
from langchain.text_splitter import CharacterTextSplitter
# Split by characters (2)
splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=50, separator=" ")
# %%
docs_chunks = splitter.split_documents(docs)
# %% Check the number of chunks
len(docs_chunks)
# %% check some random Documents (5)
from pprint import pprint
pprint(docs_chunks[100].page_content)
# %%
pprint(docs_chunks[101].page_content)

# %% visualize the chunk size (6)
import seaborn as sns
import matplotlib.pyplot as plt
# get number of characters in each chunk
chunk_lengths = [len(chunk.page_content) for chunk in docs_chunks]

sns.histplot(chunk_lengths, bins=50, binrange=(100, 300))
# add title
plt.title("Distribution of chunk lengths")
# add x-axis label
plt.xlabel("Number of characters")
# add y-axis label
plt.ylabel("Number of chunks")
# %%
