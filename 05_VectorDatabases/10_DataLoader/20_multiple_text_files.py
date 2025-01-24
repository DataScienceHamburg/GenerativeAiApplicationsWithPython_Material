#%% (1) Packages
import os
from langchain.document_loaders import TextLoader, DirectoryLoader
from pprint import pprint
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

