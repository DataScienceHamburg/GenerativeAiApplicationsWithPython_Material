#%% (1) Packages
import os
from langchain.document_loaders import TextLoader

#%% (2) File Handling
# Get the current working directory
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)

# Go up one directory level
parent_dir = os.path.dirname(current_dir)

file_path = os.path.join(parent_dir, "data","HoundOfBaskerville.txt")
file_path

#%% (3) Load a single document
text_loader = TextLoader(file_path=file_path, encoding="utf-8")
doc = text_loader.load()

#%% (4) Understand the document
# Metadata
doc[0].metadata

# %% Page content
doc[0].page_content
