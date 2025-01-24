#%% Packages
import os
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings 
from langchain.schema import Document
from langchain.vectorstores import Chroma
#%%
def create_chunks(text_file_name:str) -> list[Document]:
    # Path Handling
    # Get the current working directory
    file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(file_path)

    # Go up one directory level
    parent_dir = os.path.dirname(current_dir)
    text_file_path = os.path.join(parent_dir, "data", text_file_name)

    # load all files in a directory
    loader = TextLoader(file_path=text_file_path,
                            encoding="utf-8")
    docs = loader.load()

    # Set up the splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                            chunk_overlap=200,
                                            separators=["\n\n", "\n"," ", ".", ","])
    chunks = splitter.split_documents(docs)
    return chunks
# %%
