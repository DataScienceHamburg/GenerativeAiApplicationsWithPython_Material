#%% Packages
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from pprint import pprint
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# %% Load the article
ai_article_title = "Artificial_intelligence"
loader = WikipediaLoader(query=ai_article_title, 
                             load_all_available_meta=True, 
                             doc_content_chars_max=10000, 
                             load_max_docs=1)
doc = loader.load()

# %% Create splitter instance
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                            chunk_overlap=200,
                                            separators=["\n\n", "\n"," ", ".", ","])

# %% Apply semantic chunking
chunks = splitter.split_documents(doc)
# %% Number of Chunks
len(chunks)

# %% Create instance of embedding model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# %% extract the texts from "page_content" attribute of each chunk
texts = [chunk.page_content for chunk in chunks]
# %% create embeddings
embeddings = embeddings_model.embed_documents(texts=texts)

# %% get number of embeddings
len(embeddings)
# %% check the dimension of the embeddings
len(embeddings[0])
# %%
