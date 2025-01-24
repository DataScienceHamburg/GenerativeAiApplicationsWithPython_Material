#%% packages
import os
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings 
from langchain_chroma import Chroma
from langchain.schema import Document

#%% load dataset
dataset = load_dataset("MongoDB/embedded_movies", split="train")
# license: https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md

#%% number of films in the dataset
len(dataset)
# %% which keys are in the dataset?
dataset[0].keys()

# %% used keys
# fullplot (will be 'document';used as embedding)
# title (metadata; shown as result)
# genres (metadata; for filtering)
# imdb_rating (metadata; for filtering)
# poster (metadata; shown as result)


# %% Create List of Documents
docs = []
for doc in dataset:
    title = doc['title'] if doc['title'] is not None else ""
    poster = doc['poster'] if doc['poster'] is not None else ""
    genres = ';'.join(doc['genres']) if doc['genres'] is not None else ""
    imdb_rating = doc['imdb']['rating'] if doc['imdb']['rating'] is not None else ""
    meta = {'title': title, 'poster': poster, 'genres': genres, 'imdb_rating': imdb_rating}
    
    if doc['fullplot'] is not None:
        docs.append(Document(page_content=doc["fullplot"], metadata=meta))


# %% Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
docs_chunked = []
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                            chunk_overlap=CHUNK_OVERLAP,
                                            separators=["\n\n", "\n"," ", ".", ","])
chunks = splitter.split_documents(docs)


# %% store chunks in Chroma
embedding_function = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
script_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(script_dir, "db")
if not os.path.exists(db_dir):
    os.makedirs(db_dir)
    db = Chroma(persist_directory=db_dir, embedding_function=embedding_function, collection_name="movies")
    db.add_documents(chunks)

# %% check the result
db.get()

#%% get all genres
genres = set()
for doc in dataset:
    if doc['genres'] is not None:
        genres.update(doc['genres'])



# %% Exercise: Get all genres from the database
documents = db.get()
genres = set()

for metadata in documents['metadatas']:
    genre = metadata.get('genres')
    genres_list = genre.split(';')
    genres.update(genres_list)
    



# %%
