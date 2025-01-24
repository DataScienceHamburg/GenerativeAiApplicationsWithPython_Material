#%% packages
from langchain_openai import OpenAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
#%% Documents
docs = [
    "The weather tomorrow will be sunny with a slight chance of rain.",
"Dogs are known to be loyal and friendly companions to humans.",
"The climate in tropical regions is warm and humid, often with frequent rain.",
"Python is a powerful programming language used for machine learning.",
"The temperature in deserts can vary widely between day and night.",
"Cats are independent animals, often more solitary than dogs.",
"Artificial intelligence and machine learning are rapidly evolving fields.",
"Hiking in the mountains is an exhilarating experience, but it can be unpredictable due to weather changes.",
"Winter sports like skiing and snowboarding require specific types of weather conditions.",
"Programming languages like Python and JavaScript are popular choices for web development."
]

#%% remove stop words for sparse similarity
docs_without_stopwords = [
    ' '.join([word for word in doc.split() if word.lower() not in ENGLISH_STOP_WORDS])
    for doc in docs
]
# %% Sparse Search
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs_without_stopwords)

#%% Set up user query
user_query = "Which weather is good for outdoor activities?"

query_sparse_vec = vectorizer.transform([user_query])
sparse_similarities = cosine_similarity(query_sparse_vec, tfidf_matrix).flatten()

#%% filter documents below threshold
def getFilteredDocsIndices(similarities, threshold = 0.0):
    filt_docs_indices = sorted(
        [(i, sim) for i, sim in enumerate(similarities) if sim > threshold],
        key=lambda x: x[1],
        reverse=True
    )
    return [i for i, sim in filt_docs_indices]
    
#%% filter documents below threshold and get indices
filtered_docs_indices_sparse = getFilteredDocsIndices(similarities=sparse_similarities, threshold=0.2)
filtered_docs_indices_sparse

# %% Dense Search
embeddings = OpenAIEmbeddings()
embedded_docs = [embeddings.embed_query(doc) for doc in docs]

#%% embed user query
query_dense_vec = embeddings.embed_query(user_query)

#%% calculate cosine similarity
dense_similarities = cosine_similarity([query_dense_vec], embedded_docs)
dense_similarities
#%%
filtered_docs_indices_dense = getFilteredDocsIndices(similarities=dense_similarities[0], threshold=0.8)
filtered_docs_indices_dense

# %% Reciprocal Rank Fusion
def reciprocal_rank_fusion(filtered_docs_indices_sparse, filtered_docs_indices_dense, alpha=0.2):
    # Create a dictionary to store the ranks
    rank_dict = {}
    
    # Assign ranks for sparse indices
    for rank, doc_index in enumerate(filtered_docs_indices_sparse, start=1):
        if doc_index not in rank_dict:
            rank_dict[doc_index] = 0
        rank_dict[doc_index] += (1 / (rank + 60)) * alpha
    
    # Assign ranks for dense indices
    for rank, doc_index in enumerate(filtered_docs_indices_dense, start=1):
        if doc_index not in rank_dict:
            rank_dict[doc_index] = 0
        rank_dict[doc_index] += (1 / (rank + 60)) * (1 - alpha)
    
    # Sort the documents by their reciprocal rank fusion score
    sorted_docs = sorted(rank_dict.items(), key=lambda item: item[1], reverse=True)
    
    # Return the sorted document indices
    return [doc_index for doc_index, _ in sorted_docs]

#%% Example usage
reciprocal_rank_fusion(filtered_docs_indices_sparse, filtered_docs_indices_dense, alpha=0.2)


# %%
