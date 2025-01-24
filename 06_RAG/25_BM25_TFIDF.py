#%% packages
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from typing import List
import string
#%% Documents
def preprocess_text(text: str) -> List[str]:
    # Remove punctuation and convert to lowercase
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

corpus = [
    "Artificial intelligence is a field of artificial intelligence. The field of artificial intelligence involves machine learning. Machine learning is an artificial intelligence field. Artificial intelligence is rapidly evolving.",
    "Artificial intelligence robots are taking over the world. Robots are machines that can do anything a human can do. Robots are taking over the world. Robots are taking over the world.",
    "The weather in tropical regions is typically warm. Warm weather is common in these regions, and warm weather affects both daily life and natural ecosystems. The warm and humid climate is a defining feature of these regions.",
    "The climate in various parts of the world differs. Weather patterns change due to geographic features. Some regions experience rain, while others are dry."
]

# Preprocess the corpus
tokenized_corpus = [preprocess_text(doc) for doc in corpus]
# %% Sparse Search (BM25)
bm25 = BM25Okapi(tokenized_corpus)

#%% Set up user query
user_query = "humid climate"

tokenized_query_BM25 = user_query.lower().split()
tokenized_query_tfidf = ' '.join(tokenized_query_BM25)
# Process query to remove stop words

bm25_similarities = bm25.get_scores(tokenized_query_BM25)
print(f"Tokenized Query BM25: {tokenized_query_BM25}")
print(f"Tokenized Query TFIDF: {tokenized_query_tfidf}")
print(f"BM25 Similarities: {bm25_similarities}")

#%% calculate tfidf
tfidf = TfidfVectorizer()
tokenized_corpus_tfidf = [' '.join(words) for words in tokenized_corpus]
tfidf_matrix = tfidf.fit_transform(tokenized_corpus_tfidf)

query_tfidf_vec = tfidf.transform([tokenized_query_tfidf])
tfidf_similarities = cosine_similarity(query_tfidf_vec, tfidf_matrix).flatten()
print(f"TFIDF Similarities: {tfidf_similarities}")

# %%
