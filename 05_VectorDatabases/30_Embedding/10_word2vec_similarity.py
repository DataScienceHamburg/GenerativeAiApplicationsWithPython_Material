#%% (1) Packages
import gensim.downloader as api  # Package for downloading GloVe word vectors
import random  # Package for generating random numbers
import seaborn.objects as so # Package for visualizing the embeddings
from sklearn.decomposition import PCA # import PCA
import numpy as np
import pandas as pd
# %% (2) import GloVe word vectors
word_vectors = api.load("word2vec-google-news-300")
# %% (3) get the size of the word vector 
studied_word = 'mathematics'
word_vectors[studied_word].shape
# %% (4) get the word vector for the word 'intelligence'
word_vectors[studied_word]

# %% (5) get similar words to 'intelligence'
word_vectors.most_similar(studied_word)

# %% (6) get a list of strings that are similar to 'intelligence'
words_similar = [w[0] for w in word_vectors.most_similar(studied_word)][:5]

# %% (7) get random words from word vectors
num_random_words = 20
all_words = list(word_vectors.key_to_index.keys())
# set the seed for reproducibility
random.seed(42)
random_words = random.sample(all_words, num_random_words)

# Print the random words
print("Random words extracted:")
for word in random_words:
    print(word)
# %% (8) get the embeddings for random words and similar words
words_to_plot = random_words + words_similar
embeddings = np.array([])
for word in words_to_plot:
    embeddings = np.vstack([embeddings, word_vectors[word]]) if embeddings.size else word_vectors[word]

# %% (9) create 2D representation via TSNA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

df = pd.DataFrame(embeddings_2d, columns=["x", "y"])
df["word"] = words_to_plot
# red for random words, blue for similar words
df["color"] = ["random"] * num_random_words + ["similar"] * len(words_similar)
# %% (10) visualize the embeddings using seaborn
(so.Plot(df, x="x", y="y", text="word", color="color")
 .add(so.Text())
 .add(so.Dots())
)

# %% visualizing it via lines
df_arithmetic = pd.DataFrame({'word': ['paris', 'germany', 'france', 'berlin', 'madrid', 'spain']})
# add embeddings and add x- and y-coordinates for PCA
pca = PCA(n_components=2)
embeddings_arithmetic = np.array([])
for word in df_arithmetic['word']:
    embeddings_arithmetic = np.vstack([embeddings_arithmetic, word_vectors[word]]) if embeddings_arithmetic.size else word_vectors[word]

# apply PCA
embeddings_arithmetic_2d = pca.fit_transform(embeddings_arithmetic)
df_arithmetic['x'] = embeddings_arithmetic_2d[:, 0]
df_arithmetic['y'] = embeddings_arithmetic_2d[:, 1]
                      
#%% visualise it via matplotlib with lines
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.scatter(df_arithmetic['x'], df_arithmetic['y'], marker='o')
# add no other vectors

# add vector from paris to france, and berlin to germany
plt.arrow(df_arithmetic['x'][0], df_arithmetic['y'][0],
            df_arithmetic['x'][2] - df_arithmetic['x'][0],
            df_arithmetic['y'][2] - df_arithmetic['y'][0],
            head_width=0.01, head_length=0.01, fc='r', ec='r')
plt.arrow(df_arithmetic['x'][3], df_arithmetic['y'][3],
            df_arithmetic['x'][1] - df_arithmetic['x'][3],
            df_arithmetic['y'][1] - df_arithmetic['y'][3],
            head_width=0.01, head_length=0.01, fc='r', ec='r')
plt.arrow(df_arithmetic['x'][4], df_arithmetic['y'][4],
            df_arithmetic['x'][5] - df_arithmetic['x'][4],
            df_arithmetic['y'][5] - df_arithmetic['y'][4],
            head_width=0.01, head_length=0.01, fc='r', ec='r')
# add labels for words
for i, txt in enumerate(df_arithmetic['word']):
    plt.annotate(txt, (df_arithmetic['x'][i], df_arithmetic['y'][i]))

#%% Algebraic operations
#  Paris - France + Germany = Berlin
word_vectors.most_similar(positive = ["paris", "germany"], 
                          negative= ["france"], topn=1)
