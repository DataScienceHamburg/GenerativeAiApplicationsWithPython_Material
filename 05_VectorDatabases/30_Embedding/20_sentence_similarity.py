#%% (1) Packages
from sentence_transformers import SentenceTransformer
import numpy as np
import seaborn as sns

#%% (2) Load the model
MODEL = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
model = SentenceTransformer(MODEL)
# %% (3) Define the sentences
sentences = [
    'The cat lounged lazily on the warm windowsill.',
    'A feline relaxed comfortably on the sun-soaked ledge.',
    'The kitty reclined peacefully on the heated window perch.',
    'Quantum mechanics challenges our understanding of reality.',
    'The chef expertly julienned the carrots for the salad.',
    'The vibrant flowers bloomed in the garden.',
    'Las flores vibrantes florecieron en el jardín. ',
    'Die lebhaften Blumen blühten im Garten.'
]
# %% (4) Get the embeddings
sentence_embeddings = model.encode(sentences)

# %% (5) Calculate linear correlation matrix for embeddings
sentence_embeddings_corr = np.corrcoef(sentence_embeddings)
import seaborn as sns
# show annotation with one digit
sns.heatmap(sentence_embeddings_corr, annot=True,
            fmt=".1f",
            xticklabels=sentences, 
            yticklabels=sentences)