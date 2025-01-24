#%% packages
from transformers import pipeline

#%%
unmasker = pipeline(task='fill-mask', model='bert-base-uncased')
unmasker("I am a [MASK] model.")
# %%

