#%% packages
from transformers import pipeline

#%% model selection
task = "translation"
model = "Mitsua/elan-mt-bt-en-ja"
translator = pipeline(task=task, model=model)

# %%
text = "Be the change you wish to see in the world."
result = translator(text)
result[0]['translation_text']
# %%
