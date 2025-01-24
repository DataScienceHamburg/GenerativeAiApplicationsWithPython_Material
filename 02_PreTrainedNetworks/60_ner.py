#%%
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from pprint import pprint
#%% model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

#%% pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Bert. I live in Hamburg."

ner_results = nlp(example)
pprint(ner_results)
# %%
