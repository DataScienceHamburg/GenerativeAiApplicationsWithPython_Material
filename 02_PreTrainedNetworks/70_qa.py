#%% packages
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from pprint import pprint
#%% constants
MODEL = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline(task='question-answering', model=MODEL, tokenizer=MODEL)
QA_input = {
    'question': 'What are the benefits of remote work?',
    'context': 'Remote work allows employees to work from anywhere, providing flexibility and a better work-life balance. It reduces commuting time, lowers operational costs for companies, and can increase productivity for self-motivated workers.'
}
res = nlp(QA_input)
pprint(res)

# %%
