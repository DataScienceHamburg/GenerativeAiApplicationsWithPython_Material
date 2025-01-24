#%% packages
from datasets import Dataset
from ragas.metrics import context_precision, answer_relevancy, faithfulness
from ragas import evaluate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))
# %%
my_sample = {
    "question": ["What is the capital of Germany in 1960?"],  # The main question
    "contexts": [
        [
            "Berlin is the capital of Germany.", 
            "Between 1949 and 1990, East Berlin was the capital of East Germany.", 
            "Bonn was the capital of West Germany during the same period."
        ]
    ],  # Nested list for multiple contexts
    "answer": ["In 1960, the capital of Germany was Bonn. East Berlin was the capital of East Germany."],
    "ground_truth": ["Berlin"]
}

dataset = Dataset.from_dict(my_sample)
# %%
llm = ChatOpenAI(model="gpt-4o-mini")
metrics = [context_precision, answer_relevancy, faithfulness]
res = evaluate(dataset=dataset, 
               metrics=metrics, 
               llm=llm)
res

