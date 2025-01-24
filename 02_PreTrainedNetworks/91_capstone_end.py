#%% packages
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
from typing import List

#%% data
feedback = [
    "I recently bought the EcoSmart Kettle, and while I love its design, the heating element broke after just two weeks. Customer service was friendly, but I had to wait over a week for a response. It's frustrating, especially given the high price I paid.",
    "Die Lieferung war super schnell, und die Verpackung war großartig! Die Galaxy Wireless Headphones kamen in perfektem Zustand an. Ich benutze sie jetzt seit einer Woche, und die Klangqualität ist erstaunlich. Vielen Dank für ein tolles Einkaufserlebnis!",
    "Je ne suis pas satisfait de la dernière mise à jour de l'application EasyHome. L'interface est devenue encombrée et le chargement des pages prend plus de temps. J'utilise cette application quotidiennement et cela affecte ma productivité. J'espère que ces problèmes seront bientôt résolus."
]


# %%
def process_feedback(feedback: List[str]) -> dict[str, List[str]]:
    """
    Process the feedback and return a DataFrame with the sentiment and the most likely label.
    Input:
        feedback: List[str]
    Output:
        pd.DataFrame
    """
    CANDIDATES = ['defect', 'delivery', 'interface']
    ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
    SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
    # initialize the classifiers
    zero_shot_classifier = pipeline(task="zero-shot-classification", 
                                    model=ZERO_SHOT_MODEL)
    sentiment_classifier = pipeline(task="text-classification", 
                                    model=SENTIMENT_MODEL)

    zero_shot_res = zero_shot_classifier(feedback, 
                                         candidate_labels = CANDIDATES)
    sentiment_res = sentiment_classifier(feedback)
    sentiment_labels = [res['label'] for res in sentiment_res]
    most_likely_labels = [res['labels'][0] for res in zero_shot_res]
    res = {'feedback': feedback, 'sentiment': sentiment_labels, 'label': most_likely_labels}
    return res

#%% Test
process_feedback(feedback)
# %%