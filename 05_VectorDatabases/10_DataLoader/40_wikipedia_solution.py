#%% Packages
from langchain.document_loaders import WikipediaLoader

#%% Articles to load
articles = [
    {'title': 'Artificial Intelligence'},
    {'title': 'Artificial General Intelligence'},
    {'title': 'Superintelligence'},
]

# %% Load all articles (2)
docs = []
for i in range(len(articles)):
    print(f"Loading article on {articles[i].get('title')}")
    loader = WikipediaLoader(query=articles[i].get("title"), 
                             load_all_available_meta=True, 
                             doc_content_chars_max=100000, 
                             load_max_docs=1)
    doc = loader.load()
    docs.append(doc)


# %%
docs

# %%
