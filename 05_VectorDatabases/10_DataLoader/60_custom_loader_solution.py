#%% Packages
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import GutenbergLoader
# %% The book details
book_details = {
    "title": "The Adventures of Sherlock Holmes",
    "author": "Arthur Conan Doyle",
    "year": 1892,
    "language": "English",
    "genre": "Detective Fiction",
    "url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
}

loader = GutenbergLoader(book_details.get("url"))
data = loader.load()

#%% Add metadata from book_details
data[0].metadata = book_details
