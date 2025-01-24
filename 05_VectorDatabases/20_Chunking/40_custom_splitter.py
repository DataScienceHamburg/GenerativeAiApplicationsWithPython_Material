#%% Packages
import re
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

# %% Custom splitter
def custom_splitter(text):
    # This pattern looks for Roman numerals followed by a title
    pattern = r'\n(?=[IVX]+\.\s[A-Z])' 
    return re.split(pattern, text)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# Override the default split method
text_splitter.split_text = custom_splitter

# Assuming you have the full text in a variable called 'full_text'
books = text_splitter.split_documents(data)
# %% remove the first element, because it only holds metadata, not real books
books = books[1: ]

#%% Extract the book title from beginning of page content
for i in range(len(books)):
    print(i)
    # extract title
    pattern = r'\b[IVXLCDM]+\.\s+([A-Z\s\-]+)\r\n'
    match = re.match(pattern, books[i].page_content)
    if match:
        title = match.group(1).replace("\r", "").replace("\n", "")
        print(title)
    # add title to metadata
    books[i].metadata["title"] = title
    print(title)


# %% apply RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(books)
len(chunks)
# %%
chunks
# %%
