#%% packages
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
load_dotenv(find_dotenv(usecwd=True))


# Load documents for retrieval (can be replaced with any source of text)
# Here we're using a text loader with some sample text files as an example
#%% import wikipedia
loader = WikipediaLoader("Principle of relativity",     
                         load_max_docs=10)
docs = loader.load()

#%% create chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)


#%% models and tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embedding = OpenAIEmbeddings()
search_tool = TavilySearchResults(max_results=5, include_answer=True)

#%% use FAISS to store the chunks
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever(return_similarities=True)

#%% user query

query = "What is relativity?"
#%% RAG chain
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
     You are a helpful assistant that can answer questions about the principle of relativity. You will get contextual information from the retrieved documents. If you don't know the answer, just say 'insufficient information'
     """),
    ("user", "<context>{context}</context>\n\n<question>{question}</question>"),
])
retrieved_docs = retriever.invoke(query)
retrieved_docs_str = ";".join([doc.page_content for doc in retrieved_docs])
chain = prompt_template | llm
rag_response = chain.invoke({"question": query, 
                             "context": retrieved_docs_str})
#%%

if rag_response.content == "insufficient information":
    print("using search tool")
    final_response = search_tool.invoke({"query": query})
    final_response_str = ";".join([doc['content'] for doc in final_response])
    final_response = chain.invoke({"question": query, 
                                     "context": final_response_str})
else:
    print("using vector store")
    final_response = rag_response.content

final_response

# %%
