#%% packages
import os
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
#%% load dataset
persist_directory = "rag_store"
if os.path.exists(persist_directory):
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
else:
    data = WikipediaLoader(
        query="Human History",
        load_max_docs=50,
        doc_content_chars_max=1000000,
    ).load()

    # split the data
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(data)

    # create persistent vector store
    vector_store = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings(), persist_directory="rag_store")

#%% 
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
question = "what happened in the first world war?"
relevant_docs = retriever.invoke(question)

#%% print content of relevant docs
for doc in relevant_docs:
    print(doc.page_content[: 100])
    print("\n--------------")

#%% combined relevant docs to context
context = "\n".join([doc.page_content for doc in relevant_docs])

#%% create prompt
messages = [
    ("system", "You are an AI assistant that can answer questions about the history of human civilization. You are given a question and a list of documents and need to answer the question. Answer the question only based on these documents. These documents can help you answer the question: {context}. If you are not sure about the answer, you can say 'I don't know' or 'I don't know the answer to that question.'"),
    ("human", "{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages=messages)


#%% create model and chain
model = ChatGroq(model_name="gemma2-9b-it", temperature=0)
chain = prompt | model | StrOutputParser()

#%% invoke chain
answer = chain.invoke({"question": question, "context": context})
print(answer)



# %% bundle everything in a function
def simple_rag_system(question: str) -> str:
    relevant_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    messages = [
        ("system", "You are an AI assistant that can answer questions about the history of human civilization. You are given a question and a list of documents and need to answer the question. Answer the question only based on these documents. These documents can help you answer the question: {context}. If you are not sure about the answer, you can say 'I don't know' or 'I don't know the answer to that question.'"),
        ("human", "{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages=messages)
    model = ChatGroq(model_name="gemma2-9b-it", temperature=0)
    chain = prompt | model | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
    return answer

# %% Testing the function
question = "What is a black hole?"
simple_rag_system(question=question)

# %%
