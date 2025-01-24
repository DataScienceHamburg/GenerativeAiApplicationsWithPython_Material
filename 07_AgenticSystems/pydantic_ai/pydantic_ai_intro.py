#%%
from langchain.document_loaders import WikipediaLoader
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
import nest_asyncio
nest_asyncio.apply()

#%% load wikipedia article on Alan Turing
loader = WikipediaLoader(query="Alan Turing", load_all_available_meta=True, doc_content_chars_max=100000, load_max_docs=1)
doc = loader.load()

#%% extract page content
page_content = doc[0].page_content

#%% define pydantic model
class PersonDetails(BaseModel):
    date_born: str = Field(description="The date of birth of the person in the format YYYY-MM-DD")
    date_died: str = Field(description="The date of death of the person in the format YYYY-MM-DD")
    publications: list[str] = Field(description="A list of publications of the person")
    achievements: list[str] = Field(description="A list of achievements of the person")
    
# %% agent instance
MODEL = "openai:gpt-4o-mini"
agent = Agent(model=MODEL, result_type=PersonDetails)
result = agent.run_sync(page_content)
    
# %% print result
result.data.model_dump()
