#%% packages
from dotenv import load_dotenv, find_dotenv
import anthropic
import os
from langchain_community.document_loaders import TextLoader
from rich.console import Console
from rich.markdown import Markdown

load_dotenv(find_dotenv(usecwd=True))

#%% anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


#%% model class
class PromptCachingChat:
    def __init__(self, initial_context: str):
        self.messages = []
        self.context = None
        self.initial_context = initial_context

    def run_model(self):
        self.context = client.beta.prompt_caching.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system=[
      {
        "type": "text", 
        "text": "You are a patent expert. You are given a patent and will be asked to answer questions about it.\n",
      },
      {
        "type": "text", 
        "text": f"Initial Context: {self.initial_context}",
        "cache_control": {"type": "ephemeral"}
      }
    ],
    messages=self.messages,
    )
        # add the model response to the messages
        self.messages.append({"role": "assistant", "content": self.context.content[0].text})
        return self.context
    
    def user_turn(self, user_query: str):
        self.messages.append({"role": "user", "content": user_query})
        self.context = self.run_model()
        return self.context
    
    def show_model_response(self):
        console = Console()
        
        console.print(Markdown(self.messages[-1]["content"]))
        console.print(f"Usage: {self.context.usage}")


#%% Testing
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
parent_dir = os.path.dirname(current_dir)

file_path = os.path.join(parent_dir, "05_VectorDatabases", "data","HoundOfBaskerville.txt")
file_path

#%% (3) Load a single document
text_loader = TextLoader(file_path=file_path, encoding="utf-8")
doc = text_loader.load()
initialContext = doc[0].page_content
#%%
promptCachingChat = PromptCachingChat(initial_context=initialContext)
promptCachingChat.user_turn("what is special about the hound of baskerville?")
promptCachingChat.show_model_response()
# %%
promptCachingChat.user_turn("Is the hound the murderer?")
promptCachingChat.show_model_response()
print(promptCachingChat.context.usage)

# %%
promptCachingChat.context.usage
