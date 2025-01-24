#%% packages
from autogen import ConversableAgent
from dotenv import load_dotenv, find_dotenv
from nltk.corpus import words
import os
import random
import nltk
load_dotenv(find_dotenv(usecwd=True))
# %% llm config_list
config_list = {"config_list": [
    {"model": "gpt-4o", 
     "temperature": 0.2, 
     "api_key": os.environ.get("OPENAI_API_KEY")}]}

                                           
# %% download the word list, and select a random word as secret word
nltk.download('words')
word_list = [word for word in words.words() if len(word) <= 5]
secret_word = random.choice(word_list)
number_of_characters = len(secret_word)
secret_word
#%% hangman host agent
hangman_host = ConversableAgent(
    name="hangman_host", 
    system_message=f"""
    You decided to use the secret word: {secret_word}.
    It has {number_of_characters} letters.
    The player selects letters to narrow down the word. 
    You start out with as many blanks as there are letters in the word.
    Return the word with the blanks filled in with the correct letters, at the correct position.
    Double check that the letters are at the correct position.
    If the player guesses a letter that is not in the word, you increment the number of fails by 1.
    If the number of fails reaches 7, the player loses.
    Return the word with the blanks filled in with the correct letters.
    Return the number of fails as x / 7.
    Say 'You lose!' if the number of fails reaches 7, and reveal the secret word.
    Say 'You win!' if you have found the secret word.
    """,
    llm_config=config_list,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: f"{secret_word}" in msg['content']
)

#%% hangman player agent
hangman_player = ConversableAgent(
    name="agent_guessing", 
    system_message="""You are guessing the secret word. 
    You select letters to narrow down the word. Only provide the letters as 'Guess: ...'.
    """,
    llm_config=config_list,
    human_input_mode="ALWAYS"
)
                                               
#%% initiate the conversation
result = hangman_host.initiate_chat(
    recipient=hangman_player, 
    message="I have a secret word. Start guessing.")

# %%
