#%% packages
from autogen import ConversableAgent
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

#%% load the environment variables
load_dotenv(find_dotenv(usecwd=True))
import agentops
from agentops import track_agent, record_action
agentops.init()
import logging
logging.basicConfig(
    level=logging.DEBUG
)  # this will let us see that calls are assigned to an agent

openai_client = OpenAI()

@track_agent(name="jack")
class FlatEarthAgent:
    def completion(self, prompt: str):
        res = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are Jack, a flat earth believer who thinks the earth is flat and tries to convince others. You communicate in a passionate but friendly way.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return res.choices[0].message.content


@track_agent(name="alice")
class ScientistAgent:
    def completion(self, prompt: str):
        res = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are Alice, a scientist who uses evidence and logic to explain scientific concepts. You are patient and educational in your responses.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        return res.choices[0].message.content

jack = FlatEarthAgent()
alice = ScientistAgent()

flat_earth_argument = jack.completion("Explain why you think the earth is flat")

@record_action(event_name="make_flat_earth_argument")
def make_flat_earth_argument():
    return jack.completion("Explain why you think the earth is flat")


@record_action(event_name="respond_with_science")
def respond_with_science():
    return alice.completion(
        "Respond to this flat earth argument with scientific evidence: \n" + flat_earth_argument
    )
    
make_flat_earth_argument()

respond_with_science()

# end session
agentops.end_session(end_state="Success")