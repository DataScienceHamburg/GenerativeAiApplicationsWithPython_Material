#%% packages
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from pred_conv import predict_conversation

#%% create the app
app = FastAPI()

#%% create a pydantic model
class Prompt(BaseModel):
    prompt: str
    number_of_turns: int
    
#%% define the function to predict

#%% define the endpoint "predict"
@app.post("/predict")
def predict_endpoint(parameters: Prompt):
    prompt = parameters.prompt
    turns = parameters.number_of_turns
    print(prompt)
    print(turns)
    result = predict_conversation(user_prompt=prompt,
                     number_of_turns=turns)
    return result


# %% run the server
if __name__ == '__main__':
    uvicorn.run("main:app", reload=True)