#%% packages
from groq import Groq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
import base64
# %%
MODEL = "llama-3.2-90b-vision-preview"
IMAGE_PATH = "sample_image.png"
USER_PROMPT = "What is shown in this image? Answer in one sentence."
# %% 
# source: https://console.groq.com/docs/vision
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image(IMAGE_PATH )
#%% Getting the base64 string
client = Groq()

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model=MODEL,
)

#%% analyze the output
print(chat_completion.choices[0].message.content)
# %%
