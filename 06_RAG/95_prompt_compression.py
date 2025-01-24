#%% packages
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))
# %% Model
model = ChatGroq(model_name="gemma2-9b-it")

#%% Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Compress the user query. keep essential information, but shorten it as much as possible."),
        ("user", "{input}"),
    ]
)
chain = prompt | model

# %%
long_user_query = "Looking for your dream home? This stunning 2-bedroom flat located in the heart of the city offers modern living with a spacious open-plan living room, large windows that fill the space with natural light, and a sleek, modern kitchen equipped with high-end appliances. The flat includes two large bedrooms with ample closet space, a stylish bathroom with contemporary fittings, and a private balcony that provides a perfect space for relaxation or entertaining. You’ll also enjoy the convenience of a reserved parking space and an extra storage room. Situated in a prime location, you're just minutes away from top restaurants, shopping, and public transport, making it ideal for both commuters and those who enjoy the city's vibrant lifestyle. Whether you're a first-time buyer or a young professional, this low-maintenance, move-in-ready flat combines modern design with a welcoming atmosphere. Don’t miss out on this opportunity! Contact us today to schedule a viewing. Priced at €320,000."

# %%
res = chain.invoke({"input": long_user_query})
print(res.content)
# %% get model dump
res.model_dump()
# %% calculate compression ratio
compression_ratio = (len(long_user_query) - len(res.content)) / len(long_user_query) *100

print(f"Compression ratio: {compression_ratio:.2f} %")
# %%
