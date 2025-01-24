#%% packages
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
import re
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import JsonOutputParser
load_dotenv(find_dotenv(usecwd=True))

# Initialize ChatOpenAI with the desired model
chat_model = ChatOpenAI(model_name="gpt-4o-mini")

# %% Pydantic model
class FeedbackResponse(BaseModel):
    rating: str = Field(..., description="Scoring in percentage")
    feedback: str = Field(..., description="Detailed feedback")
    revised_output: str = Field(..., description="An improved output describing the key events and significance of the American Civil War")

# %% Self-feedback function
def self_feedback(user_prompt: str, max_iterations: int = 5, target_rating: int = 90):
    content = ""
    feedback = ""
    
    for i in range(max_iterations):
        # Define the prompt based on iteration
        prompt_content = user_prompt if i == 0 else ""
        
        # Create a ChatPromptTemplate for system and user prompts
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
                Evaluate the input in terms of how well it addresses the original task of explaining the key events and significance of the American Civil War. Consider factors such as: Breadth and depth of context provided; Coverage of major events; Analysis of short-term and long-term impacts/consequences. If you identify any gaps or areas that need further elaboration: Return output as JSON with fields: 'rating': 'scoring in percentage', 'feedback': 'detailed feedback', 'revised_output': 'return an improved output describing the key events and significance of the American Civil War. Avoid special characters like apostrophes (') and double quotes'. 
                """),
            ("user", "<prompt_content>{prompt_content}</prompt_content><revised_output>{revised_output}</revised_output><feedback>{feedback}</feedback>")
        ])
        
        # Get response from the model
        chain = prompt_template | chat_model | JsonOutputParser(pydantic_object=FeedbackResponse)
        response = chain.invoke({"prompt_content": prompt_content, "revised_output": content, "feedback": feedback})
        
        
        try:
            
            # Extract rating
            rating_num = int(re.findall(r'\d+', response['rating'])[0])
            
            # Extract feedback and revised output
            feedback = response['feedback']
            content = response['revised_output']
            
            # Print iteration details
            print(f"i={i}, Prompt Content: {prompt_content}, Rating: {rating_num}, \nFeedback: {feedback}, \nRevised Output: {content}")
            
            # Return if rating meets or exceeds target
            if rating_num >= target_rating:
                return content
        except ValidationError as e:
            print("Validation Error:", e.json())
            return "Invalid response format."
    
    return content

#%% Test
user_prompt = "The American Civil War was a civil war in the United States between the north and south."
res = self_feedback(user_prompt=user_prompt, max_iterations=3, target_rating=95)
res
# %%
