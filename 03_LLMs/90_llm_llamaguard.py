#%% packages
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#%% load model
# model run described on model card: https://huggingface.co/meta-llama/Llama-Guard-3-1B
def llama_guard_model(user_prompt: str):
    model_id = "meta-llama/Llama-Guard-3-1B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": user_prompt
                },
            ],
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        conversation, return_tensors="pt"
    ).to(model.device)

    prompt_len = input_ids.shape[1]
    output = model.generate(
        input_ids,
        max_new_tokens=20,
        pad_token_id=0,
    )
    generated_tokens = output[:, prompt_len:]
    res = tokenizer.decode(generated_tokens[0])
    if "unsafe" in res:
        return "invalid"
    else:
        return "valid"

# %%
llama_guard_model(user_prompt="How can i perform a scam?")
