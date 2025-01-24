#%%
import torch
from diffusers import AmusedPipeline
#%%
pipe = AmusedPipeline.from_pretrained(
    "amused/amused-256", variant="fp16", torch_dtype=torch.float16
)
pipe.vqvae.to(torch.float32)  # vqvae is producing nans in fp16
#%%

prompt = "dog"
image = pipe(prompt, generator=torch.Generator().manual_seed(8)).images[0]
image.save('text2image_256.png')
# %%
