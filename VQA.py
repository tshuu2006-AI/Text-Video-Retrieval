# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True, cache_dir = "./Models")
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True, cache_dir = "./Models")
model.eval()





image = Image.open('Data/Videos_L23_a/frames/L23_V007/L23_V007_keyframe_0033.jpg').convert('RGB')
question = 'What are colors of the outfits that people in the frame are wearing, answer short in below 10 words'
msgs = [{'role': 'user', 'content': question}]

res = model.chat(
    image=image,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True, # if sampling=False, beam_search will be used by default
    temperature=0.7,
    # system_prompt='' # pass system_prompt if needed
)
print(res)
