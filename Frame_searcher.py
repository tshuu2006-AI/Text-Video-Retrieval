from pymilvus import Collection, connections
from Frame_extractor_minibatch import load_model
import torch
connections.connect(alias = "default", host = "localhost", port = "19530")
collection = Collection("VIDEO_KeyFrames")
collection.load()
model, processor = load_model(use_fast=False)
text_input = [input()]
inputs = processor(text = text_input, images= None, return_tensors="pt", padding=True).to("cuda")

with torch.no_grad():
    embeddings = model.get_text_features(**inputs).norm(p = 2, dim = 1, keepdim = True)

results = collection.search(
    data=embeddings,     # embedding cần tìm
    anns_field="embedding",  # tên field vector trong schema
    param={"metric_type": "IP", "params": {"nprobe": 10}},
    limit=5,                 # lấy top-5 gần nhất
    output_fields=["video_id", "frame_id", "time_stamp"]  # trả về thêm các field khác
)

print(results)

