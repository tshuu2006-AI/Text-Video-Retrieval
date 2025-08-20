from pymilvus import Collection, connections
from Frame_extractor_minibatch import load_embedding_model
import torch
import torch.nn.functional as F
import Constants as C



connections.connect(alias = "default", host = "localhost", port = "19530")
collection = Collection(C.PYMILVUS_COLLECTION_NAME)
collection.load()
model, processor = load_embedding_model(use_fast=False)
text_input = [input()]
inputs = processor(text = text_input, images= None, return_tensors="pt", padding=True).to("cuda")

with torch.no_grad():
    embeddings = F.normalize(model.get_text_features(**inputs), p = 2, dim = 1).cpu().numpy()

print(embeddings)

results = collection.search(
    data=embeddings,     # embedding cần tìm
    anns_field="embedding",  # tên field vector trong schema
    param={"metric_type": "IP", "params": {"nprobe": 10}},
    limit=4,                 # lấy top-5 gần nhất
    output_fields=["video_id", "frame_id", "time_stamp"]  # trả về thêm các field khác
)

i = 0
dirs = []
frame_dir = []
for hits in results:
    for hit in hits:
        video_id = hit.entity.get("video_id")
        frame_id = hit.entity.get("frame_id")
        time_stamp = hit.entity.get("time_stamp")

        print(f"video_id {i}: {video_id}")
        print(f"frame_id {i}: {frame_id}")
        print(f"time_stamp {i}: {time_stamp}")
        print()
        i += 1


