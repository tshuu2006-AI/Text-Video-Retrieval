from pymilvus import Collection, connections
from Frame_extractor_minibatch import load_model
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from translate_vi_en import translate_vi_to_en   # import hàm dịch


def plot_4_images(img_paths, titles=None):
    """
    Hiển thị 4 ảnh trong lưới 2x2 bằng matplotlib

    Args:
        img_paths (list): danh sách 4 đường dẫn ảnh
        titles (list): danh sách 4 tiêu đề (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()  # flatten về 1D để dễ duyệt

    for i, path in enumerate(img_paths):
        img = mpimg.imread(path)
        axes[i].imshow(img)
        axes[i].axis("off")  # tắt trục
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=12)

    plt.tight_layout()
    plt.show()


connections.connect(alias = "default", host = "localhost", port = "19530")
collection = Collection("VIDEO_KeyFrames")
collection.load()
model, processor = load_model(use_fast=False)
text_input = [input()]

text_en = translate_vi_to_en(text_input)
text_input = [text_en]

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


