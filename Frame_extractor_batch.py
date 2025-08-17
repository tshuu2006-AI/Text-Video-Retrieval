from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import torch
from transformers import CLIPModel, CLIPProcessor
import cv2
import torch.nn.functional as F
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import gc

# Khởi tạo hằng số
data_dir = "Data"
batch_ids = sorted(os.listdir(data_dir))
batch_dirs = [os.path.join(data_dir, batch_id) for batch_id in batch_ids]
cache_dir = "Models"
VECTOR_FIELD_NAME = "embedding"

# Khởi tạo database
def db_initialization(name, alias, host, port):
    # Kết nối
    connections.connect(alias=alias,
                        host=host,
                        port=port)


    # Nếu chưa có collection
    fields = [
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="frame_id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
        FieldSchema(name="time_stamp", dtype=DataType.FLOAT),
        FieldSchema(name=VECTOR_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=768)
    ]

    schema = CollectionSchema(fields, description="Video keyframe embeddings")
    collection_name = name

    # Kiểm tra xem collection đã tồn tại chưa
    from pymilvus import utility
    if not utility.has_collection(collection_name, using=alias):
        collection = Collection(collection_name, schema)
        print(f"Collection '{collection_name}' created.")

        index_params = {
            "metric_type": "IP",
            "index_type": "AUTOINDEX",
            "params": {}
        }

        collection.create_index(
            field_name=VECTOR_FIELD_NAME,
            index_params=index_params
        )
    else:
        collection = Collection(collection_name)

        print(f"Connected to existing collection '{collection_name}'.")

    collection.load()
    return collection


def add_records(collection, batch):
    if len(batch) == 0:
        return
    collection.insert(batch)
    collection.flush()


# Load mô hình và processor của CLIP
def load_model(model_id="laion/CLIP-ViT-L-14-laion2B-s32B-b82K", device="cuda", use_fast=False):
    model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir).to(device)
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir, use_fast=use_fast)
    return model, processor

def is_blurry(image, threshold=100):
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return lap_var < threshold

def keyframe_detection(embeddings, threshold=0.85):
    # Normalize
    normalize_embeddings = F.normalize(embeddings, p=2, dim=1)
    cosine_similarity = torch.matmul(normalize_embeddings, normalize_embeddings.T)

    triangle_mask = torch.tril(torch.ones_like(cosine_similarity, dtype=torch.bool), diagonal= -1)
    keep_mask = ((cosine_similarity > threshold) & triangle_mask).any(dim = 1)
    keep_mask = ~keep_mask

    return normalize_embeddings[keep_mask], keep_mask

def embedding_comparison(collection, embeddings, threshold):

    results = collection.search(
        data = embeddings,
        anns_field = VECTOR_FIELD_NAME,
        param = {"metric_type": "IP", "params": {"nprobe": 10}},
        limit = 1
    )
    duplicate_idx = []
    for hits in results:
        if len(hits) > 0 and hits[0].score> threshold:
            duplicate_idx.append(True)
        else:
            duplicate_idx.append(False)

    return torch.tensor(duplicate_idx, dtype=torch.bool)

def frame_processing(model, processor, collection, frame_buffer, time_stamps, threshold, device = "cuda"):
    inputs = processor(images=frame_buffer, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)

    keyframe_embeddings, keep_mask = keyframe_detection(embeddings, threshold)
    time_stamps = [time_stamps[i] for i in range(len(frame_buffer)) if keep_mask[i]]
    keyframes = [frame_buffer[i] for i in range(len(frame_buffer)) if keep_mask[i]]

    if len(keyframes) == 0:
        return np.array([]), [], []
    keyframe_embeddings = keyframe_embeddings.detach().cpu().numpy()
    duplicate_mask = embedding_comparison(collection, keyframe_embeddings, threshold=threshold)
    unduplicate_mask = ~duplicate_mask

    # Cập nhật time_stamps và keyframes dựa trên keep_mask
    keyframe_embeddings = keyframe_embeddings[unduplicate_mask]
    time_stamps = [time_stamps[i] for i in range(len(unduplicate_mask)) if unduplicate_mask[i]]
    keyframes = [keyframes[i] for i in range(len(unduplicate_mask)) if unduplicate_mask[i]]

    return keyframe_embeddings, keyframes, time_stamps


def write_frames(video_id_without_ext, frame_dir, keyframes, frame_count):
    frame_ids = []
    for i in range(len(keyframes)):
        frame_id = f"{video_id_without_ext}_keyframe_{frame_count:04d}.jpg"
        save_path = os.path.join(frame_dir, frame_id)
        frame_ids.append(frame_id)
        keyframes[i].save(save_path)
        frame_count += 1

    return frame_ids, frame_count


def extract_keyframes(model, processor, collection, batch_path, outer_bar, threshold=0.9,
                      frame_interval=1, device="cuda"):
    video_folder = os.path.join(batch_path, "video")
    root_frame_dir = os.path.join(batch_path, "frames")
    os.makedirs(root_frame_dir, exist_ok=True)
    video_ids = os.listdir(video_folder)

    for idx, video_id in enumerate(video_ids, start=1):
        outer_bar.set_postfix(video=f"{idx}/{len(video_ids)} {video_id}")

        video_id_without_ext, ext = os.path.splitext(video_id)
        frame_dir = os.path.join(root_frame_dir, video_id_without_ext)
        os.makedirs(frame_dir, exist_ok=True)

        frame_buffer = []
        frame_count = 0
        time_stamps = []
        current_time = 0
        cap = cv2.VideoCapture(os.path.join(video_folder, video_id))

        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
            ret, frame = cap.read()
            if not ret:
                break

            if is_blurry(frame):
                continue

            frame_buffer.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            time_stamps.append(current_time)
            current_time += frame_interval

        keyframe_embeddings, keyframes, time_stamps = frame_processing(model, processor, collection, frame_buffer, time_stamps, threshold)
        frame_ids, frame_count = write_frames(video_id_without_ext = video_id_without_ext, frame_dir = frame_dir, keyframes = keyframes, frame_count = frame_count)

        batch = [{"video_id": video_id,
                  "frame_id": frame_ids[i],
                  "time_stamp": time_stamps[i],
                  "embedding": keyframe_embeddings[i]}
                for i in range(len(frame_ids))]

        add_records(collection, batch)
        cap.release()

        #Clean the GPU
        torch.cuda.empty_cache()
        del frame_buffer, time_stamps, keyframe_embeddings, keyframes
        gc.collect()


if __name__ == "__main__":
    model, processor = load_model()
    collection = db_initialization(name="VIDEO_KeyFrames",
                                   alias="default",
                                   host="localhost",
                                   port="19530")
    print()

    outer_bar = tqdm(range(len(batch_dirs)), desc="Processing batches", unit="batch", position=0)

    for i in outer_bar:
        batch_dir = batch_dirs[i]
        batch_id = batch_ids[i]
        extract_keyframes(model=model, processor=processor, collection=collection, batch_path=batch_dir,
                            frame_interval=1, outer_bar=outer_bar)