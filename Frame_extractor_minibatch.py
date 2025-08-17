import os

os.environ['OPENCV_LOG_LEVEL'] = "SILENT"
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = "-8"

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import torch
from transformers import CLIPModel, CLIPProcessor
import cv2
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np

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


def add_records(collection, batch) -> None:
    """
    Add records to Milvus Database.
        Args:
            collection: The collection that contains records (pymilvus.Collection)
            batch: Batch of records (List[Dictionary])

        Returns:
            None
    """
    if len(batch) == 0:
        return
    collection.insert(batch)
    collection.flush()


# Load mô hình và processor của CLIP
def load_model(model_id="laion/CLIP-ViT-L-14-laion2B-s32B-b82K", device="cuda", use_fast=False):
    model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir).to(device)
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir, use_fast=use_fast)
    return model, processor

#Nhận biết ảnh nhòe
def is_blurry(image, threshold=100) -> bool:

    """
    Define an image is blurred or not
        Args:
            image: OpenCV.image (numpy.ndarray)
            threshold: Laplacian threshold (int)

        Returns:
            Is blur or not (bool)
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return lap_var < threshold

def keyframe_detection(embeddings, threshold=0.8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    detect the keyframes of the mini batch of current embeddings
        Args:
            embeddings: Tensor[float32] shape (N, D)
            threshold: cosine similarity threshold

        Returns:
            keyframes: Tensor[float32] shape (M, D)
            keep_mask: Tensor[bool] shape (N,)
    """

    # Normalize
    normalize_embeddings = F.normalize(embeddings, p=2, dim=1)
    cosine_similarity = torch.matmul(normalize_embeddings, normalize_embeddings.T)

    triangle_mask = torch.tril(torch.ones_like(cosine_similarity, dtype=torch.bool), diagonal= -1)
    keep_mask = ((cosine_similarity > threshold) & triangle_mask).any(dim = 1)
    keep_mask = ~keep_mask

    return normalize_embeddings[keep_mask], keep_mask

def embedding_comparison(collection, embeddings, threshold):
    """
        Detect the embeddings that differ from the embeddings in the database

            Args:
                collection: The collection that contains embeddings (pymilvus.Collection)
                embeddings: Tensor[float32] shape (N, D)
                threshold: cosine similarity threshold

            Returns:
                duplicate_idx: determine which embeddings are similar (Tensor[bool], shape = (N,) )
    """

    results = collection.search(
        data = embeddings,
        anns_field = VECTOR_FIELD_NAME,
        param = {"metric_type": "IP", "params": {"nprobe": 10}},
        limit = 1
    )

    duplicate_idx = []
    for hits in results:

        if hits[0].score> threshold:
            duplicate_idx.append(True)
        else:
            duplicate_idx.append(False)

    duplicate_idx = torch.tensor(duplicate_idx, dtype = torch.bool)
    return duplicate_idx

def frame_processing(model, processor, collection, frame_buffer, time_stamps, threshold, device = "cuda"):
    """
        Detect the embeddings that differ from the embeddings in the database

            Args:
                model: The model that generate semantic embeddings (huggingface model)
                processor: The processor that preprocess the images (huggingface processor)
                collection: The collection that contains embeddings (pymilvus.Collection)
                frame_buffer: The buffer for raw images (list[PIL.Image])
                time_stamps: The time stamps for corresponding frame (list[float])
                threshold: The cosine similarity threshold (float)
                device: the device for pytorch (String)

            Returns:
                keyframe_embeddings: The distinct embeddings (torch.Tensor, shape = (N, D))
                keyframes: The distinct frames (list[PIL.Image])
                time_stamps: The time stamps corresponding with the distinct keyframes (list[int])
    """

    inputs = processor(images=frame_buffer, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)

    keyframe_embeddings, keep_mask = keyframe_detection(embeddings, threshold)
    time_stamps = [time_stamps[i] for i in range(len(keep_mask)) if keep_mask[i]]
    keyframes = [frame_buffer[i] for i in range(len(keep_mask)) if keep_mask[i]]

    if len(keyframes) == 0:
        return np.array([]), [], []
    keyframe_embeddings = keyframe_embeddings.cpu().numpy()
    duplicate_mask = torch.zeros(size = (len(keyframe_embeddings), ), dtype = torch.bool)

    if collection.num_entities > 0:
        duplicate_mask = embedding_comparison(collection, keyframe_embeddings, threshold=threshold)
    if duplicate_mask.shape[0] == 0:
        return np.array([]), [], []
    unduplicate_mask = ~duplicate_mask

    # Cập nhật time_stamps và keyframes dựa trên keep_mask
    keyframe_embeddings = keyframe_embeddings[unduplicate_mask]
    time_stamps = [time_stamps[i] for i in range(len(unduplicate_mask)) if unduplicate_mask[i]]
    keyframes = [keyframes[i] for i in range(len(unduplicate_mask)) if unduplicate_mask[i]]

    return keyframe_embeddings, keyframes, time_stamps


def write_frames(video_id_without_ext, frame_dir, keyframes, frame_count):
    """
        Detect the embeddings that differ from the embeddings in the database

            Args:
                video_id_without_ext: The video ids without extensions (String)
                frame_dir: the directories of the frames that will be added (String)
                keyframes: The frames that will be added (list[PIL.Image])
                frame_count: The current orders of the frames before adding (int)


            Returns:
                frame_ids: The ids of the frame (list[String])
                frame_count: The current orders of the frames after adding (int)
        """
    frame_ids = []
    for i in range(len(keyframes)):
        frame_id = f"{video_id_without_ext}_keyframe_{frame_count:04d}.jpg"
        save_path = os.path.join(frame_dir, frame_id)
        frame_ids.append(frame_id)
        keyframes[i].save(save_path)
        frame_count += 1

    return frame_ids, frame_count


def extract_keyframes(model, processor, collection, batch_path, outer_bar, batch_size=64, threshold=0.8,
                      frame_interval=1, device="cuda"):
    video_folder = os.path.join(batch_path, "video")
    root_frame_dir = os.path.join          (batch_path, "frames")
    os.makedirs(root_frame_dir, exist_ok=True)
    video_ids = os.listdir(video_folder)
    batch = []
    for idx, video_id in enumerate(video_ids, start=1):
        outer_bar.set_postfix(video=f"{idx}/{len(video_ids)} {video_id}")

        video_id_without_ext, ext = os.path.splitext(video_id)
        frame_dir = os.path.join(root_frame_dir, video_id_without_ext)
        os.makedirs(frame_dir, exist_ok=True)

        frame_buffer = []
        frame_count = 0
        time_stamps = []
        cap = cv2.VideoCapture(os.path.join(video_folder, video_id))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0

        while True:

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            if is_blurry(frame):
                frame_idx += round(fps * frame_interval)
                continue

            frame_buffer.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            timestamp = frame_idx / fps
            time_stamps.append(timestamp)
            frame_idx += round(fps * frame_interval)

            # Khi buffer đầy, xử lý batch
            if len(frame_buffer) == batch_size:

                keyframe_embeddings, keyframes, time_stamps = frame_processing(model, processor, collection, frame_buffer, time_stamps, threshold, device = device)
                frame_ids, frame_count = write_frames(video_id_without_ext = video_id_without_ext, frame_dir = frame_dir, keyframes = keyframes, frame_count = frame_count)

                batch.extend([{"video_id": video_id,
                          "frame_id": frame_ids[i],
                          "time_stamp": time_stamps[i],
                          "embedding": keyframe_embeddings[i]}
                        for i in range(len(frame_ids))])

                torch.cuda.empty_cache()

                # Clear buffer
                frame_buffer = []
                time_stamps = []


        # Xử lý phần còn lại trong buffer
        if len(frame_buffer) > 0:
            keyframe_embeddings, keyframes, time_stamps = frame_processing(model, processor, collection, frame_buffer, time_stamps, threshold, device=device)

            frame_ids, frame_count = write_frames(video_id_without_ext=video_id_without_ext, frame_dir=frame_dir,
                                     keyframes=keyframes, frame_count = frame_count)

            batch.extend([{"video_id": video_id,
                      "frame_id": frame_ids[i],
                      "time_stamp": time_stamps[i],
                      "embedding": keyframe_embeddings[i]}
                     for i in range(len(frame_ids))])

        final_embeddings = torch.stack([item["embedding"] for item in batch])
        final_distinct_embeddings, final_keep_mask = keyframe_detection(final_embeddings, threshold = 0.9)

        batch = [batch[i] for i in range(len(batch)) if final_keep_mask[i]]

        torch.cuda.empty_cache()
        cap.release()

    return batch


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
        vector_batch = extract_keyframes(model=model,
                          processor=processor,
                          collection=collection,
                          batch_path=batch_dir,
                          threshold=0.9,
                          frame_interval=1,
                          outer_bar=outer_bar)

        add_records(collection, vector_batch)
    collection.compact()