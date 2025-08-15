from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import torch
from transformers import CLIPModel, CLIPProcessor
import cv2
import torch.nn.functional as F
from tqdm import tqdm
import os
from PIL import Image

#Khởi tạo hằng số
data_dir = "Data"
batch_ids = sorted(os.listdir(data_dir))
batch_dirs = [os.path.join(data_dir, batch_id) for batch_id in batch_ids]

#Khởi tạo database
def db_initialization(name, alias, uri, token):
    # Kết nối
    connections.connect(alias = alias,
                        uri = uri,
                        token= token)

    # Nếu chưa có collection
    fields = [
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="frame_id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
        FieldSchema(name ="time_stamp", dtype = DataType.FLOAT),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]

    schema = CollectionSchema(fields, description="Video keyframe embeddings")
    collection_name = name
    collection = Collection(collection_name, schema)

    return collection

def add_records(collection, video_id, frame_id, time_stamp, embedding):
  if len(embedding) == 0:
    return
  records = [[video_id] * len(frame_id), frame_id, time_stamp, embedding.cpu().tolist()]
  collection.insert(records)
  collection.flush()

#Load mô hình và processor của CLIP
def load_model(model_id = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", device="cuda"):
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    return model, processor

def keyframe_detection(embeddings, threshold = 0.8, last_keyframe_emb = None):

    if last_keyframe_emb is not None:
        embeddings = torch.cat((last_keyframe_emb.unsqueeze(0), embeddings), dim = 0)

    num_embeddings = len(embeddings)
    keep_mask = torch.ones(num_embeddings, dtype=torch.bool)

    if last_keyframe_emb is not None:
        keep_mask[0] = False

    #Normalize
    normalize_embeddings = F.normalize(embeddings, p=2, dim=1)

    cosine_similarity = torch.matmul(normalize_embeddings, normalize_embeddings.T)

    for i in range(1, num_embeddings):
        if (cosine_similarity[i, :i] > threshold).any():
            keep_mask[i] = False

    if last_keyframe_emb is not None:
        return embeddings[keep_mask == True], keep_mask[1:]

    return embeddings[keep_mask == True], keep_mask

def extract_keyframes(model, processor, collection, batch_path, batch_id, outer_bar, batch_size=32, threshold=0.9, frame_interval = 1, device = "cuda"):

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
      frame_ids = []
      last_keyframe_emb = None
      frame_count = 0
      time_stamps = []
      current_time = 0
      cap = cv2.VideoCapture(os.path.join(video_folder, video_id))

      while True:
          cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
          ret, frame = cap.read()

          if not ret:
              break


          frame_buffer.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
          time_stamps.append(current_time)
          current_time += frame_interval

          # Khi buffer đầy, xử lý batch
          if len(frame_buffer) == batch_size:
              inputs = processor(images=frame_buffer, return_tensors="pt", padding=True).to(device)
              with torch.no_grad():
                  embeddings = model.get_image_features(**inputs)

              keyframe_embeddings, keep_mask = keyframe_detection(embeddings, threshold, last_keyframe_emb)
              time_stamps = [time_stamps[i] for i in range(len(frame_buffer)) if keep_mask[i]]
              keyframes = [frame_buffer[i] for i in range(len(frame_buffer)) if keep_mask[i]]

              for i in range(len(keyframe_embeddings)):
                  frame_id = f"{video_id_without_ext}_keyframe_{frame_count:04d}.jpg"
                  frame_ids.append(frame_id)
                  save_path = os.path.join(frame_dir, frame_id)
                  keyframes[i].save(save_path)
                  frame_count += 1

              if len(keyframe_embeddings) > 1:
                last_keyframe_emb = keyframe_embeddings[-1]

              else:
                last_keyframe_emb = None
              add_records(collection, video_id, frame_ids, time_stamps, keyframe_embeddings)

              # Clear buffer
              frame_buffer = []
              frame_ids = []
              time_stamps = []


      # Xử lý phần còn lại
      if len(frame_buffer) > 0:
          inputs = processor(images=frame_buffer, return_tensors="pt", padding=True).to("cuda")
          with torch.no_grad():
              embeddings = model.get_image_features(**inputs)

          keyframe_embeddings, keep_mask = keyframe_detection(embeddings, threshold, last_keyframe_emb)
          time_stamps = [time_stamps[i] for i in range(len(frame_buffer)) if keep_mask[i]]
          keyframes = [frame_buffer[i] for i in range(len(frame_buffer)) if keep_mask[i]]

          for i in range(len(keyframe_embeddings)):
              frame_id = f"{video_id_without_ext}_keyframe_{frame_count:04d}.jpg"
              frame_ids.append(frame_id)
              save_path = os.path.join(frame_dir, frame_id)
              keyframes[i].save(save_path)
              frame_count += 1

          add_records(collection, video_id, frame_ids, time_stamps, keyframe_embeddings)

      cap.release()


if __name__ == "__main__":

    model, processor = load_model()
    outer_bar = tqdm(range(len(batch_dirs)), desc="Processing batches", unit="batch", position=0)
    collection = db_initialization(name = "VIDEO_KeyFrames", alias = , uri = , token = )

    for i in outer_bar:
      batch_dir = batch_dirs[i]
      batch_id = batch_ids[i]
      extract_keyframes(model = model, processor = processor, collection = collection, batch_path = batch_dir, batch_id = batch_id, frame_interval=1, outer_bar = outer_bar)