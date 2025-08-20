import numpy as np
import cv2
import os
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor
import Constants as C
import torch
import torch.nn.functional as F
from PIL import Image


def load_Yolo(model_dir = "Models/yolov8x-oiv7.pt"):
    """
    Load the pre-trained object detecting model
        Args:
            model_dir: the directory of the model

        Returns:
            detector: the object detecting model
    """
    detector = YOLO(model_dir)
    return detector


# def faster_rcnn_object_detection(detector, frames, batch_size = 1, threshold = 0.45):
#     if not frames:
#         return []
#     frames = tf.convert_to_tensor(np.array(frames) / 255, dtype = tf.float32)
#     results = []
#     for i in range(0, len(frames), batch_size):
#         batch = frames[i:i + batch_size]
#         output = detector(batch)
#         scores = output["detection_scores"]
#         labels = output["detection_class_labels"]
#         result = labels[scores > threshold]
#         results.extend(result.numpy().astype(int).tolist())
#
#     return results



def Yolo_object_detection(detector, frames, batch_size = 4, threshold = 0.35, max_padding = 50, pad_value = -1):
    """
        Detect th e objects in the images
            Args:
                detector: Yolo object detector
                frames: list of frames
                batch_size: frames batch size (int)
                threshold: the threshold of confidence (float)

            Returns:
                class_results: The classes in each of the images
    """

    if len(frames) == 0:
        return []
    results = []
    for i in range(0, len(frames), batch_size):
        result = detector(frames[i: i + batch_size], verbose = False)
        results.extend(result)
    class_results = []
    num_objects = []
    for r in results:
        class_ids = r.boxes.cls.cpu().numpy()
        confidence = r.boxes.conf.cpu().numpy()
        objects = class_ids[confidence > threshold].astype(np.float32)
        if len(objects) > max_padding:
            num_objects.append(max_padding)
            padded_object = objects[:max_padding]
        else:
            num_objects.append(len(objects))
            right_pad = max_padding - len(objects)
            padded_object = np.pad(objects, (0, right_pad), mode = "constant", constant_values=-1)
        class_results.append(padded_object)
    torch.cuda.empty_cache()
    return class_results, num_objects


def load_embedding_model(model_id = C.EMBEDDING_MODEL_ID, device="cuda", use_fast=False):
    embedding_model = CLIPModel.from_pretrained(model_id, cache_dir=C.CACHE_DIR).to(device)
    embedding_processor = CLIPProcessor.from_pretrained(model_id, cache_dir=C.CACHE_DIR, use_fast=use_fast)
    return embedding_model, embedding_processor

def write_frames(video_id_without_ext, root_frame_dir, keyframes):
    """
        Detect the embeddings that differ from the embeddings in the database

            Args:
                video_id_without_ext: The video ids without extensions (list[String])
                root_frame_dir: the directory of the root frame folder (String)
                keyframes: The frames that will be added (list[PIL.Image])

            Returns:
                frame_ids: The ids of the frame (list[String])
        """
    frame_ids = []
    frame_dir = root_frame_dir
    frame_paths = []
    frame_count = 0
    prev_id = None
    for i in range(len(video_id_without_ext)):
        cur_id = video_id_without_ext[i]
        if cur_id != prev_id:
            frame_dir = os.path.join(root_frame_dir, cur_id)
            os.makedirs(frame_dir, exist_ok=True)
            prev_id = cur_id
            frame_count = 0

        frame_id = f"{cur_id}_keyframe_{frame_count:04d}.jpg"

        save_path = os.path.join(frame_dir, frame_id)
        frame_paths.append(save_path)

        frame_ids.append(frame_id)
        keyframes[i].save(save_path)

        frame_count += 1

    return frame_ids, frame_paths


def frame_processing(embedding_model, embedding_processor, collection, frame_buffer, time_stamps, threshold, device = "cuda"):
    """
        Detect the embeddings that differ from the embeddings in the database

            Args:
                embedding_model: The model that generate semantic embeddings (huggingface model)
                embedding_processor: The processor that preprocess the images (huggingface processor)
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

    inputs = embedding_processor(images=frame_buffer, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model.get_image_features(**inputs)

    keyframe_embeddings, keep_mask = keyframe_detection(embeddings, threshold)
    time_stamps = [time_stamps[i] for i in range(len(keep_mask)) if keep_mask[i]]
    keyframes = [frame_buffer[i] for i in range(len(keep_mask)) if keep_mask[i]]

    if len(keyframes) == 0:
        return torch.tensor([]), [], []
    duplicate_mask = torch.zeros(size = (len(keyframe_embeddings), ), dtype = torch.bool)

    if collection.num_entities > 0:
        duplicate_mask = embedding_comparison(collection, keyframe_embeddings, threshold=threshold)
    if duplicate_mask.shape[0] == 0:
        return torch.tensor([]), [], []
    unduplicate_mask = ~duplicate_mask

    # Update time_stamps and keyframes base on keep_mask
    keyframe_embeddings = keyframe_embeddings[unduplicate_mask]
    time_stamps = [time_stamps[i] for i in range(len(unduplicate_mask)) if unduplicate_mask[i]]
    keyframes = [keyframes[i] for i in range(len(unduplicate_mask)) if unduplicate_mask[i]]

    return keyframe_embeddings, keyframes, time_stamps


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
        data = embeddings.cpu().tolist(),
        anns_field = C.VECTOR_EMBEDDING_NAME,
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

def keyframe_detection(embeddings, threshold=0.95) -> tuple[torch.Tensor, torch.Tensor]:
    """
    detect the keyframes of the mini batch of current embeddings
        Args:
            embeddings: Tensor[float32] shape (N, D)
            threshold: cosine similarity threshold

        Returns:
            keyframes: Tensor[float32] shape (M, D)
            keep_mask: Tensor[bool] shape (N)
    """

    # Normalize
    normalize_embeddings = F.normalize(embeddings, p=2, dim=1)
    cosine_similarity = torch.matmul(normalize_embeddings, normalize_embeddings.T)

    triangle_mask = torch.tril(torch.ones_like(cosine_similarity, dtype=torch.bool), diagonal= -1)
    keep_mask = ((cosine_similarity > threshold) & triangle_mask).any(dim = 1)
    keep_mask = ~keep_mask

    return normalize_embeddings[keep_mask], keep_mask

#Detect blurred image
def is_blurry(image, blur_threshold=100, histogram_threshold = 0.55) -> bool:

    """
    Define an image is blurred or not
        Args:
            image: OpenCV.image (numpy.ndarray)
            blur_threshold: Laplacian threshold (float)
            histogram_threshold: Histogram threshold (float)

        Returns:
            Is blur or not (bool)
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(image, cv2.CV_64F).var()

    std_dev = np.std(image).astype(np.float32)
    normalized_std_dev = std_dev / C.AVERAGE_STD
    return lap_var < blur_threshold or normalized_std_dev < histogram_threshold

def extract_keyframes(model, processor, detector, collection, batch_path, outer_bar, batch_size=32, threshold=0.95,
                      frame_interval=0.5, device="cuda"):
    video_folder = os.path.join(batch_path, "video")
    root_frame_dir = os.path.join(batch_path, "frames")
    os.makedirs(root_frame_dir, exist_ok=True)
    video_ids = os.listdir(video_folder)
    frame_dirs = []
    video_ids_without_ext = []
    final_embeddings = None
    final_keyframes = []
    final_timestamps = []

    for idx, video_id in enumerate(video_ids, start=1):
        frame_count = 0
        outer_bar.set_postfix(video=f"{idx}/{len(video_ids)} {video_id}")

        video_id_without_ext, ext = os.path.splitext(video_id)

        frame_dir = os.path.join(root_frame_dir, video_id_without_ext)
        frame_dirs.append(frame_dir)
        os.makedirs(frame_dir, exist_ok=True)

        frame_buffer = []
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
                frame_count += len(keyframe_embeddings)

                if final_embeddings is None:
                    final_embeddings = keyframe_embeddings
                else:
                    final_embeddings = torch.cat([final_embeddings, keyframe_embeddings])
                final_keyframes.extend(keyframes)
                final_timestamps.extend(time_stamps)

                torch.cuda.empty_cache()

                # Clear buffer
                frame_buffer = []
                time_stamps = []


        # Xử lý phần còn lại trong buffer
        if len(frame_buffer) > 0:
            keyframe_embeddings, keyframes, time_stamps = frame_processing(model, processor, collection, frame_buffer, time_stamps, threshold, device=device)
            frame_count += len(keyframe_embeddings)

            if final_embeddings is None:
                final_embeddings = keyframe_embeddings
            else:
                final_embeddings = torch.cat([final_embeddings, keyframe_embeddings])

            final_keyframes.extend(keyframes)
            final_timestamps.extend(time_stamps)


        video_ids_without_ext.extend([video_id_without_ext for _ in range(frame_count)])
        torch.cuda.empty_cache()
        cap.release()

    final_distinct_embeddings, final_keep_mask = keyframe_detection(final_embeddings, threshold = 0.95)
    final_timestamps = [final_timestamps[i] for i in range(len(final_keep_mask)) if final_keep_mask[i]]
    final_keyframes = [final_keyframes[i] for i in range(len(final_keep_mask)) if final_keep_mask[i]]
    objects, num_objects = Yolo_object_detection(detector, final_keyframes)

    video_ids_without_ext = [video_ids_without_ext[i] for i in range(len(final_keep_mask)) if final_keep_mask[i]]
    final_distinct_embeddings = final_distinct_embeddings.cpu().numpy().astype(np.float32)
    frame_ids, frame_paths = write_frames(video_id_without_ext = video_ids_without_ext, root_frame_dir = root_frame_dir, keyframes = final_keyframes)

    batch = [{C.VIDEO_ID_NAME: video_ids_without_ext[i],
              C.FRAME_ID_NAME: frame_ids[i],
              C.TIME_STAMPS_NAME: final_timestamps[i],
              C.VECTOR_EMBEDDING_NAME: final_distinct_embeddings[i],
              C.FRAME_PATH_NAME: frame_paths[i],
              C.VECTOR_OBJECT_NAME: objects[i],
              C.NUM_OBJECTS: num_objects[i]}
             for i in range(len(frame_ids))]

    return batch



