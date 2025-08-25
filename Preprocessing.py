import numpy as np
import cv2
import os
import gc
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor
import Constants as C
import torch
import json
import torch.nn.functional as F
import shutil

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

def _clear_temp(directory = "temp"):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    os.makedirs(f"{directory}/Batch_embeddings")
    os.makedirs(f"{directory}/batch_images")
    os.makedirs(f"{directory}/Embeddings")
    os.makedirs(f"{directory}/images")


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



def Yolo_object_detection(detector, frames, batch_size = 4, threshold = 0.5):
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

    for r in results:
        class_ids = r.boxes.cls.cpu().numpy()
        confidence = r.boxes.conf.cpu().numpy()
        objects = class_ids[confidence > threshold].astype(int).tolist()
        class_results.append(objects)
    torch.cuda.empty_cache()
    return class_results

def _save_image_batchs(video_id, directory = "temp/images", batch_directory = "temp/batch_images"):

    list_dirs = sorted(os.listdir(directory))
    full_temp_images = []
    for dir in list_dirs:
        image_batch_dir = os.path.join(directory, dir)
        image_batch = np.load(image_batch_dir)
        full_temp_images.append(image_batch)
        os.remove(image_batch_dir)

    if not os.path.exists(batch_directory):
        os.makedirs(batch_directory)
    if not full_temp_images:
        return
    full_temp_images = np.concatenate(full_temp_images, axis = 0)
    save_path = os.path.join(batch_directory, f"{video_id}.npy")
    np.save(save_path, full_temp_images)


def load_embedding_model(model_id = C.EMBEDDING_MODEL_ID, device="cuda", use_fast=False):
    embedding_model = CLIPModel.from_pretrained(model_id, cache_dir=C.CACHE_DIR).to(device)
    embedding_processor = CLIPProcessor.from_pretrained(model_id, cache_dir=C.CACHE_DIR, use_fast=use_fast)
    return embedding_model, embedding_processor

def write_frames(video_ids_without_ext, root_frame_dir, keyframes):
    """
        Detect the embeddings that differ from the embeddings in the database

            Args:
                video_ids_without_ext: The video ids without extensions (list[String])
                root_frame_dir: the directory of the root frame folder (String)
                keyframes: The frames that will be added (list[numpy ndarray])

            Returns:
                frame_ids: The ids of the frame (list[String])
        """
    frame_ids = []
    frame_dir = root_frame_dir
    frame_paths = []
    frame_count = 0
    prev_id = None
    for i in range(len(video_ids_without_ext)):
        cur_id = video_ids_without_ext[i]
        if cur_id != prev_id:
            frame_dir = os.path.join(root_frame_dir, cur_id)
            os.makedirs(frame_dir, exist_ok=True)
            prev_id = cur_id
            frame_count = 0

        frame_id = f"{cur_id}_keyframe_{frame_count:04d}.jpg"

        save_path = os.path.join(frame_dir, frame_id)
        frame_paths.append(save_path)

        frame_ids.append(frame_id)
        cv2.imwrite(save_path, cv2.cvtColor(keyframes[i], cv2.COLOR_RGB2BGR))

        frame_count += 1

    return frame_ids, frame_paths


def frame_processing(embedding_model,
                     embedding_processor,
                     collection,
                     frame_buffer,
                     time_stamps,
                     threshold,
                     device = "cuda"):
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
def is_blurry(image, blur_threshold=90, histogram_threshold = 0.5) -> bool:

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

def load_temp_embeddings(directory = f"{C.TEMP_DIR}/Embeddings"):
    list_dir = sorted(os.listdir(directory))
    final_embeddings = []
    for dir in list_dir:
        path = os.path.join(directory, dir)
        embedding = torch.load(path)
        final_embeddings.append(embedding)
        os.remove(path)
    final_embeddings = torch.concat(final_embeddings)
    return final_embeddings

def _load_images(directory = f"{C.TEMP_DIR}/batch_images"):
    list_dir = sorted(os.listdir(directory))
    keyframes = []
    for dir in list_dir:
        path = os.path.join(directory, dir)
        images = list(np.load(path))
        keyframes.extend(images)
        os.remove(path)
    return keyframes

def _save_video_embeddings(video_id,
                           directory = "temp/Embeddings",
                           destination = f"temp/Batch_embeddings"):

    os.makedirs(destination,
                exist_ok=True)

    list_dirs = sorted(os.listdir(directory))
    video_embeddings = []
    for dir in list_dirs:
        path = os.path.join(directory, dir)
        embeddings = torch.load(path, weights_only=True)
        video_embeddings.append(embeddings)
        os.remove(path)
    if not video_embeddings:
        return
    video_embeddings = torch.concat(video_embeddings, dim = 0)
    save_path = os.path.join(destination, f"{video_id}.pt")
    torch.save(video_embeddings, save_path)

def _load_final_embeddings(directory = "temp/Batch_embeddings"):
    list_dirs = os.listdir(directory)
    final_embeddings = []

    for video_id in list_dirs:
        path = os.path.join(directory, video_id)
        embeddings = torch.load(path, weights_only=True)
        final_embeddings.append(embeddings)
        os.remove(path)

    final_embeddings = torch.concat(final_embeddings, dim = 0)
    return final_embeddings


def _write_metadata(path,
                    video_id_without_ext,
                    time_stamps):
    record = {
        "video_id": video_id_without_ext,
        "time_stamps": time_stamps
    }
    with open(path, "a") as file:
        file.write(json.dumps(record) + "\n")


def _load_metadata(path):

    video_ids_without_ext = []
    final_time_stamps = []

    with open(path, "r") as file:
        for line in file:
            metadata = json.loads(line)
            video_id, time_stamps = metadata["video_id"], metadata["time_stamps"]
            video_ids_without_ext.extend([video_id for _ in range(len(time_stamps))])

            final_time_stamps.extend(time_stamps)

    return video_ids_without_ext, final_time_stamps

def extract_keyframes(model,
                      processor,
                      detector,
                      collection,
                      batch_path,
                      outer_bar,
                      batch_size=64,
                      threshold=0.90,
                      frame_interval=0.5,
                      device="cuda",
                      video_batch_size = 8):

    video_folder = os.path.join(batch_path, "video")
    root_frame_dir = os.path.join(batch_path, "frames")
    os.makedirs(root_frame_dir, exist_ok=True)
    video_ids = os.listdir(video_folder)
    frame_dirs = []
    start = 0

    while start < len(video_ids):
        _clear_temp()
        for idx, video_id in enumerate(video_ids[start: start + video_batch_size], start = start + 1):
            frame_count = 0
            outer_bar.set_postfix(video=f"{idx}/{len(video_ids)} {video_id}")
            video_id_without_ext, ext = os.path.splitext(video_id)

            frame_dir = os.path.join(root_frame_dir, video_id_without_ext)
            frame_dirs.append(frame_dir)
            os.makedirs(frame_dir,
                        exist_ok=True)

            final_timestamps = []
            frame_buffer = []
            time_stamps = []
            cap = cv2.VideoCapture(os.path.join(video_folder, video_id))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = 0
            embeddings_idx = 0
            while True:

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    break

                if is_blurry(frame):
                    frame_idx += round(fps * frame_interval)
                    continue

                frame_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamp = frame_idx / fps
                time_stamps.append(timestamp)
                frame_idx += round(fps * frame_interval)

                # Khi buffer đầy, xử lý batch
                if len(frame_buffer) == batch_size:

                    keyframe_embeddings, keyframes, time_stamps = frame_processing(embedding_model=model,
                                                                                   embedding_processor= processor,
                                                                                   collection= collection,
                                                                                   frame_buffer= frame_buffer,
                                                                                   time_stamps=time_stamps,
                                                                                   threshold = threshold,
                                                                                   device = device)
                    frame_count += len(keyframe_embeddings)

                    if not keyframes:
                        frame_buffer.clear()
                        time_stamps.clear()
                        continue

                    torch.save(keyframe_embeddings, f"{C.TEMP_DIR}/Embeddings/embeddings_{embeddings_idx}.pt")
                    np.save(f"{C.TEMP_DIR}/images/frames_{embeddings_idx}.npy", np.stack(keyframes))
                    final_timestamps.extend(time_stamps)

                    del keyframe_embeddings, keyframes
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Clear buffer
                    frame_buffer.clear()
                    time_stamps.clear()
                    embeddings_idx += 1


            # Xử lý phần còn lại trong buffer
            if len(frame_buffer) > 0:
                keyframe_embeddings, keyframes, time_stamps = frame_processing(model, processor, collection, frame_buffer, time_stamps, threshold, device=device)
                frame_count += len(keyframe_embeddings)

                if keyframes:
                    torch.save(keyframe_embeddings, f"{C.TEMP_DIR}/Embeddings/embeddings_{embeddings_idx}.pt")
                    np.save(f"{C.TEMP_DIR}/images/frames_{embeddings_idx}.npy", np.stack(keyframes))

                    final_timestamps.extend(time_stamps)

                    del keyframes, time_stamps
                    torch.cuda.empty_cache()
                    gc.collect()
                    cap.release()

            _write_metadata(path = "temp/metadata.jsonl",
                           video_id_without_ext=video_id_without_ext,
                           time_stamps=final_timestamps)

            _save_video_embeddings(video_id=video_id_without_ext,
                                   directory="temp/Embeddings",
                                   destination="temp/Batch_embeddings")

            _save_image_batchs(video_id_without_ext,
                               directory = "temp/images",
                               batch_directory=f"temp/batch_images")

        final_embeddings = _load_final_embeddings()
        final_distinct_embeddings, final_keep_mask = keyframe_detection(final_embeddings,
                                                                        threshold = 0.90)
        final_keyframes = _load_images()

        final_distinct_embeddings = final_distinct_embeddings.cpu().numpy().astype(np.float32)
        video_ids_without_ext, final_timestamps = _load_metadata(path = "temp/metadata.jsonl")
        video_ids_without_ext = [video_ids_without_ext[i] for i in range(len(final_keep_mask)) if final_keep_mask[i]]
        final_keyframes = [final_keyframes[i] for i in range(len(final_keep_mask)) if final_keep_mask[i]]
        final_timestamps = [final_timestamps[i] for i in range(len(final_keep_mask)) if final_keep_mask[i]]
        objects = Yolo_object_detection(detector, final_keyframes)

        frame_ids, frame_paths = write_frames(video_ids_without_ext = video_ids_without_ext,
                                              root_frame_dir=root_frame_dir,
                                              keyframes=final_keyframes)

        batch = [{C.VIDEO_ID_NAME: video_ids_without_ext[i],
                  C.FRAME_ID_NAME: frame_ids[i],
                  C.TIME_STAMPS_NAME: final_timestamps[i],
                  C.VECTOR_EMBEDDING_NAME: final_distinct_embeddings[i],
                  C.FRAME_PATH_NAME: frame_paths[i],
                  C.VECTOR_OBJECT_NAME: objects[i]}
                 for i in range(len(frame_ids))]


        del video_ids_without_ext, final_keyframes, final_timestamps, objects
        torch.cuda.empty_cache()
        gc.collect()
        yield batch
        start += video_batch_size



