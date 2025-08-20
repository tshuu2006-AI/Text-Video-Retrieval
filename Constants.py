import os

#Field Names
VECTOR_EMBEDDING_NAME = "embedding"
VECTOR_OBJECT_NAME = "objects"
VECTOR_NUM_OBJECTS_NAME = "num_objects"
VIDEO_ID_NAME = "video_id"
TIME_STAMPS_NAME = "time_stamp"
FRAME_ID_NAME = "frame_id"
FRAME_PATH_NAME = "frame_path"
NUM_OBJECTS = "num_objects"

#Directories
CACHE_DIR = "Models"
DATA_DIR = "Data"
BATCH_IDS = sorted(os.listdir(DATA_DIR))
BATCH_DIRS = [os.path.join(DATA_DIR, batch_id) for batch_id in BATCH_IDS]

#COLLECTION and DATABASE names
PYMILVUS_COLLECTION_NAME = "VIDEO_KeyFrames"
PYMILVUS_DESCRIPTION = "Video keyframe embeddings"

#OTHERS
AVERAGE_STD = 127.5

#MODEL_IDS
EMBEDDING_MODEL_ID = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
OBJECT_DETECTING_MODEL_DIR = "Models/faster-rcnn-inception-resnet-v2-tensorflow1-faster-rcnn-openimages-v4-inception-resnet-v2-v1"
