from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

from Constants import BATCH_DIRS
from Preprocessing import load_embedding_model, extract_keyframes, load_Yolo
from tqdm import tqdm
import Constants as C

# Khởi tạo database
def db_initialization(name, alias, host, port):
    # Kết nối
    connections.connect(alias=alias,
                        host=host,
                         port=port
                        )


    # Nếu chưa có milvus_collection
    fields = [
        FieldSchema(name = C.VIDEO_ID_NAME,
                    dtype=DataType.VARCHAR,
                    max_length=255),
        FieldSchema(name= C.FRAME_ID_NAME,
                    dtype=DataType.VARCHAR,
                    max_length=255,
                    is_primary=True),
        FieldSchema(name= C.TIME_STAMPS_NAME,
                    dtype=DataType.FLOAT),
        FieldSchema(name = C.VECTOR_EMBEDDING_NAME,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=768),
        FieldSchema(name = C.FRAME_PATH_NAME,
                    dtype = DataType.VARCHAR,
                    max_length = 255),
        FieldSchema(name = C.VECTOR_OBJECT_NAME,
                    dtype = DataType.ARRAY,
                    dim = 50,
                    element_type=DataType.INT16,
                    max_capacity = 50,
                    nullable=True),
    ]

    schema = CollectionSchema(fields, description=C.PYMILVUS_DESCRIPTION)
    collection_name = name

    # Kiểm tra xem milvus_collection đã tồn tại chưa
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
            field_name=C.VECTOR_EMBEDDING_NAME,
            index_params=index_params
        )
    else:
        collection = Collection(collection_name)

        print(f"Connected to existing milvus_collection '{collection_name}'.")

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


if __name__ == "__main__":
    embedding_model, embedding_processor = load_embedding_model()
    detector = load_Yolo()
    milvus_collection = db_initialization(name=C.PYMILVUS_COLLECTION_NAME,
                                          alias="default",
                                          host="localhost",
                                          port="19530")
    print()

    outer_bar = tqdm(range(len(BATCH_DIRS)), desc="Processing batches", unit="batch", position=0)

    for i in outer_bar:
        batch_dir = C.BATCH_DIRS[i]
        batch_id = C.BATCH_IDS[i]
        vector_batch = extract_keyframes(model=embedding_model,
                                         processor=embedding_processor,
                                         detector = detector,
                                         collection=milvus_collection,
                                         batch_path=batch_dir,
                                         threshold=0.9,
                                         frame_interval = 0.75,
                                         outer_bar=outer_bar)


        add_records(milvus_collection, vector_batch)
    milvus_collection.compact()