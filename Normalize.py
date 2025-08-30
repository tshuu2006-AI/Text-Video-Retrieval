import numpy as np

from pymilvus import (
    connections, utility,
    Collection, FieldSchema, CollectionSchema, DataType
)

import Constants as C

# --- 1. THIẾT LẬP KẾT NỐI VÀ CÁC BIẾN ---
HOST = "localhost"
PORT = "19530"
connections.connect("default", host=HOST, port=PORT)

OLD_COLLECTION_NAME = "VIDEO_KeyFrames"
NEW_COLLECTION_NAME = "Keyframes_collection"
VECTOR_DIM = 768  # Thay đổi chiều vector cho đúng với dữ liệu của bạn
PK_FIELD = "id"
VECTOR_FIELD = "embedding"
BATCH_SIZE = 1000  # Lấy 1000 vector mỗi lần, có thể điều chỉnh

# --- 2. KẾT NỐI TỚI COLLECTION CŨ VÀ TẠO COLLECTION MỚI ---

# Kết nối tới collection cũ
if not utility.has_collection(OLD_COLLECTION_NAME):
    raise Exception(f"Collection '{OLD_COLLECTION_NAME}' không tồn tại!")
collection_cu = Collection(OLD_COLLECTION_NAME)
collection_cu.load()  # Tải collection cũ vào bộ nhớ để query nhanh hơn

# Xóa collection mới nếu đã tồn tại để làm lại từ đầu
if utility.has_collection(NEW_COLLECTION_NAME):
    utility.drop_collection(NEW_COLLECTION_NAME)

# Lấy schema từ collection cũ để tạo collection mới y hệt
schema_cu = collection_cu.schema
collection_moi = Collection(name=NEW_COLLECTION_NAME, schema=schema_cu)

print(f"Đã tạo collection mới '{NEW_COLLECTION_NAME}' thành công.")

# --- 3. QUÁ TRÌNH DI CHUYỂN DỮ LIỆU ---

offset = 0
total_vectors = 0
iterator = collection_cu.query_iterator(
    batch_size=BATCH_SIZE,
    output_fields=[C.VIDEO_ID_NAME, C.FRAME_ID_NAME, C.TIME_STAMPS_NAME, C.VECTOR_EMBEDDING_NAME, C.FRAME_PATH_NAME, C.VECTOR_OBJECT_NAME]
)
while True:
    results = iterator.next()
    # Nếu không còn kết quả, dừng vòng lặp
    if not results:
        break

    # Trích xuất dữ liệu từ kết quả query

    vectors = [res[C.VECTOR_EMBEDDING_NAME] for res in results]

    # Chuyển sang numpy array để chuẩn hóa
    vectors_np = np.array(vectors, dtype=np.float32)

    # *** BƯỚC QUAN TRỌNG: CHUẨN HÓA VECTOR ***
    norm = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    normalized_vectors = vectors_np / norm

    # Chuẩn bị dữ liệu để insert vào collection mới
    batch_to_insert = []
    for i, res in enumerate(results):
        record = {
            C.VIDEO_ID_NAME: res[C.VIDEO_ID_NAME],
            C.FRAME_ID_NAME: res[C.FRAME_ID_NAME],
            C.TIME_STAMPS_NAME: res[C.TIME_STAMPS_NAME],
            # ✨ SỬA LỖI: Sử dụng vector ĐÃ ĐƯỢC CHUẨN HÓA tại đúng vị trí
            C.VECTOR_EMBEDDING_NAME: normalized_vectors[i],
            C.FRAME_PATH_NAME: res[C.FRAME_PATH_NAME],
            C.VECTOR_OBJECT_NAME: res[C.VECTOR_OBJECT_NAME]
        }
        batch_to_insert.append(record)

    # Insert vào collection mới
    collection_moi.insert(batch_to_insert)

    total_vectors += len(results)
    print(f"Đã chuyển thành công {total_vectors} vector...")

    # Tăng offset để lấy batch tiếp theo
    offset += BATCH_SIZE

print("\nHoàn tất quá trình di chuyển dữ liệu!")

# --- 4. HOÀN TẤT ---

# Flush để đảm bảo dữ liệu được ghi xuống đĩa
collection_moi.flush()
print(f"Tổng số thực thể trong collection mới: {collection_moi.num_entities}")

# Bây giờ bạn có thể tạo index cho collection mới
print("Đang tạo index IP cho collection mới...")
index_params = {
    "metric_type": "IP",
    "index_type": "AUTOINDEX",
    "params": {}
}
collection_moi.create_index(
    field_name=C.VECTOR_EMBEDDING_NAME,
    index_params=index_params
)
print("Tạo index thành công!")

# (Tùy chọn) Xóa collection cũ để giải phóng dung lượng
# utility.drop_collection(OLD_COLLECTION_NAME)
# print(f"Đã xóa collection cũ '{OLD_COLLECTION_NAME}'.")