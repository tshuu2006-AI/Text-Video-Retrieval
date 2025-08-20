from pymilvus import connections, db

# Kết nối tới Milvus server
connections.connect(
    alias="default",
    host="localhost",   # hoặc IP của Milvus server
    port="19530"
)

from pymilvus import db

# Tạo database tên "video_db"
print(db.list_database())