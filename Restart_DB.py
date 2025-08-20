from pymilvus import connections, Collection, utility


connections.connect(alias="default",
                    host="localhost",
                    port="19530")
alias = "default"
name = "VIDEO_KeyFrames"
if utility.has_collection(name, using=alias):
    print(f"Collection '{name}' tồn tại. Đang chuẩn bị xóa...")
    collection = Collection("VIDEO_KeyFrames")
    print(collection.num_entities)

    # 3. Thực hiện lệnh xóa collection
    utility.drop_collection(name, using=alias)

    print(f"✔️ Collection '{name}' đã được xóa thành công.")
else:
    print(f"⚠️ Collection '{name}' không tồn tại, không có gì để xóa.")