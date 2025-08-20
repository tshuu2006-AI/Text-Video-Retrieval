from pymilvus import Collection, connections
connections.connect(alias = "default", host = "localhost", port="19530")

collection = Collection("VIDEO_KeyFrames")

# Lấy các record có score > 50
results = collection.query(
    expr="video_id == 'L21_V031'",
    output_fields=["frame_id", "video_id", "objects"]
)

print(results)