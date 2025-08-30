from pymilvus import connections, Collection, utility

connections.connect(alias = "default", host = "localhost", port = "19530")
collection = Collection("VIDEO_KeyFrames")
utility.drop_collection("VIDEO_KeyFrames")

