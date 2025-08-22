from pymilvus import Collection, connections
from Searcher import search_by_semantic, search_by_objects, extract_noun_phrases_spacy, get_embeddings
from Preprocessing import load_embedding_model

connections.connect(alias = "default", host = "localhost", port = "19530")
collection = Collection("VIDEO_KeyFrames")
collection.load()
model, processor = load_embedding_model(use_fast=False)

text_input = input()
entities = extract_noun_phrases_spacy(text_input)
entity_embeddings = get_embeddings(model, processor, entities)
results = search_by_semantic(text_input, processor=processor, model=model, collection= collection, top_k=100)

for hits in results:
    best, scores = search_by_objects(hits, entity_embeddings, top_k=5)
    for i, hit in enumerate(best):
        print(f"Video Id {i}: {hit.entity.get('video_id')}")
        print(f"scores {i}: {hit.score}")
        print(f"total scores {i}: {scores[i]}")
        print(f"Frame Id {i}: {hit.entity.get('frame_id')}")




