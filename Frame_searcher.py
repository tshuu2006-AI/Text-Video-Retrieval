from pymilvus import Collection, connections

import Constants as C
import matplotlib.pyplot as plt
from Searcher import search_by_semantic, search_by_objects, extract_noun_phrases_spacy, get_embeddings
from Preprocessing import load_embedding_model

import matplotlib.pyplot as plt
import numpy as np
import os

# Create a directory to store dummy images
def plot_images(image_paths):


    # Now, plot the images from the paths
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle('25 Images from Paths', fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < len(image_paths):
            # Read the image
            img = plt.imread(image_paths[i])
            # Display the image
            ax.imshow(img)
            # Add a title with the image number
            ax.set_title(f'Image {i + 1}')
        # Turn off the axes
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

connections.connect(alias = "default", host = "localhost", port = "19530")
collection = Collection(C.PYMILVUS_COLLECTION_NAME)
collection.load()
model, processor = load_embedding_model(use_fast=False)

text_input = input()
entities = extract_noun_phrases_spacy(text_input)
entity_embeddings = get_embeddings(model, processor, entities)
results = search_by_semantic(text_input, processor=processor, model=model, collection= collection, top_k=100)

paths = []
for hits in results:
    best, scores = search_by_objects(hits, entity_embeddings, top_k=25)
    for i, hit in enumerate(best):
        paths.append(hit.entity.get(C.FRAME_PATH_NAME))
        print(f"Video Id {i}: {hit.entity.get('video_id')}")
        print(f"scores {i}: {hit.score}")
        print(f"total scores {i}: {scores[i]}")
        print(f"Frame Id {i}: {hit.entity.get('frame_id')}")
        print(f"time stamp {i}: {hit.entity.get('time_stamp')}")
        print()

plot_images(paths)



