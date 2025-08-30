import torch
import torch.nn.functional as F
import Constants as C
from scipy.optimize import linear_sum_assignment
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")


def extract_noun_phrases_spacy(text: str, nlp=nlp):
    """
    Trích xuất toàn bộ các cụm danh từ (noun phrases) từ văn bản.
    """
    doc = nlp(text)
    # Duyệt qua các cụm danh từ có sẵn trong doc.noun_chunks
    nouns = [token.text for token in doc if token.pos_ == "NOUN" or token.pos_ == "PROPN"]
    print(nouns)
    return nouns

def _load_embedding_layer(dir = "Models/object_embedding_weights.pt"):
    weights = torch.load(f = dir, weights_only=True)
    embedding_layer = torch.nn.Embedding.from_pretrained(embeddings=weights, freeze=True).to("cuda")
    return embedding_layer


def search_by_semantic(queries, processor, model, collection, top_k = 100):
    if isinstance(queries, str):
        queries = [queries]

    inputs = processor(text = queries, images= None, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        embeddings = F.normalize(model.get_text_features(**inputs), p = 2, dim = 1).cpu().numpy()

    results = collection.search(
        data=embeddings,  # embedding cần tìm
        anns_field=C.VECTOR_EMBEDDING_NAME,  # tên field vector trong schema
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,  # lấy top-k gần nhất
        output_fields=[C.VIDEO_ID_NAME,
                       C.FRAME_ID_NAME,
                       C.FRAME_PATH_NAME,
                       C.TIME_STAMPS_NAME,
                       C.VECTOR_OBJECT_NAME]  # trả về thêm các field khác
    )

    return results



def get_embeddings(model, processor, texts: list[str]):
    input_text = processor(text=texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        embeddings = F.normalize(model.get_text_features(**input_text), p = 2, dim = 1).cpu().numpy()
    return embeddings





def search_by_objects(hits, entity_embeddings, top_k, alpha = 0.95):
    scores = []
    embedding_layer = _load_embedding_layer()
    num_entities = len(entity_embeddings)
    for hit in hits:
        objects = torch.tensor(list(hit.entity.get(C.VECTOR_OBJECT_NAME)),dtype=torch.long, device="cuda")
        embedding_objects = embedding_layer(objects).cpu().numpy()
        cost_matrix = np.dot(entity_embeddings, embedding_objects.T)
        cost_matrix[cost_matrix < 0] = 0
        cost_matrix = -1 * cost_matrix

        a,b = cost_matrix.shape
        if a > b:
            pad = np.zeros(shape = (a, a - b))
            cost_matrix = np.concatenate([cost_matrix, pad], axis = 1)
        elif a < b:
            pad = np.zeros(shape = (b - a, b))
            cost_matrix = np.concatenate([cost_matrix, pad], axis=0)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        score = -1 * np.sum(cost_matrix[row_ind, col_ind]) * (1 - alpha) / num_entities + hit.score * alpha
        scores.append(score)
    scores = np.array(scores, dtype = np.float32)
    top_k_indices = np.argsort(scores)[::-1][:top_k]
    top_k_scores = scores[top_k_indices]
    best = [hits[i] for i in top_k_indices]
    return best, top_k_scores

def search_by_id(frame_ids, collection):
    s = str(frame_ids)
    results = collection.query(
        expr = f"frame_id in {s}",
        output_fields = ["frame_id", "objects"]
    )
    return results











