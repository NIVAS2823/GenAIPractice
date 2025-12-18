import numpy as np
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

try:
    _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    MODEL_LOADED = True
except Exception as e:
    logger.exception("Failed to load SentenceTransformer model.")
    _model = None
    MODEL_LOADED = False


def get_model():
    if _model is None:
        raise RuntimeError("Embedding model is not loaded")
    return _model


def compute_embeddings(sentence1: str, sentence2: str):
    model = get_model()
    embeddings = model.encode([sentence1, sentence2])
    return np.array(embeddings[0]), np.array(embeddings[1])


def l2_norm(vec):
    return float(np.linalg.norm(vec))


def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def dot_product(a, b):
    return float(np.dot(a, b))


def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))


def manhattan_distance(a, b):
    return float(np.sum(np.abs(a - b)))
