import numpy as np
from scipy.spatial.distance import cosine

def cosine_sim(a, b):
    return 1 - cosine(a, b)

def average_embeddings(emb_list):
    return np.mean(emb_list, axis=0)
