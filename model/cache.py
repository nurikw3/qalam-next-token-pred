import os
import pickle

from data.loader import load_corpus
from model.ngram import NgramModel


def load_or_build_model(csv_path, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        if os.path.getmtime(cache_path) > os.path.getmtime(csv_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    sentences = load_corpus(csv_path)
    model = NgramModel(sentences)

    with open(cache_path, "wb") as f:
        pickle.dump(model, f)

    return model
