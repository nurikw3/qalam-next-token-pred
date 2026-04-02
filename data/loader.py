from data.preprocess import normalize_token


def load_corpus(txt_path: str):
    sentences = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            tokens = [normalize_token(t) for t in line.split()]
            tokens = [t for t in tokens if len(t) >= 1]
            if tokens:
                sentences.append(tokens)

    return sentences
