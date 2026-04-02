from config import PUNCT_RE


def normalize_token(token: str) -> str:
    return PUNCT_RE.sub("", token).lower()
