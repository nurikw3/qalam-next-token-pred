import re

CSV_PATH = "data/merged.txt"
CACHE_PATH = "data/.ngram_cache.pkl"

PUNCT_RE = re.compile(r"[^\w\-ʿʾ]", re.UNICODE)

WEIGHTS = {4: 5, 3: 3, 2: 2, 1: 1}
LAPLACE_ALPHA = 0.3
