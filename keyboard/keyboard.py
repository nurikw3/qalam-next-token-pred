from model.cache import load_or_build_model
from config import CSV_PATH, CACHE_PATH
from data.preprocess import normalize_token


class ChagataiKeyboard:
    def __init__(self):
        self.model = load_or_build_model(CSV_PATH, CACHE_PATH)

    def _parse(self, typed_text: str) -> tuple[list[str], str]:
        tokens = typed_text.strip().split()
        if not tokens:
            return [], ""

        ends_with_space = typed_text.endswith(" ")
        if ends_with_space:
            return [normalize_token(t) for t in tokens], ""
        else:
            return [normalize_token(t) for t in tokens[:-1]], tokens[-1]

    def suggest(self, text, top_k=5):
        context, prefix = self._parse(text)
        return [s.word for s in self.model.predict(context, prefix, top_k)]

    def suggest_full(self, text, top_k=5):
        context, prefix = self._parse(text)
        return self.model.predict(context, prefix, top_k)
