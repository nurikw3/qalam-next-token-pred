from model.cache import load_or_build_model
from config import CSV_PATH, CACHE_PATH


class ChagataiKeyboard:
    def __init__(self):
        self.model = load_or_build_model(CSV_PATH, CACHE_PATH)

    def _parse(self, typed_text):
        tokens = typed_text.strip().split()

        if not tokens:
            return [], ""

        if typed_text.endswith(" "):
            return tokens, ""
        else:
            return tokens[:-1], tokens[-1]

    def suggest(self, text, top_k=5):
        context, prefix = self._parse(text)
        return [s.word for s in self.model.predict(context, prefix, top_k)]

    def suggest_full(self, text, top_k=5):
        context, prefix = self._parse(text)
        return self.model.predict(context, prefix, top_k)
