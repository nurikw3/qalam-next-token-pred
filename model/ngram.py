from collections import Counter, defaultdict
from typing import NamedTuple
import math
from config import WEIGHTS
from data.preprocess import normalize_token
import time


class Suggestion(NamedTuple):
    word: str
    score: float
    source: str


class NgramModel:
    def __init__(self, sentences: list[list[str]]):
        self.vocab: set[str] = set()
        self.unigram: Counter[str] = Counter()
        self.bigram: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.trigram: defaultdict[tuple, Counter[str]] = defaultdict(Counter)
        self.fourgram: defaultdict[tuple, Counter[str]] = defaultdict(Counter)

        # Kneser-Ney: сколько уникальных контекстов предшествует слову
        self.kn_continuation: Counter[str] = Counter()
        # сколько уникальных слов следует за контекстом (для лямбды)
        self.kn_bigram_types: Counter[str] = Counter()

        self._cache: dict[tuple, list[Suggestion]] = {}
        self._build(sentences)

    # ── Построение ────────────────────────────────────────────────────────────

    def _build(self, sentences: list[list[str]]) -> None:
        t0 = time.perf_counter()

        # Считаем обычные n-gram counts
        for tokens in sentences:
            for tok in tokens:
                self.unigram[tok] += 1
                self.vocab.add(tok)

            for i in range(len(tokens) - 1):
                self.bigram[tokens[i]][tokens[i + 1]] += 1

            for i in range(len(tokens) - 2):
                self.trigram[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1

            for i in range(len(tokens) - 3):
                self.fourgram[(tokens[i], tokens[i + 1], tokens[i + 2])][
                    tokens[i + 3]
                ] += 1

        # Kneser-Ney continuation counts:
        # kn_continuation[w] = кол-во уникальных слов v таких что bigram (v, w) существует
        for left, rights in self.bigram.items():
            for right in rights:
                self.kn_continuation[right] += 1

        # kn_bigram_types[w] = кол-во уникальных слов u таких что bigram (w, u) существует
        for left, rights in self.bigram.items():
            self.kn_bigram_types[left] = len(rights)

        self._kn_total = sum(self.kn_continuation.values()) or 1

        elapsed = time.perf_counter() - t0
        print(
            f"[NgramModel] built in {elapsed:.2f}s | "
            f"vocab={len(self.vocab):,} | sents — | "
            f"2g={len(self.bigram):,} | 3g={len(self.trigram):,} | 4g={len(self.fourgram):,}"
        )

    # ── Kneser-Ney вероятности ────────────────────────────────────────────────

    def _kn_unigram(self, word: str) -> float:
        """P_kn(word) — continuation probability (базовый уровень KN)."""
        return self.kn_continuation.get(word, 0) / self._kn_total

    def _kn_bigram(self, context: str, word: str, d: float = 0.75) -> float:
        """P_kn(word | context) с интерполяцией на unigram."""
        ctx_count = self.bigram.get(context)
        if not ctx_count:
            return self._kn_unigram(word)

        total = sum(ctx_count.values())
        n_types = self.kn_bigram_types.get(context, 0)

        discounted = max(ctx_count.get(word, 0) - d, 0) / total
        lam = (d * n_types) / total
        return discounted + lam * self._kn_unigram(word)

    def _kn_trigram(self, ctx: tuple, word: str, d: float = 0.75) -> float:
        """P_kn(word | ctx[0], ctx[1]) с интерполяцией на bigram."""
        tri_count = self.trigram.get(ctx)
        if not tri_count:
            return self._kn_bigram(ctx[-1], word, d)

        total = sum(tri_count.values())
        n_types = len(tri_count)

        discounted = max(tri_count.get(word, 0) - d, 0) / total
        lam = (d * n_types) / total
        return discounted + lam * self._kn_bigram(ctx[-1], word, d)

    def _kn_fourgram(self, ctx: tuple, word: str, d: float = 0.75) -> float:
        """P_kn(word | ctx[0], ctx[1], ctx[2]) с интерполяцией на trigram."""
        four_count = self.fourgram.get(ctx)
        if not four_count:
            return self._kn_trigram(ctx[-2:], word, d)

        total = sum(four_count.values())
        n_types = len(four_count)

        discounted = max(four_count.get(word, 0) - d, 0) / total
        lam = (d * n_types) / total
        return discounted + lam * self._kn_trigram(ctx[-2:], word, d)

    # ── Предсказание ──────────────────────────────────────────────────────────

    def predict(
        self,
        context: list[str],
        prefix: str = "",
        top_k: int = 5,
    ) -> list[Suggestion]:
        ctx_norm = tuple(normalize_token(w) for w in context)
        pre_norm = normalize_token(prefix) if prefix else ""
        cache_key = (ctx_norm, pre_norm, top_k)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Выбираем кандидатов — фильтруем по префиксу если есть
        if pre_norm:
            candidates = [w for w in self.vocab if w.startswith(pre_norm)]
        else:
            candidates = list(self.vocab)

        if not candidates:
            return []

        # Считаем KN вероятность для каждого кандидата
        scores: list[tuple[str, float, str]] = []

        for word in candidates:
            if len(ctx_norm) >= 3:
                p = self._kn_fourgram(ctx_norm[-3:], word)
                src = "4gram"
            elif len(ctx_norm) == 2:
                p = self._kn_trigram(ctx_norm[-2:], word)
                src = "trigram"
            elif len(ctx_norm) == 1:
                p = self._kn_bigram(ctx_norm[-1], word)
                src = "bigram"
            else:
                p = self._kn_unigram(word)
                src = "unigram"
            scores.append((word, p, src))

        # Сортируем и берём top_k * 3 для softmax
        scores.sort(key=lambda x: -x[1])
        top = scores[: top_k * 3]

        # Softmax для нормализации в [0, 1]
        vals = [p for _, p, _ in top]
        max_v = max(vals)
        exp_v = [math.exp(v - max_v) for v in vals]
        total = sum(exp_v)
        softmax = [e / total for e in exp_v]

        results = [
            Suggestion(word=w, score=round(sm, 4), source=src)
            for (w, _, src), sm in zip(top, softmax)
        ][:top_k]

        self._cache[cache_key] = results
        return results
