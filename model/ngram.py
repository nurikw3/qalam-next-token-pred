from collections import Counter, defaultdict
from typing import NamedTuple
import math
from config import WEIGHTS, LAPLACE_ALPHA
from data.preprocess import normalize_token
import time


class Suggestion(NamedTuple):
    word: str
    score: float
    source: str


class NgramModel:
    """
    Строит unigram / bigram / trigram / 4-gram таблицы.
    Поддерживает сглаживание Лапласа и LRU-кэш предсказаний.
    """

    def __init__(self, sentences: list[list[str]]):
        self.vocab: set[str] = set()
        self.unigram: Counter[str] = Counter()
        self.bigram: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.trigram: defaultdict[tuple, Counter[str]] = defaultdict(Counter)
        self.fourgram: defaultdict[tuple, Counter[str]] = defaultdict(Counter)

        self._build(sentences)

        # Кэш: (context_tuple, prefix) → list[Suggestion]
        # Используем dict вручную, чтобы обойти ограничения lru_cache с изменяемыми типами
        self._cache: dict[tuple, list[Suggestion]] = {}

    # ── Построение ──

    def _build(self, sentences: list[list[str]]) -> None:
        t0 = time.perf_counter()
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

        V = len(self.vocab)
        elapsed = time.perf_counter() - t0
        print(
            f"[NgramModel] built in {elapsed:.2f}s | "
            f"vocab={V:,} | sents={len(sentences):,} | "
            f"2g={len(self.bigram):,} | 3g={len(self.trigram):,} | 4g={len(self.fourgram):,}"
        )

    # ── Сглаживание Лапласа ──

    def _laplace(
        self, counter: Counter[str], alpha: float = LAPLACE_ALPHA
    ) -> Counter[str]:
        """
        Добавляет alpha к каждому слову словаря.
        Возвращает новый Counter (оригинал не изменяется).
        """
        smoothed = Counter({w: counter.get(w, 0) + alpha for w in self.vocab})
        return smoothed

    # ── Предсказание ──

    def predict(
        self,
        context: list[str],
        prefix: str = "",
        top_k: int = 5,
    ) -> list[Suggestion]:
        """
        Предсказывает следующее слово / завершение текущего.

        Args:
            context: Предыдущие слова (нормализованные).
            prefix:  Уже введённые буквы текущего слова.
            top_k:   Количество результатов.

        Returns:
            Список Suggestion(word, score, source), убывающий по score.
        """
        ctx_norm = tuple(normalize_token(w) for w in context)
        pre_norm = normalize_token(prefix) if prefix else ""
        cache_key = (ctx_norm, pre_norm, top_k)

        if cache_key in self._cache:
            return self._cache[cache_key]

        raw: Counter[str] = Counter()
        source_map: dict[str, str] = {}  # слово → источник

        def _merge(counter: Counter[str], weight: int, label: str) -> None:
            for word, cnt in counter.items():
                if word not in source_map:
                    source_map[word] = label
                raw[word] += cnt * weight

        # 4-gram
        if len(ctx_norm) >= 3:
            ctx4 = ctx_norm[-3:]
            if ctx4 in self.fourgram:
                _merge(self._laplace(self.fourgram[ctx4]), WEIGHTS[4], "4gram")

        # Trigram
        if len(ctx_norm) >= 2:
            ctx3 = ctx_norm[-2:]
            if ctx3 in self.trigram:
                _merge(self._laplace(self.trigram[ctx3]), WEIGHTS[3], "trigram")

        # Bigram
        if len(ctx_norm) >= 1:
            ctx2 = ctx_norm[-1]
            if ctx2 in self.bigram:
                _merge(self._laplace(self.bigram[ctx2]), WEIGHTS[2], "bigram")

        # Unigram fallback
        if not raw:
            _merge(self.unigram, WEIGHTS[1], "unigram")
            for w in source_map:
                source_map[w] = "unigram"

        # Фильтрация по префиксу
        if pre_norm:
            raw = Counter({w: s for w, s in raw.items() if w.startswith(pre_norm)})
            if not raw:
                # n-gram ничего не дал — берём из словаря
                raw = Counter(
                    {
                        w: cnt
                        for w, cnt in self.unigram.items()
                        if w.startswith(pre_norm)
                    }
                )
                for w in raw:
                    source_map[w] = "prefix"

        if not raw:
            return []

        # Топ-K до softmax (экономим время)
        top_raw = raw.most_common(top_k * 3)

        # Softmax
        scores_raw = [s for _, s in top_raw]
        max_s = max(scores_raw)
        exp_s = [math.exp(s - max_s) for s in scores_raw]  # numerically stable
        total = sum(exp_s)
        softmax = [e / total for e in exp_s]

        results = [
            Suggestion(
                word=word,
                score=round(sm, 4),
                source=source_map.get(word, "?"),
            )
            for (word, _), sm in zip(top_raw, softmax)
        ]
        results.sort(key=lambda s: s.score, reverse=True)
        results = results[:top_k]

        self._cache[cache_key] = results
        return results
