def _bar(score: float, width: int = 20) -> str:
    return "█" * round(score * width)


def print_suggestions(label, suggestions):
    print(f"\n  ▸ {label}")
    for s in suggestions:
        print(f"[{s.source}] {s.word} {s.score}")
