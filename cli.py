"""Іске қосу:  python cli.py            (сұрақ-жауап режимі)
              python cli.py "сұрақ"     (бір сұрақ)"""
import sys

from bot import answer, build_kb


def show(q, kb):
    a = answer(q, kb)
    print("\n" + a.text)
    if a.found:
        print("\nДереккөз: " + "; ".join(f"{s} → {h}" for s, h, _ in a.sources))
        if a.conflict:
            print(f"Ескерту: сайттың беттері бір-біріне қайшы келеді ({a.conflict}). Менеджермен нақтылаңыз.")
    print()


if __name__ == "__main__":
    kb = build_kb()
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]), kb)
    else:
        print("CyberQazaq FAQ-бот. Шығу: Enter (бос жол) немесе Ctrl+C.")
        while True:
            q = input("Сұрақ: ").strip()
            if not q:
                break
            show(q, kb)
