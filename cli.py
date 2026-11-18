"""Іске қосу:  python cli.py                (сұрақ-жауап режимі)
              python cli.py "сұрақ"         (бір сұрақ)
              python cli.py --check-llm     (Gemini кілті мен моделін тексеру)
              python cli.py --list-models   (қолжетімді Gemini модельдерінің тізімі)"""
import sys

import llm
from bot import build_kb, respond


def show(q, kb):
    a = respond(q, kb)
    print("\n" + a.text)
    if a.llm_error:
        print(f"\n(Тіл моделі қолжетімсіз: {a.llm_error}. Жоғарыда құжаттан үзінді көрсетілді.)")
    if a.found:
        print("\nДереккөз: " + "; ".join(f"{s} → {h}" for s, h, _ in a.sources)
              + ("   [жауапты Gemini құрастырды]" if a.generated else ""))
        if a.conflict:
            print(f"Ескерту: сайттың беттері бір-біріне қайшы келеді ({a.conflict}). Менеджермен нақтылаңыз.")
    print()


if __name__ == "__main__":
    llm.load_env()
    if sys.argv[1:] == ["--check-llm"]:
        try:
            print("Gemini жауабы:", llm.ping())
        except llm.LLMError as e:
            print("Қате:", e)
        sys.exit(0)
    if sys.argv[1:] == ["--list-models"]:
        try:
            names = llm.list_models()
            print("Қолжетімді модельдер (GEMINI_MODEL= үшін атауды осыдан алыңыз):")
            for n in names:
                print("  " + n + ("   <- жылдам/арзан, бастау үшін қолайлы" if "flash" in n and "image" not in n else ""))
        except llm.LLMError as e:
            print("Қате:", e)
        sys.exit(0)
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
