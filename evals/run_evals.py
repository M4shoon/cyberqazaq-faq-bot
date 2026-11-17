"""FAQ-бот evals. Іске қосу (жобаның түбірінде):  python evals/run_evals.py

Екі түрлі кейс:
  ANSWER — құжатта жауап бар: бот жауап беруі және дұрыс бөлімді көрсетуі керек.
  REFUSE — құжатта жауап жоқ: бот «білмеймін» деуі керек (ойдан жауап құрмауы).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot import answer, build_kb  # noqa: E402

# (түрі, сұрақ, күтілетін тақырыптағы сөз)
CASES = [
    ("ANSWER", "Kaspi арқылы қалай төлеймін?",                         "Kaspi арқылы"),
    ("ANSWER", "Halyk картасымен төлеуге бола ма?",                    "Halyk"),
    ("ANSWER", "Қолма-қол төлем бар ма?",                              "Қолма-қол"),
    ("ANSWER", "Бөліп төлеу қанша айға дейін?",                        "Бөліп төлеу (рассрочка)"),
    ("ANSWER", "500 000 теңгелік сетті 12 айға бөлсем ай сайын қанша төлеймін?", "жоспары"),
    ("ANSWER", "Тауарды қайтаруға бола ма?",                           "қайтаруға"),
    ("ANSWER", "Кепілдік мерзімі қанша?",                              "Кепілдік мерзімі"),
    ("ANSWER", "Жеткізу қанша күн алады?",                             "Жеткізу"),
    ("ANSWER", "Сатпаевта жеткізу тегін бе?",                          "Жеткізу бағасы"),
    ("ANSWER", "Дүкен қай уақытта жұмыс істейді?",                     "Байланыс деректері"),
    ("ANSWER", "WhatsApp нөміріңіз қандай?",                           "Байланыс деректері"),
    ("ANSWER", "Компьютер жинап бересіздер ме?",                       "Компьютер жинау"),
    ("ANSWER", "Дүкен қашан ашылды?",                                  "Компания туралы"),
    ("ANSWER", "Жеткізу кезінде тауар зақымдалса не болады?",          "Жеткізу кепілдігі"),
    ("REFUSE", "RTX 5090 бағасы қанша?",                               None),
    ("REFUSE", "Ауа райы қандай?",                                     None),
    ("REFUSE", "Python қалай үйренемін?",                              None),
    ("REFUSE", "Қазақстанның президенті кім?",                         None),
]

# ҚАЙШЫЛЫҚ кейстері: сайттың әр беті әр түрлі жауап береді. Бот екеуін де (кемінде 2 файлдан)
# көрсетуі керек, тек біреуін таңдап қайшылықты жасырмауы керек.
CONFLICTS = [
    ("CONFLICT", "Жеткізу қанша уақыт алады?",       None),   # 1–3 жұмыс күні  vs  7–15 күн
    ("CONFLICT", "Бөліп төлеу қанша айға дейін?",    None),   # 24 айға дейін   vs  3, 6, 12 ай
    # Жалған дабыл болмауы керек: беттер бір-біріне сәйкес келеді
    ("NOCONFLICT", "Кепілдік қанша?",                None),   # 12–36 ай барлық жерде бірдей
    ("NOCONFLICT", "Кепілдік мерзімі қандай?",       None),
    ("NOCONFLICT", "Тауарды қайтаруға бола ма?",     None),   # 14 күн барлық жерде бірдей
]

# HELD-OUT: бот пен құжатты баптап болған СОҢ жазылған, баптауға қатыспаған сұрақтар.
# Бұл нәтиже бот жаңа сұрақтарға қаншалықты жалпыланатынын шынайы көрсетеді.
# Ереже: осы кейстердің нәтижесін көргеннен кейін оларға қарап кодты бейімдемеңіз.
HELDOUT = [
    ("ANSWER", "Алматыға жеткізесіздер ме?",                           "Жеткізу (Біз туралы"),
    ("ANSWER", "Айгерім кім?",                                         "Қолдау және команда"),
    ("ANSWER", "Ақау шықса не істеймін?",                              "Кепілдік мерзімі"),
    ("ANSWER", "Homebank арқылы төлесем бола ма?",                     "Homebank"),
    ("ANSWER", "Курьер трекинг нөмірін береді ме?",                    "Жеткізу"),
    ("REFUSE", "Ноутбук сатасыздар ма?",                               None),
    ("REFUSE", "Қандай курстарыңыз бар?",                              None),
]


def run(cases, kb, title):
    print(f"--- {title} ---")
    ok_total, top1 = 0, 0
    n_answer = sum(1 for c in cases if c[0] == "ANSWER")
    for kind, q, expect in cases:
        a = answer(q, kb)
        headings = [h for _, h, _ in a.sources]
        if kind == "ANSWER":
            ok = a.found and any(expect.lower() in h.lower() for h in headings)
            if a.found and headings and expect.lower() in headings[0].lower():
                top1 += 1
            detail = "" if ok else f"→ табылған: {headings or 'білмеймін'}"
        elif kind == "CONFLICT":      # қайшылық табылуы керек
            files = {s for s, _, _ in a.sources}
            ok = a.found and len(files) >= 2 and bool(a.conflict)
            detail = "" if ok else f"→ қайшылық анықталмады (дереккөздер: {sorted(files)})"
        elif kind == "NOCONFLICT":    # жалған дабыл болмауы керек
            ok = a.found and not a.conflict
            detail = "" if ok else f"→ жалған дабыл: {a.conflict}"
        else:
            ok = not a.found
            detail = "" if ok else f"→ ойдан жауап берді: {headings}"
        ok_total += ok
        print(f"{'PASS' if ok else 'FAIL'}  {kind}  {q}  {detail}")
    print(f"Нәтиже: {ok_total}/{len(cases)}  (жауап бар сұрақтарда 1-орында: {top1}/{n_answer})\n")
    return ok_total == len(cases)


def main():
    kb = build_kb()
    a = run(CASES, kb, "Негізгі кейстер (баптауға қатысқан)")
    b = run(HELDOUT, kb, "HELD-OUT (баптауға қатыспаған)")
    c = run(CONFLICTS, kb, "Қайшылықтар: табылуы керек / жалған дабыл болмауы керек")
    sys.exit(0 if a and b and c else 1)


if __name__ == "__main__":
    main()
