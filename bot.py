"""FAQ-бот: сұраққа құжаттардан жауап табады. Жауап құжатта жоқ болса, «білмеймін» дейді."""
import re
from dataclasses import dataclass, field
from pathlib import Path

import llm
from retrieval import KnowledgeBase, load_chunks

DOCS_DIR = Path(__file__).parent / "docs"
MIN_COVERAGE = 0.5        # жауап беру үшін: сұрақтың кемінде жартысы табылған бөлікте болуы керек
EXTRA_MIN_COVERAGE = 0.7  # қосымша бөлікті көрсету үшін талап қатаңырақ (артық бөлімдер аз шығу үшін)
FALLBACK = ("Бұл сұраққа жауап құжаттарда табылмады. Менеджерге жазыңыз: "
            "WhatsApp wa.me/77765007502 немесе +7 776 500 75 02.")


@dataclass
class Answer:
    found: bool
    text: str
    sources: list = field(default_factory=list)  # [(файл, тақырып, score)]
    conflict: str = ""                           # беттердегі сандар сәйкес келмесе, түсініктеме
    chunks: list = field(default_factory=list)   # LLM-ге беретін құжат бөліктері
    generated: bool = False                      # жауапты тіл моделі құрастырды ма
    llm_error: str = ""                          # модель шақырылып, сәтсіз болса, себебі


# «3 күн», «1–3 жұмыс күні», «24 айға», «1-күн»: сан + өлшем бірлігі (күн/ай)
UNIT_RE = re.compile(r"(\d+)(?:\s*[–-]\s*(\d+))?\s*-?\s*(?:жұмыс\s+)?(күн|ай)")


def find_conflict(hits):
    """Әр түрлі файлдар бір өлшем бірлігі (күн/ай) бойынша әр түрлі ең үлкен санды айтса, қайшылық деп санайды."""
    per_file = {}  # (бірлік, файл) -> ең үлкен сан
    for h in hits:
        for a, b, unit in UNIT_RE.findall(h.chunk.text):
            nums = [int(x) for x in (a, b) if x]
            key = (unit, h.chunk.source)
            per_file[key] = max(per_file.get(key, 0), *nums)
    parts = []
    for unit in sorted({u for u, _ in per_file}):
        values = {src: n for (u, src), n in per_file.items() if u == unit}
        if len(set(values.values())) > 1:
            parts.append(f"«{unit}» бойынша: " + ", ".join(f"{src} → {n}" for src, n in sorted(values.items())))
    return "; ".join(parts)


def build_kb():
    return KnowledgeBase(load_chunks(DOCS_DIR))


def answer(question, kb, k=3):
    hits = kb.search(question, k=k)
    if not hits or hits[0].coverage < MIN_COVERAGE:
        return Answer(False, FALLBACK)
    top = hits[0]
    # Қосымша бөліктер: бағасы жоғарыдан 60%-дан кем емес болса, ал ӨЗГЕ БЕТТЕН келсе 35%-дан.
    # Бір сұраққа әр түрлі беттер әр түрлі жауап берсе (мысалы жеткізу 1–3 күн / 7–15 күн),
    # соны жасырмай көрсету қайшылықты байқауға мүмкіндік береді.
    shown = [h for h in hits if h is top or (h.coverage >= EXTRA_MIN_COVERAGE and (
        h.score >= 0.6 * top.score or
        (h.chunk.source != top.chunk.source and h.score >= 0.35 * top.score)))]
    text = "\n\n".join(f"[{h.chunk.heading}]\n{h.chunk.text}" for h in shown)
    return Answer(True, text, [(h.chunk.source, h.chunk.heading, round(h.score, 2)) for h in shown],
                  conflict=find_conflict(shown), chunks=[h.chunk for h in shown])


def respond(question, kb, use_llm=None):
    """Толық жауап: іздеу → (бар болса) тіл моделі → болмаса құжаттан үзінді.
    Іздеу «білмеймін» десе, модель МҮЛДЕМ шақырылмайды."""
    a = answer(question, kb)
    if not a.found:
        return a
    if use_llm is None:
        use_llm = llm.is_enabled()
    if use_llm:
        try:
            a.text = llm.generate(question, a.chunks)
            a.generated = True
        except llm.LLMError as e:
            a.llm_error = str(e)
    return a
