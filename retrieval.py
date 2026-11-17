"""Қарапайым BM25 іздеу (сыртқы кітапханасыз). Қазақ тілі жалғаулы тіл, сондықтан
сөздерді алғашқы STEM_LEN әріпке дейін қысқартамыз (prefix stemming): «кепілдік» ~ «кепілдігі»."""
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STEM_LEN = 4
STOPWORDS = {
    "ма", "ме", "ба", "бе", "па", "пе", "және", "мен", "бен", "пен", "ол", "бұл", "осы",
    "бар", "жоқ", "не", "қалай", "қандай", "қанша", "неше", "керек", "болады", "бола",
    "ал", "да", "де", "та", "те", "үшін", "туралы", "ше", "кім", "қай", "қайда",
    "алады", "алу",  # жалпы етістіктер («қанша уақыт алады?»): ақпарат тасымайды
}


# Мағынасы бірдей сөздер (қысқартылған түрде): BM25 синонимді өзі білмейді.
# «қанша уақыт?» деген сұрақ «мерзімі» деп жазылған бөлімді таба алуы үшін.
SYNONYMS = {"мерз": "уақы"}


def tokenize(text):
    tokens = re.findall(r"\w+", text.lower())
    stems = [t[:STEM_LEN] for t in tokens if t not in STOPWORDS and (t.isdigit() or len(t) > 1)]
    return [SYNONYMS.get(s, s) for s in stems]


@dataclass
class Chunk:
    source: str
    heading: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float
    coverage: float  # сұрақтың «салмақталған» қанша бөлігі осы бөлікте бар (0..1)


def load_chunks(docs_dir):
    """Әр `## Тақырып` бөлімі = бір бөлік (chunk)."""
    chunks = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        heading, body = None, []
        def flush():
            if heading and body:
                chunks.append(Chunk(path.name, heading, " ".join(body).strip()))
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                flush()
                heading, body = line[3:].strip(), []
            elif line.startswith("#") or not line.strip():
                continue
            elif heading:
                body.append(line.strip())
        flush()
    return chunks


class KnowledgeBase:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        # Тақырып сөздері екі есе салмақты (тақырып — бөлімнің ең нақты сипаттамасы)
        self.docs = [Counter(tokenize(c.heading) * 2 + tokenize(c.text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        # Құжаттарда мүлдем кездеспейтін сөз — ең «сирек» (ең салмақты) сөз: ол сұрақтың
        # домен сыртында екенін көрсетеді («президент», «RTX 5090»).
        self.unseen_idf = math.log(1 + (n + 0.5) / 0.5)

    def search(self, query, k=3):
        q = list(dict.fromkeys(tokenize(query)))  # бірегей сөздер, реті сақталады
        if not q:
            return []
        hits = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for t in q:
                f = doc.get(t, 0)
                if f:
                    norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len))
                    score += self.idf.get(t, 0) * norm
            weights = {t: self.idf.get(t, self.unseen_idf) for t in q}
            coverage = sum(w for t, w in weights.items() if doc.get(t)) / sum(weights.values())
            if score > 0:
                hits.append(Hit(self.chunks[i], score, coverage))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
