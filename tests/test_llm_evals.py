"""LLM evals ішіндегі тексергіштердің өз тесттері (Google-ға қосылмайды)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
from retrieval import Chunk  # noqa: E402
import run_llm_evals as ev  # noqa: E402


def chunk(src, text):
    return Chunk(source=src, heading="Тақырып", text=text) if "source" in Chunk.__dataclass_fields__ else None


class Checkers(unittest.TestCase):
    def setUp(self):
        self.chunks = [chunk("a.md", "Жеткізу 1–3 жұмыс күні."), chunk("b.md", "Жеткізу 7–15 күн.")]

    def test_grounded_ok_and_invented(self):
        self.assertEqual(ev.check_grounded("1–3 күн.", self.chunks, "?"), [])
        self.assertEqual(ev.check_grounded("2 күн ішінде.", self.chunks, "?"), ["2"])

    def test_question_numbers_allowed(self):
        self.assertEqual(ev.check_grounded("12 ай.", self.chunks, "12 айға бола ма?"), [])

    def test_kazakh(self):
        self.assertTrue(ev.check_kazakh("Жеткізу бар, қазір"))
        self.assertFalse(ev.check_kazakh("Delivery takes 3 days"))

    def test_both_needs_two_sources_or_manager(self):
        self.assertTrue(ev.check_both("1–3 немесе 7–15 күн.", self.chunks))
        self.assertTrue(ev.check_both("Менеджер нақтылайды.", self.chunks))
        self.assertFalse(ev.check_both("Тек 1–3 күн.", self.chunks))

    def test_sentences(self):
        self.assertEqual(len(ev.sentences("Бір. Екі! Үш?")), 3)


if __name__ == "__main__":
    unittest.main()
