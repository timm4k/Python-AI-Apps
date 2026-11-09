from pathlib import Path

import nltk
from nltk.data import FileSystemPathPointer
from nltk.tag import PerceptronTagger
from nltk.tokenize.punkt import PunktSentenceTokenizer, load_punkt_params

from resource_downloads import ensure_language_assets, ensure_movie_reviews


def prepare_resources(
    root: Path, *, reviews_per_category: int = 100, seed: int = 42
) -> None:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ensure_language_assets(root)
    ensure_movie_reviews(
        root / "nltk/corpora/movie_reviews", reviews_per_category, seed
    )
    nltk.data.path[:] = [str(root / "nltk"), str(root)]


def load_language_tools(root: Path):
    root = root.resolve()
    tagger = PerceptronTagger(loc=str(root / "tagger"))
    sentence_tokenizer = PunktSentenceTokenizer(
        load_punkt_params(FileSystemPathPointer(str(root / "tokenizer")))
    )
    return tagger, sentence_tokenizer
