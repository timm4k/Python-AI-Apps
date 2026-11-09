import random
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

BASE_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages"


def package_bytes(package: str) -> bytes:
    with urlopen(f"{BASE_URL}/{package}", timeout=120) as response:
        return response.read()


def ensure_language_assets(root: Path) -> None:
    packages = {
        "tokenizers/punkt_tab.zip": {
            f"punkt_tab/english/{name}": root / "tokenizer" / name
            for name in (
                "abbrev_types.txt",
                "collocations.tab",
                "ortho_context.tab",
                "sent_starters.txt",
            )
        },
        "taggers/averaged_perceptron_tagger_eng.zip": {
            f"averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.{name}.json": root
            / "tagger"
            / f"averaged_perceptron_tagger_eng.{name}.json"
            for name in ("classes", "tagdict", "weights")
        },
        "corpora/stopwords.zip": {
            "stopwords/english": root / "nltk/corpora/stopwords/english"
        },
    }
    for package, members in packages.items():
        missing = {
            name: target for name, target in members.items() if not target.is_file()
        }
        if not missing:
            continue
        with ZipFile(BytesIO(package_bytes(package))) as archive:
            for name, target in missing.items():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    wordnet = root / "nltk/corpora/wordnet.zip"
    if not wordnet.is_file():
        wordnet.parent.mkdir(parents=True, exist_ok=True)
        wordnet.write_bytes(package_bytes("corpora/wordnet.zip"))


def ensure_movie_reviews(target: Path, per_category: int, seed: int) -> None:
    counts = [
        len(list((target / category).glob("*.txt"))) for category in ("neg", "pos")
    ]
    if counts == [per_category, per_category]:
        return
    if any(count > per_category for count in counts):
        raise ValueError("Movie review cache exceeds the configured sample size")
    with ZipFile(BytesIO(package_bytes("corpora/movie_reviews.zip"))) as archive:
        rng = random.Random(seed)
        selected = []
        for category in ("neg", "pos"):
            candidates = sorted(
                name
                for name in archive.namelist()
                if name.startswith(f"movie_reviews/{category}/")
                and name.endswith(".txt")
            )
            selected.extend(rng.sample(candidates, per_category))
        for name in selected:
            relative = Path(name).relative_to("movie_reviews")
            destination = (target / relative).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise ValueError("Invalid review archive path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
        (target / "README").write_bytes(archive.read("movie_reviews/README"))
