from collections import Counter

import pandas as pd
from nltk import bigrams, word_tokenize

WORDNET_POS = {"J": "a", "V": "v", "N": "n", "R": "r"}


def sentence_boundary_comparison(tokenizer) -> pd.DataFrame:
    text = "Dr. Wayne watched The Batman. He rated it 8.5 out of 10. Would he watch it again? Absolutely!"
    methods = {
        "Split at every period": [
            part.strip() for part in text.split(".") if part.strip()
        ],
        "English Punkt": tokenizer.tokenize(text),
    }
    return pd.DataFrame(
        [
            {"method": method, "segments": len(parts), "result": " | ".join(parts)}
            for method, parts in methods.items()
        ]
    )


def lemmatization_comparison(lemmatizer, stemmer, tagger) -> pd.DataFrame:
    examples = [
        ("running", "She is running every morning."),
        ("studies", "The studies were published yesterday."),
        ("better", "She sings better in this film."),
        ("children", "The children enjoyed the film."),
        ("movies", "These movies are entertaining."),
        ("went", "She went to the cinema yesterday."),
    ]
    rows = []
    for target, sentence in examples:
        tagged = tagger.tag(word_tokenize(sentence, preserve_line=True))
        tag = next(tag for word, tag in tagged if word.lower() == target)
        rows.append(
            {
                "word": target,
                "POS": tag,
                "noun-only lemma": lemmatizer.lemmatize(target),
                "POS-aware lemma": lemmatizer.lemmatize(
                    target, WORDNET_POS.get(tag[0], "n")
                ),
                "stem": stemmer.stem(target),
            }
        )
    return pd.DataFrame(rows)


def negation_comparison(english_stopwords: set[str]) -> pd.DataFrame:
    examples = [
        "The movie is bad",
        "The movie is not bad",
        "The movie is good",
        "The movie is not good",
    ]
    protected_stopwords = english_stopwords - {"not", "no", "nor", "never"}
    rows = []
    for text in examples:
        tokens = word_tokenize(text.lower(), preserve_line=True)
        rows.append(
            {
                "sentence": text,
                "all stopwords removed": " ".join(
                    word for word in tokens if word not in english_stopwords
                ),
                "negation retained": " ".join(
                    word for word in tokens if word not in protected_stopwords
                ),
            }
        )
    return pd.DataFrame(rows)


def word_order_comparison() -> pd.DataFrame:
    sentences = ["the hero defeats the villain", "the villain defeats the hero"]
    sequences = [word_tokenize(text, preserve_line=True) for text in sentences]
    vocabulary = sorted(set(word for sequence in sequences for word in sequence))
    rows = []
    for text, sequence in zip(sentences, sequences, strict=True):
        counts = Counter(sequence)
        rows.append(
            {
                "sentence": text,
                "word counts": [counts[word] for word in vocabulary],
                "bigrams": " | ".join(" ".join(pair) for pair in bigrams(sequence)),
            }
        )
    return pd.DataFrame(rows).rename(
        columns={"word counts": f"counts ({', '.join(vocabulary)})"}
    )
