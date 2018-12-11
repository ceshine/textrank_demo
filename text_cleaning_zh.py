import re
from typing import List

from summa.syntactic_unit import SyntacticUnit

from baidunlp import ner_tags


def clean_and_cut_sentences(text, sentence_delimiter="。！？；",
                            pos_tags=("noun", "verb")) -> List[SyntacticUnit]:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(\s)\1+", r"\1", text)
    tokens = ner_tags(text.strip())
    results = []
    paragraph_idx, sentence_idx = 0, 0
    raw_text, filtered = [], []
    for token in tokens:
        if token["item"] in sentence_delimiter and raw_text:
            raw_text.append(token["item"])
            results.append(
                SyntacticUnit(
                    text="".join(raw_text),
                    token=" ".join(filtered),
                    index=sentence_idx,
                    paragraph=paragraph_idx
                )
            )
            sentence_idx += 1
            raw_text, filtered = [], []
        elif token["item"] == "\n":
            if raw_text:
                # is not right after a setence delimiter
                results.append(
                    SyntacticUnit(
                        text="".join(raw_text),
                        token=" ".join(filtered),
                        index=sentence_idx,
                        paragraph=paragraph_idx
                    )
                )
                raw_text, filtered = [], []
            paragraph_idx += 1
            sentence_idx = 0
        else:
            raw_text.append(token["item"])
            if token["tag"] in pos_tags:
                filtered.append(" ".join(token["basic_words"]))
                # filtered.append(token["item"])
    return results
