import re
from typing import List, Sequence, Dict, Any

from summa.syntactic_unit import SyntacticUnit

from baidunlp import ner_tags


def insert_unit(target_list: List[SyntacticUnit], raw: List[str],
                tokens: List[str], pidx: int, sidx: int) -> None:
    target_list.append(
        SyntacticUnit(
            text="".join(raw),
            token=" ".join(tokens),
            index=sidx,
            paragraph=pidx
        )
    )


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(\s)\1+", r"\1", text)
    return text


def get_tokens(text):
    text = clean_text(text)
    return ner_tags(text.strip())


def clean_and_cut_words(text: str, pos_tags: Sequence = ("noun", "verb"),
                        stopwords: Sequence = ("是", "有")) -> List[SyntacticUnit]:
    tokens = get_tokens(text)
    return cut_words(tokens, pos_tags, stopwords)


def cut_words(tokens: List[Dict[str, Any]],  pos_tags: Sequence,
              stopwords: Sequence) -> List[SyntacticUnit]:
    results: List[SyntacticUnit] = []
    paragraph_idx, word_idx = 0, 0
    for token in tokens:
        if token["item"] == "\n":
            # Linebreak marks the end of a paragraph
            paragraph_idx += 1
            word_idx = 0
        else:
            # Usual token
            if token["tag"] in pos_tags and (token["item"] not in stopwords):
                insert_unit(results, [token["item"]],
                            [token["item"]], paragraph_idx, word_idx)
    return results


def clean_and_cut_sentences(text: str, sentence_delimiter: str = "。！？；",
                            pos_tags: Sequence = ("noun", "verb"),
                            stopwords: Sequence = ("是",)) -> List[SyntacticUnit]:
    tokens = get_tokens(text)
    return cut_sentences(tokens, sentence_delimiter, pos_tags, stopwords)


def cut_sentences(tokens: List[Dict[str, Any]], sentence_delimiter: str,
                  pos_tags: Sequence, stopwords: Sequence) -> List[SyntacticUnit]:
    results: List[SyntacticUnit] = []
    paragraph_idx, sentence_idx = 0, 0
    raw_text: List[str] = []
    filtered: List[str] = []
    for token in tokens:
        if token["item"] in sentence_delimiter and raw_text:
                # End of a sentence
            raw_text.append(token["item"])
            insert_unit(results, raw_text, filtered,
                        paragraph_idx, sentence_idx)
            sentence_idx += 1
            raw_text, filtered = [], []
        elif token["item"] == "\n":
            # Linebreak marks the end of a paragraph
            if raw_text:
                    # is not right after a setence delimiter
                insert_unit(results, raw_text, filtered,
                            paragraph_idx, sentence_idx)
                raw_text, filtered = [], []
            paragraph_idx += 1
            sentence_idx = 0
        else:
            # Usual token
            raw_text.append(token["item"])
            if token["tag"] in pos_tags and (token["item"] not in stopwords):
                filtered.append(" ".join(token["basic_words"]))
    return results
