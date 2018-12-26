import re
from collections import namedtuple
from typing import List, Sequence, Dict, Any, Tuple

import nagisa

from summa.syntactic_unit import SyntacticUnit
from text_cleaning_zh import clean_text, insert_unit

DEBUG = 1

TOKEN = namedtuple("token", ["content", "postag", "postag_raw"])

TAG_MAPPING = {
    'oov': 'unknown',
    '補助記号': 'punc',
    '名詞': 'noun',
    '空白': 'punc',
    '助詞': 'part',
    '接尾辞': 'suffix',
    '動詞': 'verb',
    '連体詞': 'adj',
    '助動詞': 'aux verb',
    '形容詞': 'adj',
    '感動詞': 'interjection',
    '接頭辞': 'prefix',
    '記号': 'symbol',
    '接続詞': 'conj',
    '副詞': 'adv',
    '代名詞': 'pronoun',
    '形状詞': 'adj',  # 形容動詞
    'web誤脱': 'unknown',
    'URL': 'url',
    '英単語': 'unknown',
    '漢文': 'unknown',
    '未知語': 'unknown',
    '言いよどみ': 'unknown',
    'ローマ字文': 'unknown'
}


def get_tokens(text) -> List[TOKEN]:
    results = nagisa.tagging(text.strip())
    mapped_tags = [TAG_MAPPING[x] for x in results.postags]
    return [TOKEN(*x) for x in zip(results.words, mapped_tags, results.postags)]


def clean_and_cut_words(text: str, pos_tags: Sequence = ("noun", "verb"),
                        stopwords: Sequence = ("です", "する", "し", "いう"), verbose=DEBUG,
                        filter_digits=True) -> List[SyntacticUnit]:
    paragraphs = clean_text(text).split("\n")
    tokens = [get_tokens(x) for x in paragraphs if x]
    results: List[SyntacticUnit] = []
    for paragraph_idx, paragraph in enumerate(tokens):
        word_idx = 0
        for token in paragraph:
            if verbose:
                print(token.content, token.postag, token.postag_raw)
            if (token.postag in pos_tags and
                (token.content not in stopwords) and
                    not (token.content.isdigit() and filter_digits)):
                insert_unit(results, [token.content],
                            [token.content], paragraph_idx, word_idx)
                word_idx += 1
    return results


def clean_and_cut_sentences(text: str, sentence_delimiter: str = "。！？；",
                            pos_tags: Sequence = ("noun", "verb"),
                            stopwords: Sequence = ("です", "する", "し", "いう"), filter_digits=True) -> List[SyntacticUnit]:
    paragraphs = clean_text(text).split("\n")
    tokens = [get_tokens(x) for x in paragraphs if x]
    results: List[SyntacticUnit] = []
    for paragraph_idx, paragraph in enumerate(tokens):
        raw_text: List[str] = []
        filtered: List[str] = []
        sentence_idx = 0
        for token in paragraph:
            if token.content in sentence_delimiter and raw_text:
                # End of a sentence
                raw_text.append(token.content)
                insert_unit(results, raw_text, filtered,
                            paragraph_idx, sentence_idx)
                sentence_idx += 1
                raw_text, filtered = [], []
            else:
                # Usual token
                raw_text.append(token.content)
                if (token.postag in pos_tags and
                    (token.content not in stopwords) and
                        not (token.content.isdigit() and filter_digits)):
                    filtered.append(token.content)
        if raw_text:
            # is not right after a setence delimiter
            insert_unit(results, raw_text, filtered,
                        paragraph_idx, sentence_idx)
    return results
