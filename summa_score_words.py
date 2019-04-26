from typing import List, Dict, Tuple, Optional

from langdetect import detect
from summa.pagerank_weighted import pagerank_weighted_scipy as _pagerank
from summa.preprocessing.textcleaner import clean_text_by_word as _clean_text_by_word
from summa.preprocessing.textcleaner import tokenize_by_word as _tokenize_by_word
from summa.commons import build_graph as _build_graph
from summa.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from summa.keywords import (
    _get_words_for_graph, _set_graph_edges, _lemmas_to_words
)
import summa.graph

# Optional Dependencies
try:
    from text_cleaning_zh import clean_and_cut_words as zh_clean_and_cut_words
    ZH_SUPPORT = True
except ImportError:
    ZH_SUPPORT = False

try:
    from text_cleaning_ja import clean_and_cut_words as ja_clean_and_cut_words
    JA_SUPPORT = True
except ImportError:
    JA_SUPPORT = False


def _extract_tokens(lemmas, scores) -> List[Tuple[float, str]]:
    lemmas.sort(key=lambda s: scores[s], reverse=True)
    return [(scores[lemmas[i]], lemmas[i]) for i in range(len(lemmas))]


def keywords(
        text: str, deaccent: bool = False,
        additional_stopwords: List[str] = None) -> Tuple[
            List[Tuple[float, str]], Optional[Dict[str, List[str]]],
            Optional[summa.graph.Graph], Dict[str, float]]:
    if not isinstance(text, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")

    lang = detect(text)[:2]
    if lang == "en":
        # Gets a dict of word -> lemma
        tokens = _clean_text_by_word(
            text, "english", deacc=deaccent,
            additional_stopwords=additional_stopwords)
        split_text = list(_tokenize_by_word(text))
    elif lang == "zh" or lang == "ko":  # zh-Hant sometimes got misclassified into ko
        if not ZH_SUPPORT:
            raise ImportError("Missing dependencies for Chinese support.")
        tokens = zh_clean_and_cut_words(text)
        split_text = [x.text for x in tokens]
        tokens = {x.text: x for x in tokens}
    elif lang == "ja":
        if not JA_SUPPORT:
            raise ImportError("Missing dependencies for Japanese support.")
        tokens = ja_clean_and_cut_words(text)
        split_text = [x.text for x in tokens]
        tokens = {x.text: x for x in tokens}
    else:
        print("Language not suppored! (supported languages: en zh)")
        return [], {}, None, {}

    # Creates the graph and adds the edges
    graph = _build_graph(_get_words_for_graph(tokens))
    _set_graph_edges(graph, tokens, split_text)
    del split_text  # It's no longer used

    _remove_unreachable_nodes(graph)

    # PageRank cannot be run in an empty graph.
    if not graph.nodes():
        return [], {}, None, {}

    # Ranks the tokens using the PageRank algorithm. Returns dict of lemma -> score
    pagerank_scores = _pagerank(graph)

    extracted_lemmas = _extract_tokens(graph.nodes(), pagerank_scores)

    lemmas_to_word = None
    if lang == "en":
        lemmas_to_word = _lemmas_to_words(tokens)

    return extracted_lemmas, lemmas_to_word, graph, pagerank_scores
