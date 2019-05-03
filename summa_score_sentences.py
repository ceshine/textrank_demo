"""Using similarity function from the original TextRank algorithm."""
from langdetect import detect
from summa.pagerank_weighted import pagerank_weighted_scipy as _pagerank
# from summa.preprocessing.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from text_cleaning_en import clean_text_by_sentences as en_clean_text_by_sentences
from summa.commons import build_graph as _build_graph
from summa.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from summa.summarizer import _set_graph_edge_weights, _add_scores_to_sentences

# Optional Dependencies
try:
    from text_cleaning_zh import clean_and_cut_sentences as zh_clean_and_cut_sentences
    ZH_SUPPORT = True
except ImportError:
    ZH_SUPPORT = False

try:
    from text_cleaning_ja import clean_and_cut_sentences as ja_clean_and_cut_sentences
    JA_SUPPORT = True
except ImportError:
    JA_SUPPORT = False


def summarize(text, additional_stopwords=None):
    if not isinstance(text, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")

    lang = detect(text)[:2]
    if lang == "en":
        paragraphs = text.split("\n")
        sentences = []
        paragraph_index = 0
        for paragraph in paragraphs:
            # Gets a list of processed sentences.
            if paragraph:
                tmp = en_clean_text_by_sentences(
                    paragraph, additional_stopwords)
                # tmp = _clean_text_by_sentences(
                #     paragraph, "english")
                if tmp:
                    for sent in tmp:
                        sent.paragraph = paragraph_index
                    sentences += tmp
                    paragraph_index += 1
    elif lang == "zh" or lang == "ko":  # zh-Hant sometimes got misclassified into ko
        if not ZH_SUPPORT:
            raise ImportError("Missing dependencies for Chinese support.")
        sentences = zh_clean_and_cut_sentences(text)
    elif lang == "ja":
        if not JA_SUPPORT:
            raise ImportError("Missing dependencies for Japanese support.")
        sentences = ja_clean_and_cut_sentences(text)
    else:
        return ["Language not suppored! (supported languages: en, zh, ja)"], None, lang

    # Creates the graph and calculates the similarity coefficient for every pair of nodes.
    graph = _build_graph(
        [sentence.token for sentence in sentences if sentence.token and len(sentence.token) > 2])
    _set_graph_edge_weights(graph)

    # Remove all nodes with all edges weights equal to zero.
    _remove_unreachable_nodes(graph)

    # PageRank cannot be run in an empty graph.
    if len(graph.nodes()) == 0:
        return []

    # Ranks the tokens using the PageRank algorithm. Returns dict of sentence -> score
    pagerank_scores = _pagerank(graph)

    # Adds the summa scores to the sentence objects.
    _add_scores_to_sentences(sentences, pagerank_scores)

    # Sorts the sentences
    sentences.sort(key=lambda s: s.score, reverse=True)
    return sentences, graph, lang


if __name__ == "__main__":
    res, _, lang = summarize("""Of all of President Trump’s former associates who have come under scrutiny in the special counsel’s Russia investigation, his former personal lawyer, Michael D. Cohen, has undertaken perhaps the most surprising and risky legal strategy.

Mr. Cohen has twice pleaded guilty in federal court in Manhattan to a litany of crimes, and he has volunteered information to the special counsel and other agencies investigating Mr. Trump and his inner circle. He did all this without first obtaining a traditional, ironclad deal under which the government would commit to seeking leniency on Mr. Cohen’s behalf when he is sentenced on Dec. 12.

Mr. Cohen has concluded that his life has been utterly destroyed by his relationship with Mr. Trump and his own actions, and to begin anew he needed to speed up the legal process by quickly confessing his crimes and serving any sentence he receives, according to his friends and associates, and analysis of documents in the case.

Cohen, has undertaken perhaps the most surprising and risky legal strategy.""")
    assert lang == "en"
    for row in res:
        print(f"{row.score} {row.text}")
