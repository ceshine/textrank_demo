import os
from pathlib import Path
from functools import partial

from langdetect import detect
import numpy as np
from sklearn.preprocessing import normalize
from summa.pagerank_weighted import pagerank_weighted_scipy as _pagerank
from summa.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from summa.summarizer import _set_graph_edge_weights, _add_scores_to_sentences
from summa.commons import build_graph as _build_graph

from text_cleaning_en import clean_text_by_sentences as _clean_text_by_sentences
from summa_score_sentences_use import cut_sentences_by_rule, cosine_similarity

MODEL_PATH = Path(os.environ["LASER"]) / "models/"


def attach_sentence_embeddings(lang, sentences, batch_size=32):
    from laser.shortcuts import lines_to_embeddings

    # don't use extremely short sentences
    sentences_subset = [x for x in sentences if len(x.text) > 5]
    sentence_embeddings_subset = lines_to_embeddings(
        lang,
        [x.text for x in sentences_subset],
        str(MODEL_PATH / "bilstm.93langs.2018-12-26.pt"),
        str(MODEL_PATH / "93langs.fcodes"),
        use_cpu=False,
        batch_size=batch_size
    ).reshape(len(sentences_subset), -1)
    # print(len(sentences_subset))
    # print(sentence_embeddings_subset.shape)
    # l2 normalize
    sentence_embeddings_subset = normalize(
        sentence_embeddings_subset, norm="l2", axis=1, copy=False, return_norm=False
    )
    sentence_embeddings = np.zeros(
        (len(sentences), sentence_embeddings_subset.shape[1]), dtype="float32")
    # A rather hacky way to attach embeddings
    for i, sentence in enumerate(sentences_subset):
        sentence_embeddings[sentence.token, :] = (
            sentence_embeddings_subset[i, :])
    similarities = sentence_embeddings @ sentence_embeddings.T
    # print(similarities[np.tril_indices(similarities.shape[0], k=-1)])
    # print(np.where(np.tril(similarities, k=-1) > 0.95))
    return similarities


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
                tmp = _clean_text_by_sentences(
                    paragraph, additional_stopwords)
                if tmp:
                    for j, sent in enumerate(tmp):
                        sent.paragraph = paragraph_index
                        # Hacky way to overwrite token
                        sent.token = len(sentences) + j
                    sentences += tmp
                    paragraph_index += 1
    elif lang == "zh" or lang == "ko":  # zh-Hant sometimes got misclassified into ko
        sentences = cut_sentences_by_rule(text)
    elif lang == "ja":
        raise NotImplementedError("No ja support yet.")
    else:
        return ["Language not suppored! (supported languages: en, zh)"], None, lang
    similarities = attach_sentence_embeddings(
        lang, sentences,  batch_size=32)
    graph = _build_graph([x.token for x in sentences])
    _set_graph_edge_weights(graph, partial(cosine_similarity, similarities))

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
    res, _, lang = summarize("""
By Wednesday morning, he was trying to marry those two thoughts into a single message — both embracing the report and trashing it. “The Mueller Report, despite being written by Angry Democrats and Trump Haters, and with unlimited money behind it ($35,000,000), didn’t lay a glove on me,” he wrote. “I DID NOTHING WRONG.”

In subsequent tweets, he tried again to claim victory amid his victimhood, casting the investigation as a contest in which he prevailed. In terms rarely used regarding a criminal investigation, he asserted that “We waited for Mueller and WON” and denounced “the Witch Hunt, which I have already won.”""")
    assert lang == "en"
    for row in res:
        print(f"{row.score} {row.text}")
