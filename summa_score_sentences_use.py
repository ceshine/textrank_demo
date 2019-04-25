"""Using cosine similarity with sentence embeddings from Universal Sentence Encoder."""
import os
import logging
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from langdetect import detect
from summa.pagerank_weighted import pagerank_weighted_scipy as _pagerank
from summa.preprocessing.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from summa.commons import build_graph as _build_graph
from summa.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from summa.summarizer import _set_graph_edge_weights, _add_scores_to_sentences

# Optional Dependencies
try:
    # required by "xling" model
    import tf_sentencepiece
except ImportError:
    pass

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


tf.logging.set_verbosity(logging.WARNING)

MODELS = {
    "xling": "https://tfhub.dev/google/universal-sentence-encoder-xling-many/1",
    "large": "https://tfhub.dev/google/universal-sentence-encoder-large/3",
    # base does not work with GPU
    "base": "https://tfhub.dev/google/universal-sentence-encoder/2"
}


def get_model(model_name="large"):
    tf.reset_default_graph()
    # Define graph
    sentence_input = tf.placeholder(tf.string, shape=(None))
    encoder = hub.Module(MODELS[model_name])
    # For evaluation we use exactly normalized rather than
    # approximately normalized.
    sentence_emb = tf.nn.l2_normalize(encoder(sentence_input), axis=1)
    return {"sentence_input": sentence_input, "sentence_emb": sentence_emb}


def cosine_similarity(similarity_matrix, id_1, id_2):
    return similarity_matrix[id_1, id_2]


def attach_setence_embeddings(sentences, model_name, batch_size=32):
    model = get_model(model_name)
    # remove extremely short sentences
    sentence = [x for x in sentences if len(x.text) > 5]
    sentence_embeddings = []
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(),
                     tf.tables_initializer()])
        for i in range(0, len(sentences), batch_size):
            sentence_embeddings.append(
                session.run(
                    model["sentence_emb"],
                    feed_dict={
                        model["sentence_input"]: [
                            x.text for x in sentences[i:(i+batch_size)]
                        ]
                    }
                )
            )
    sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
    # A rather hacky way to attach embeddings and replace token property
    for i, sentence in enumerate(sentences):
        sentence.embeddings = sentence_embeddings[i]
        sentence.token = i
    similarities = sentence_embeddings @ sentence_embeddings.T
    # print(similarities[np.tril_indices(similarities.shape[0], k=-1)])
    # print(np.where(np.tril(similarities, k=-1) > 0.95))
    return sentences, similarities


def summarize(text, model_name="large", additional_stopwords=None):
    if not isinstance(text, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")
    lang = detect(text)[:2]
    if lang == "en":
        paragraphs = text.split("\n")
        sentences = []
        for i, paragraph in enumerate(paragraphs):
            # Gets a list of processed sentences.
            if paragraph:
                tmp = _clean_text_by_sentences(
                    paragraph, "english", additional_stopwords)
                for sent in tmp:
                    sent.paragraph = i
                sentences += tmp
    elif lang == "zh" or lang == "ko":  # zh-Hant sometimes got misclassified into ko
        raise NotImplementedError("Not supported yet： zh.")
    elif lang == "ja":
        raise NotImplementedError("Not supported yet： ja.")
    else:
        return ["Language not suppored! (supported languages: en, zh, ja)"], None, lang

    # print([sentence.token for sentence in sentences if sentence.token])
    # Creates the graph and calculates the similarity coefficient for every pair of nodes.
    sentences, similarities = attach_setence_embeddings(
        sentences, batch_size=32, model_name=model_name)
    graph = _build_graph(list(range(len(sentences))))
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
    res, _, lang = summarize("""Of all of President Trump’s former associates who have come under scrutiny in the special counsel’s Russia investigation, his former personal lawyer, Michael D. Cohen, has undertaken perhaps the most surprising and risky legal strategy.

Mr. Cohen has twice pleaded guilty in federal court in Manhattan to a litany of crimes, and he has volunteered information to the special counsel and other agencies investigating Mr. Trump and his inner circle. He did all this without first obtaining a traditional, ironclad deal under which the government would commit to seeking leniency on Mr. Cohen’s behalf when he is sentenced on Dec. 12.

Mr. Cohen has concluded that his life has been utterly destroyed by his relationship with Mr. Trump and his own actions, and to begin anew he needed to speed up the legal process by quickly confessing his crimes and serving any sentence he receives, according to his friends and associates, and analysis of documents in the case.""")
    assert lang == "en"
    for row in res:
        print(f"{row.score} {row.text}")
