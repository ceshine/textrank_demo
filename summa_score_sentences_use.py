"""Using cosine similarity with sentence embeddings from Universal Sentence Encoder."""
import os
import logging
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from langdetect import detect
from summa.pagerank_weighted import pagerank_weighted_scipy as _pagerank
from text_cleaning_en import clean_text_by_sentences as _clean_text_by_sentences
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


def attach_sentence_embeddings(session, sentences, model, batch_size=32):
    # don't use extremely short sentences
    sentences_subset = [x for x in sentences if len(x.text) > 5]
    sentence_embeddings_tmp = []
    for i in range(0, len(sentences_subset), batch_size):
        sentence_embeddings_tmp.append(
            session.run(
                model["sentence_emb"],
                feed_dict={
                    model["sentence_input"]: [
                        x.text for x in sentences_subset[i:(i+batch_size)]
                    ]
                }
            )
        )
    sentence_embeddings_subset = np.concatenate(
        sentence_embeddings_tmp, axis=0)
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


def summarize(text, model_name="large", additional_stopwords=None):
    model = get_model(model_name)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(),
                     tf.tables_initializer()])
        # Make the graph read-only
        tf.get_default_graph().finalize()
        return summarize_with_model(text, session, model, model_name, additional_stopwords)


def summarize_with_model(text, session, model, model_name, additional_stopwords):
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
        if model_name != "xling":
            raise ValueError("Only 'xling' model supports zh.")
        if not ZH_SUPPORT:
            raise ImportError("Missing dependencies for Chinese support.")
        sentences = zh_clean_and_cut_sentences(text)
        for i, sent in enumerate(sentences):
            # Hacky way to overwrite token
            sent.token = i
    elif lang == "ja":
        if model_name != "xling":
            raise ValueError("Only 'xling' model supports ja.")
        if not JA_SUPPORT:
            raise ImportError("Missing dependencies for Japanese support.")
        sentences = ja_clean_and_cut_sentences(text)
        for i, sent in enumerate(sentences):
            # Hacky way to overwrite token
            sent.token = i
    else:
        return ["Language not suppored! (supported languages: en, zh, ja)"], None, lang

    # print([sentence.token for sentence in sentences if sentence.token])
    # Creates the graph and calculates the similarity coefficient for every pair of nodes.
    similarities = attach_sentence_embeddings(
        session, sentences, model, batch_size=32)
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
