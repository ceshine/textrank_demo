"""Sentenc Scoring using Preloaded Xling Universal Sentence Encoder
"""
import tensorflow as tf

from summa_score_sentences_use import get_model, summarize_with_model

XLING = get_model("xling")
SESSION = tf.Session()
SESSION.run([tf.global_variables_initializer(),
             tf.tables_initializer()])
# Make the graph read-only
tf.get_default_graph().finalize()


def summarize_xling(text, additional_stopwords=None):
    return summarize_with_model(text, SESSION, XLING, "xling", additional_stopwords)
