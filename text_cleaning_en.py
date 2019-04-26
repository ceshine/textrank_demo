import spacy

from summa.preprocessing.textcleaner import init_textcleanner, filter_words, merge_syntactic_units

NLP = spacy.load("en_core_web_sm")


def split_sentences(text):
    doc = NLP(text)
    return [sent.text.strip() for sent in doc.sents]


def clean_text_by_sentences(text, additional_stopwords=None):
    """Tokenizes a given text into sentences, applying filters and lemmatizing them.
    Returns a SyntacticUnit list. """
    init_textcleanner("english", additional_stopwords)
    original_sentences = split_sentences(text)
    filtered_sentences = filter_words(original_sentences)
    return merge_syntactic_units(original_sentences, filtered_sentences)
