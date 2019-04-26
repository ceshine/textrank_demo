import math
from typing import List, Dict

from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
# from summa.keywords import keywords as _keywords
import uvicorn
import summa.graph

from summa_score_sentences_use import summarize as summarize_use
from summa_score_sentences import summarize as summarize_textrank
from summa_score_words import keywords as _keywords


app = Starlette(debug=True)
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')


def add_alpha(sentences, n=3):
    def transform(score):
        return math.exp(score * 5)

    n = min(n, max(1, int(len(sentences) * 0.5)))
    scores = [transform(x.score) for x in sentences]
    # Note: this does not consider collision
    thres = sorted(scores)[-n]

    min_score = min(scores)
    max_score = max(scores)
    span = max_score - min_score + 1
    for sent in sentences:
        sent.transformed_score = round(
            (transform(sent.score) - min_score + 1) / span, 4) * 50
        sent.alpha = sent.transformed_score / 50
        if transform(sent.score) < thres:
            sent.alpha = 0


def find_node_in_texts(node_text, sentences, lang):
    """Only finds the first occurence"""
    for sent in sentences:
        if sent.token == node_text:
            if lang == "en":
                return [
                    sent.text, sent.paragraph, sent.index,
                    "%.4f" % sent.score, "%.2f" % sent.transformed_score]
            else:
                return [
                    sent.text + "<br/><br/>Tokens: " + str(sent.token),
                    sent.paragraph, sent.index, "%.4f" % sent.score,
                    "%.2f" % sent.transformed_score]
    return ["", -1, -1, -1]


def reconstruct_graph(graph: summa.graph.Graph, sentences: List, lang: str):
    raw_nodes = graph.nodes()
    node_mapping = {
        i: find_node_in_texts(name, sentences, lang)
        for i, name in enumerate(raw_nodes)
    }
    edges = []
    for i in range(len(raw_nodes)-1):
        for j in range(i+1, len(raw_nodes)):
            tmp = graph.get_edge_properties((raw_nodes[i], raw_nodes[j]))
            if tmp["weight"] > 0:
                edges.append((i, j, tmp["weight"]))
    return node_mapping, edges


def transform_word_scores(pagerank_scores: Dict[str, float]) -> Dict[str, float]:
    def transform(score):
        return (score * 10) ** 1.5
    SCALE = 20
    scores = [transform(x) for x in pagerank_scores.values()]
    new_scores = {}
    min_score = min(scores)
    max_score = max(scores)
    span = max_score - min_score + 1
    for key in pagerank_scores.keys():
        new_scores[key] = round(
            (transform(pagerank_scores[key]) - min_score + 1) / span, 4) * SCALE
    return new_scores


def trim_word_nodes(nodes: List[str], pagerank_scores: Dict[str, float], top_n: int):
    kv_pairs = list(sorted(pagerank_scores.items(),
                           key=lambda x: x[1], reverse=True))
    picked = [x[0] for x in kv_pairs[:top_n]]
    return set([i for i, key in enumerate(nodes) if key in picked])


def reconstruct_word_graph(graph: summa.graph.Graph, pagerank_scores: Dict[str, float], top_n: int = None):
    transformed_scores = transform_word_scores(pagerank_scores)
    raw_nodes = graph.nodes()
    included = set(range(len(raw_nodes)))
    if top_n:
        included = trim_word_nodes(raw_nodes, pagerank_scores, top_n)
    edges = []
    included_in_edges = []
    for i in range(len(raw_nodes)-1):
        if i not in included:
            continue
        for j in range(i+1, len(raw_nodes)):
            if j not in included:
                continue
            tmp = graph.get_edge_properties((raw_nodes[i], raw_nodes[j]))
            if tmp["weight"] > 0:
                assert tmp["weight"] == 1
                edges.append((i, j))
                included_in_edges.append(i)
                included_in_edges.append(j)
    node_mapping = {
        i: [name, "%.4f" % pagerank_scores[name], "%.2f" %
            transformed_scores[name]]
        for i, name in enumerate(raw_nodes) if i in included_in_edges
    }
    return node_mapping, edges


@app.route('/', methods=["GET", "POST"])
async def homepage(request):
    if request.method == "POST":
        values = await request.form()
        print("POST params:", values)
        if values['metricInput'].startswith("use-"):
            sentences, graph, lang = summarize_use(
                values['text'], model_name=values['metricInput'][4:])
        else:
            sentences, graph, lang = summarize_textrank(
                values['text'])
        print("Language dected:", lang)
        extra_info = []
        keywords, lemma2words, word_graph, pagerank_scores = _keywords(
            values['text'])
        if lang == "en":
            keyword_formatted = [
                key + " %.2f (%s)" % (score, ", ".join(lemma2words[key]))
                for score, key in keywords[:int(values["n_keywords"])]
            ]
        else:
            keyword_formatted = [
                key + " %.2f" % score
                for score, key in keywords[:int(values["n_keywords"])]
            ]
        if graph is None:
            return HTMLResponse(sentences[0] + "\nDectected language: " + lang)
        # print([sentence.token for sentence in sentences if sentence.token])
        try:
            add_alpha(sentences, int(values["n_sentences"]))
        except (ValueError, KeyError):
            print("Warning: invalid *n* parameter passed!")
            add_alpha(sentences)
        n_paragraphs = max([x.paragraph for x in sentences]) + 1
        paragraphs = []
        for i in range(n_paragraphs):
            paragraphs.append(
                sorted([x for x in sentences if x.paragraph == i], key=lambda x: x.index))
        node_mapping, edges = reconstruct_graph(graph, sentences, lang)
        word_node_mapping, word_edges = reconstruct_word_graph(
            word_graph, pagerank_scores, top_n=int(values["n_keywords"])*5)
        response = templates.TemplateResponse(
            'index.jinja',
            dict(
                request=request,
                paragraphs=paragraphs,
                text=values['text'],
                n_sentences=values["n_sentences"],
                n_keywords=values["n_keywords"],
                metricInput=values["metricInput"],
                word_edges=word_edges,
                edges=edges,
                n_nodes=len(node_mapping),
                n_word_nodes=len(word_node_mapping),
                node_mapping=node_mapping,
                word_node_mapping=word_node_mapping,
                stats=[
                    ("# of Sentence Nodes", len(node_mapping)),
                    ("# of Sentence Edges", len(edges)),
                    ("Max Edge Weight", "%.4f" %
                     max([float(x[2]) for x in edges])),
                    ("Min Edge Weight", "%.4f" %
                     min([float(x[2]) for x in edges])),
                    ("Max Node Score", "%.4f" %
                     max([float(x[3]) for x in node_mapping.values()])),
                    ("Min Node Score", "%.4f" %
                     min([float(x[3]) for x in node_mapping.values()]))
                ],
                keywords=keyword_formatted)
        )
    else:
        response = templates.TemplateResponse(
            'index.jinja', dict(request=request, text="", n_sentences=2, n_keywords=5, metricInput="textrank"))
    return response

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
