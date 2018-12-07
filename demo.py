import math
from typing import List
# from collections import namedtuple

from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
import uvicorn
import summa.graph

from summa_score_sentences import summarize


app = Starlette(debug=True, template_directory='templates')
app.mount('/static', StaticFiles(directory='static'), name='static')


def add_alpha(sentences, n=3):
    def transform(score):
        return math.exp(score * 5)

    n = min(n, max(1, int(len(sentences) * 0.4)))
    scores = [transform(x.score) for x in sentences]
    # Note: this does not consider collision
    thres = sorted(scores)[-n]

    min_score = min(scores)
    max_score = max(scores)
    span = max_score - min_score
    for sent in sentences:
        if transform(sent.score) < thres:
            sent.alpha = 0
        else:
            sent.alpha = round(
                (transform(sent.score) - min_score) / span, 4)


def find_node_in_texts(node_text, sentences):
    """Only finds the first occurence"""
    for sent in sentences:
        if sent.token == node_text:
            return [sent.text, sent.paragraph, sent.index, "%.4f" % sent.score]
    return ["", -1, -1, -1]


def reconstruct_graph(graph: summa.graph.Graph, sentences: List):
    raw_nodes = graph.nodes()
    node_mapping = {
        i: find_node_in_texts(name, sentences)
        for i, name in enumerate(raw_nodes)
    }
    edges = []
    for i in range(len(raw_nodes)-1):
        for j in range(i+1, len(raw_nodes)):
            tmp = graph.get_edge_properties((raw_nodes[i], raw_nodes[j]))
            if tmp["weight"] > 0:
                edges.append((i, j, tmp["weight"]))
    return node_mapping, edges


@app.route('/', methods=["GET", "POST"])
async def homepage(request):
    template = app.get_template('index.jinja')
    if request.method == "POST":
        values = await request.form()
        print("POST params:", values)
        sentences, graph = summarize(values['text'])
        try:
            add_alpha(sentences, int(values["n"]))
        except (ValueError, KeyError):
            print("Warning: invalid *n* parameter passed!")
            add_alpha(sentences)
        n_paragraphs = max([x.paragraph for x in sentences]) + 1
        paragraphs = []
        for i in range(n_paragraphs):
            paragraphs.append(
                sorted([x for x in sentences if x.paragraph == i], key=lambda x: x.index))
        node_mapping, edges = reconstruct_graph(graph, sentences)
        content = template.render(
            paragraphs=paragraphs,
            text=values['text'],
            n=values["n"],
            edges=edges,
            n_nodes=len(node_mapping),
            node_mapping=node_mapping,
            stats=[
                ("# of Sentences", len(node_mapping)),
                ("# of Edges", len(edges)),
                ("Max Edge Weight", "%.4f" %
                 max([float(x[2]) for x in edges])),
                ("Min Edge Weight", "%.4f" %
                 min([float(x[2]) for x in edges])),
                ("Max Node Score", "%.4f" %
                 max([float(x[3]) for x in node_mapping.values()])),
                ("Min Node Score", "%.4f" %
                 min([float(x[3]) for x in node_mapping.values()]))
            ]
        )
    else:
        content = template.render(text="", n=2)
    return HTMLResponse(content)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
