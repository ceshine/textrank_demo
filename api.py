from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import functools

import uvicorn
from starlette.middleware.cors import CORSMiddleware

from summa_score_sentences import summarize as summarize_textrank
from summa_score_sentences_xling import summarize_xling


class HighlightRequest(BaseModel):
    text: str
    model: str = "textrank"


class Sentence(BaseModel):
    paragraph: int
    index: int
    text: str
    score: float


class HighlightResults(BaseModel):
    success: bool
    message: str = ""
    sentences: List[Sentence] = []


app = FastAPI()

origins = [
    # "http:localhost",
    # "http:localhost:8080",
    # "http:localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


def sentence_sort_function(sent_1, sent_2) -> bool:
    if sent_1.paragraph == sent_2.paragraph:
        return sent_1.index - sent_2.index
    return sent_1.paragraph - sent_2.paragraph


@app.post("/highlight/", response_model=HighlightResults)
def read_item(highlight_request: HighlightRequest):
    if highlight_request.model == "textrank":
        sentences, _, _ = summarize_textrank(highlight_request.text)
    elif highlight_request.model == "use-xling":
        sentences, _, _ = summarize_xling(highlight_request.text)
    elif highlight_request.model == "laser":
        sentences, _, _ = summarize_laser(highlight_request.text)
    else:
        return HighlightResults(
            success=False,
            message=f"'{highlight_request.model}' is not supported."
        )
    # Sort sentences
    sentences = sorted(
        sentences, key=functools.cmp_to_key(sentence_sort_function))
    # Create sentence obj
    sentence_objs = [
        Sentence(paragraph=x.paragraph, score=x.score,
                 text=x.text, index=x.index)
        for x in sentences
    ]
    # print(sentence_objs)
    return HighlightResults(
        success=True,
        sentences=sentence_objs
    )


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
