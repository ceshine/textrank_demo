import os

from aip import AipNlp
from opencc import OpenCC

CC = OpenCC("t2s")
APP_ID = os.environ["BAIDU_APP_ID"]
API_KEY = os.environ["BAIDU_APP_KEY"]
SECRET_KEY = os.environ["BAIDU_SECRET_KEY"]


TAG_MAPPING = {
    "PER": "noun",
    "TIME": "noun",
    "ORG": "noun",
    "LOC": "noun",
    "w": "punc",   # 標點符號
    "a": "adj",
    "ad": "adj",
    "c": "conj",   # 連接詞
    "n": "noun",
    "f": "noun",   # 方位名词
    "s": "noun",   # 处所名词
    "t": "noun",   # 時間名词
    "nr": "noun",  # 人名
    "ns": "noun",  # 地名
    "nt": "noun",  # 機構團體名
    "nw": "noun",  # 作品名
    "v": "verb",   # 普通動詞
    "nz": "noun",  # 其他名詞
    "m": "quant",  # 數量詞
    "q": "quant",  # 量詞
    "vn": "noun",  # 名動詞
    "u": "part",   # 助詞
    "d": "adv",    # 副詞
    "vd": "adv",   # 動副詞
    "an": "noun",  # 名形詞
    "r": "pron",   # 代詞
    "p": "prep",   # 介詞
    "xc": "func"   # 其他虚词
}


def get_client():
    return AipNlp(APP_ID, API_KEY, SECRET_KEY)


def convert_ner_tags(ner_items):
    for entry in ner_items:
        if entry["pos"] != "":
            entry["tag"] = TAG_MAPPING[entry["pos"]]
        else:
            entry["tag"] = TAG_MAPPING[entry["ne"]]
    return ner_items


def ner_tags(text, verbose=False):
    res = get_client().lexer(CC.convert(text))
    if verbose:
        print([
            (x["item"], x["pos"], x["ne"])
            for x in res["items"]
        ])
    return convert_ner_tags(res["items"])
