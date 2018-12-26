from typing import List, Dict, Any
import pytest

from text_cleaning_ja import clean_and_cut_sentences, clean_and_cut_words, get_tokens

ASAHI_1 = (
    "クリスマスの金融市場は米国発の世界株安となった。"
    "２４日の米ダウ工業株平均の急落を受け、２５日の日経平均株価は年内２度目の１千円超の下げ幅となり、１０月初めのバブル崩壊後の最高値圏から５千円超も下落。\r\n"
    "米国が引っ張った世界の景気拡大局面が終わりつつあるとの見方が広がる中、トランプ米政権は不安定さを増し、世界経済を振り回している。"
)


def test_get_token():
    text = "日本列島における人類の歴史は、次第に人が住み始めた約10万年前以前ないし約3.5万年前に始まったとされる。"
    results = get_tokens(text)
    # for i, row in enumerate(results):
    #     print(i, row)
    assert results[0].content == "日本"
    assert results[0].postag == "noun"
    assert results[0].postag_raw == "名詞"
    assert results[3].content == "おけ"
    assert results[3].postag == "verb"
    assert results[3].postag_raw == "動詞"
    assert results[6].content == "の"
    assert results[6].postag == "part"
    assert results[6].postag_raw == "助詞"
    assert results[-1].content == "。"
    assert results[-1].postag == "punc"
    assert results[-1].postag_raw == "補助記号"
    assert len(results) == 40
    results = get_tokens(ASAHI_1)
    for i, row in enumerate(results):
        print(i, row)
    assert results[0].content == "クリスマス"
    assert results[0].postag == "noun"
    assert results[0].postag_raw == "名詞"
    assert results[-3].content == "て"
    assert results[-3].postag == "part"
    assert results[-3].postag_raw == "助詞"


def test_clean_and_cut_sentences():
    results = clean_and_cut_sentences(ASAHI_1, pos_tags=("noun", "verb"))
    assert len(results) == 3
    assert results[0].text == "クリスマスの金融市場は米国発の世界株安となった。"
    assert results[0].token == "クリスマス 金融 市場 米国 世界 株安 なっ"
    assert results[0].index == 0
    assert results[0].paragraph == 0
    assert results[1].text == "24日の米ダウ工業株平均の急落を受け、25日の日経平均株価は年内2度目の1千円超の下げ幅となり、10月初めのバブル崩壊後の最高値圏から5千円超も下落。"
    assert results[1].token == "米 ダウ 工業 株 平均 急落 受け 日経 平均 株価 年内 度 1千 円 下げ幅 なり 月 初め バブル 崩壊 高値 5千 円 下落"
    assert results[1].index == 1
    assert results[1].paragraph == 0
    assert results[2].text == "米国が引っ張った世界の景気拡大局面が終わりつつあるとの見方が広がる中、トランプ米政権は不安定さを増し、世界経済を振り回している。"
    assert results[2].token == "米国 引っ張っ 世界 景気 拡大 局面 終わり ある 見 広がる 中 トランプ 政権 安定 増し 世界 経済 振り回し いる"
    assert results[2].index == 0
    assert results[2].paragraph == 1


def test_cut_words_n():
    text = "日本列島における人類の歴史は、次第に人が住み始めた約10万年前以前ないし約3.5万年前に始まったとされる。"
    results = clean_and_cut_words(text, ("noun",), verbose=False)
    assert len(results) == 14
    # third word
    assert results[2].index == 2
    assert results[2].paragraph == 0
    assert results[2].text == "人類"
    assert results[2].token == "人類"
    assert results[4].index == 4
    assert results[4].paragraph == 0
    assert results[4].text == "次第"
    assert results[4].token == "次第"


def test_cut_words_vadj():
    text = "孫は佐賀県鳥栖市の朝鮮人集落で幼少期を過ごし、差別も経験する。豚や羊と一緒に生活する非常に貧しく不衛生な場所であったが"
    # results = get_tokens(text)
    # for i, row in enumerate(results):
    #     print(i, row)
    # assert False
    results = clean_and_cut_words(text, ("verb", "adj"), verbose=False)
    # print([x.text for x in results])
    assert len(results) == 4
    # third word
    assert results[0].index == 0
    assert results[0].paragraph == 0
    assert results[0].text == "過ごし"
    assert results[0].token == "過ごし"
    assert results[2].index == 2
    assert results[2].paragraph == 0
    assert results[2].text == "貧しく"
    assert results[2].token == "貧しく"
