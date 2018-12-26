from typing import List, Dict, Any
import pytest

from text_cleaning_zh import insert_unit, cut_sentences, cut_words, clean_text


@pytest.fixture
def sample_tagged_items() -> List[Dict[str, Any]]:
    return [
        {
            'item': '华为',
            'basic_words': ['华为'],
            'tag': 'noun'
        },
        {
            'item': '高管',
            'basic_words': ['高管'],
            'byte_length': 4,
            'tag': 'noun'
        },
        {
            'item': '、',
            'basic_words': ['、'],
            'tag': 'punc'
        },
        {
            'item': '创办人',
            'basic_words': ['创办', '人'],
            'tag': 'noun'
        },
        {
            'item': '任正非',
            'basic_words': ['任正非'],
            'tag': 'noun'
        },
        {
            'item': '之',
            'basic_words': ['之'],
            'tag': 'part'
        },
        {
            'item': '女',
            'basic_words': ['女'],
            'tag': 'noun'
        },
        {
            'item': '。',
            'basic_words': ['。'],
            'tag': 'punc'
        },
        {
            'item': '贸易战',
            'basic_words': ['贸易', '战'],
            'tag': 'noun'
        },
        {
            'item': '恶化',
            'basic_words': ['恶化'],
            'tag': 'verb'
        },
        {
            'item': '\n',
            'basic_words': ['\n'],
            'tag': 'noun'
        },
        {
            'item': '尽管',
            'basic_words': ['尽管'],
            'tag': 'conj'
        },
        {
            'item': '孟晚舟',
            'basic_words': ['孟晚舟'],
            'tag': 'noun'
        },
        {
            'item': '被捕',
            'basic_words': ['被捕'],
            'tag': 'verb'
        },
        {
            'item': '是',
            'basic_words': ['是'],
            'tag': 'verb'
        },
        {
            'item': '因',
            'basic_words': ['因'],
            'tag': 'prep'
        },
        {
            'item': '涉嫌',
            'basic_words': ['涉嫌'],
            'tag': 'verb'
        },
        {
            'item': '违反',
            'basic_words': ['违反'],
            'tag': 'verb'
        },
        {
            'item': '。',
            'basic_words': ['。'],
            'tag': 'punc'
        }
    ]


def test_insert_unit():
    tmp = []
    insert_unit(
        tmp,
        ["美", "中", "两国", "正在", "就", "贸易",
         "争端", "展开", "关键", "谈判", "之时", "，",
         "围绕", "孟晚舟", "和", "华为", "的", "这一",
         "事件", "引起", "市场", "的", "波动"],
        ["美", "中", "两 国", "正 在", "贸易",
         "争端", "展开", "谈判",
         "围绕", "孟晚舟", "华为", "事件", "引起", "市场", "波动"], 0, 0
    )
    assert len(tmp) == 1
    assert tmp[0].text == "美中两国正在就贸易争端展开关键谈判之时，围绕孟晚舟和华为的这一事件引起市场的波动"
    assert tmp[0].token == "美 中 两 国 正 在 贸易 争端 展开 谈判 围绕 孟晚舟 华为 事件 引起 市场 波动"
    assert tmp[0].index == 0
    assert tmp[0].paragraph == 0
    insert_unit(
        tmp,
        ["中国", "国家", "副主席", "王岐山", "表示", "，", "中国",
         "正", "经历", "险峻", "的", "国际", "环境"],
        ["中国", "国家", "副 主席", "王岐山", "表示", "中国", "经历", "国际", "环境"], 0, 1
    )
    assert len(tmp) == 2
    assert tmp[1].text == "中国国家副主席王岐山表示，中国正经历险峻的国际环境"
    assert tmp[1].token == "中国 国家 副 主席 王岐山 表示 中国 经历 国际 环境"
    assert tmp[1].index == 1
    assert tmp[1].paragraph == 0
    insert_unit(
        tmp,
        ["要", "顺应", "历史大势", "、", "保持", "战略", "定力"],
        ["顺应", "历史 大势", "保持", "战略", "定力"], 1, 0
    )
    assert len(tmp) == 3
    assert tmp[2].text == "要顺应历史大势、保持战略定力"
    assert tmp[2].token == "顺应 历史 大势 保持 战略 定力"
    assert tmp[2].index == 0
    assert tmp[2].paragraph == 1


def test_cut_sentences_default(sample_tagged_items):
    results = cut_sentences(sample_tagged_items, "。！？；",
                            ("noun", "verb"), ("是",))
    assert len(results) == 3
    # first sentence
    assert results[0].index == 0
    assert results[0].paragraph == 0
    assert results[0].text == "华为高管、创办人任正非之女。"
    assert results[0].token == "华为 高管 创办 人 任正非 女"
    # second sentence
    assert results[1].index == 1
    assert results[1].paragraph == 0
    assert results[1].text == "贸易战恶化"
    assert results[1].token == "贸易 战 恶化"
    # third sentence
    assert results[2].index == 0
    assert results[2].paragraph == 1
    assert results[2].text == "尽管孟晚舟被捕是因涉嫌违反。"
    assert results[2].token == "孟晚舟 被捕 涉嫌 违反"


def test_cut_words_nv(sample_tagged_items):
    results = cut_words(sample_tagged_items,
                        ("noun", "verb"), ("是", "有"))
    assert len(results) == 11
    # first word
    assert results[0].index == 0
    assert results[0].paragraph == 0
    assert results[0].text == "华为"
    assert results[0].token == "华为"


def test_cut_words_n(sample_tagged_items):
    results = cut_words(sample_tagged_items,
                        ("noun",), ("是", "有"))
    assert len(results) == 7
    # third word
    assert results[2].index == 2
    assert results[2].paragraph == 0
    assert results[2].text == "创办人"
    assert results[2].token == "创办人"


def test_clean_text():
    result = clean_text(
        "如果你能（其實，你應該）想像一下這種事情：\r\n一位美國   大型科技企業的高管去北京旅行時，因不明確的指控被拘留。")
    assert result == "如果你能（其實，你應該）想像一下這種事情：\n一位美國 大型科技企業的高管去北京旅行時，因不明確的指控被拘留。"
    result = clean_text(
        "如果你能\t\t（其實，你應該）\t\t想像一下這種事情：")
    assert result == "如果你能\t（其實，你應該）\t想像一下這種事情："
