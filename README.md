# TextRank Demo

A simple website demonstrating TextRank's extractive summarization capability. Currently supports English and Chinese.

## Usage

* This project uses Starlette (a lightweight ASGI framework/toolkit), so **Python 3.6+** is required.

* Install dependencies by running `pip install -r requirements`.

* Start the demo server by running `python demo.py`, and then visit `http://localhost:8000` in your browser.

(Depending on your Python setup, you might need to replace `pip` with `pip3`, and `python` with `python3`.)

WARNING: At the current state, the backend does almost to none input value validation. Please do not anticipate it to have production quality.

## Languages supported

### English

Demo: **[A static snapshot](https://publicb2.ceshine.net/file/ceshine-public/misc/textrank_demo.html) with an example from Wikipedia.**

This largely depends on language preprocessing functions and classes from [summanlp/textrank](https://github.com/summanlp/textrank). This project just exposes some of their internal data.

Accoring to [summanlp/textrank](https://github.com/summanlp/textrank), you can install an extra library to improve keyword extraction:

> For a better performance of keyword extraction, install Pattern.

From a quick glance at the source code, it seems to be using Pattern (if available) to do some POS tag filtering.

### Chinese

Demo: **[A static snapshot](https://publicb2.ceshine.net/file/ceshine-public/misc/textrank_demo_zh.html) with an example from a news article.**

This project uses [Baidu's free NLP API](https://cloud.baidu.com/product/nlp) to do word segmentation and POS tagging. You need to create an account there and set the following environment variables:

* BAIDU_APP_ID
* BAIDU_APP_KEY
* BAIDU_SECRET_KEY

You can of course use other offline NLP tools instead. Please refer to `test_text_cleaning_zh.py` for information on the data structures expected by the main function.

Traditional Chinese will be converted to Simplified Chinese due to the restrictions of Baidu API.

## Snapshots

### English

![highlights](imgs/snapshot_texts.png)

![sentence network](imgs/snapshot_sentence_network.png)

![word network](imgs/snapshot_word_network.png)

### Chinese

![highlights](imgs/snapshot_zh_texts.png)

![sentence network](imgs/snapshot_zh_sentence_network.png)

![word network](imgs/snapshot_zh_word_network.png)