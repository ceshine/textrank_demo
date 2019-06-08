import tempfile
from pathlib import Path
from typing import List

from .lib.indexing import IndexCreate
from .embed import SentenceEncoder, EncodeFile
from .lib.text_processing import Token, BPEfastApply


def lines_to_index(lang: str, lines: List, model_path: str, bpe_code_path: str, use_cpu: bool = False):
    """Suitable for small amounts of data."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        target = str(Path(tmpdirname) / "source")
        with open(target, "w") as fout:
            fout.write("\n".join(lines))
        return text_file_to_index(
            lang, target, model_path, bpe_code_path, use_cpu
        )


def text_file_to_index(lang: str, input_path: str, model_path: str, bpe_code_path: str, use_cpu: bool = False):
    """Suitable for small amounts of data."""
    encoder = SentenceEncoder(
        model_path,
        max_sentences=None,
        max_tokens=10000,
        cpu=use_cpu)
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        Token(
            input_path,
            str(tmpdir / "token"),
            lang=lang,
            romanize=False,
            lower_case=True, gzip=False,
            verbose=True)
        BPEfastApply(
            str(tmpdir / "token"),
            str(tmpdir / "bpe"),
            bpe_code_path,
            verbose=True, over_write=True)
        EncodeFile(
            encoder,
            str(tmpdir / "bpe"),
            str(tmpdir / "enc"),
            verbose=True, over_write=True)
        data, index = IndexCreate(
            str(tmpdir / "enc"), 'FlatL2',
            verbose=True, save_index=False)
        return data, index
