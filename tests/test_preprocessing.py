import pandas as pd

from ml.dataset.preprocessing import tokenize_text, transform_text_with_embedding
import pytest

@pytest.mark.parametrize(
    argnames="text, remove_stop_words",
    argvalues=[
        ("can i do that for you? yeep???", True),
        ("labubu taicun yep hello", True)
    ]
)
def test_tokenize_text(text: str, remove_stop_words: bool):
    tokens = tokenize_text(text=text, remove_stop_words=remove_stop_words)
    assert tokens


@pytest.mark.parametrize(
    argnames="text",
    argvalues=[
        "can i do that for you? yeep???",
        "labubu taicun yep hello"
    ]
)
def test_embedding(text: str):
    data = pd.Series([text])
    tokens = transform_text_with_embedding(sentence=data)
    assert len(tokens) > 0