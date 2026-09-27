"""Plan §24 Truncate Test on the real klue/roberta-base tokenizer."""

import pytest
from transformers import AutoTokenizer

from src.data.truncate import HEAD_TOKENS, MAX_LENGTH, text_token_budget, truncate_head_tail


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("klue/roberta-base")


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def test_budget_comes_from_tokenizer_not_a_constant(tokenizer):
    assert text_token_budget(tokenizer) == MAX_LENGTH - tokenizer.num_special_tokens_to_add(pair=False)


def test_short_text_is_returned_unchanged(tokenizer):
    text = "서울중앙지검입니다. 본인 명의 계좌가 발견됐습니다."
    assert truncate_head_tail(text, tokenizer) == text


def test_long_text_fits_and_keeps_head_and_tail(tokenizer):
    sentence = "고객님 계좌로 삼백만 원을 지금 송금해 주셔야 합니다. "
    long_text = (sentence * 200).strip()  # far beyond 512 tokens
    assert token_count(tokenizer, long_text) > text_token_budget(tokenizer)

    result = truncate_head_tail(long_text, tokenizer)
    budget = text_token_budget(tokenizer)

    # Fits the model once special tokens are added. Re-tokenizing the joined
    # string can differ by a token or two at the seam, so allow small slack below.
    with_special = len(tokenizer(result)["input_ids"])
    assert with_special <= MAX_LENGTH
    assert with_special >= MAX_LENGTH - 4

    # Head: the first HEAD_TOKENS tokens of the original survive verbatim.
    head_end = tokenizer(long_text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"][HEAD_TOKENS - 1][1]
    assert result.startswith(long_text[:head_end])
    # Tail: the original ending survives verbatim.
    assert result.endswith(long_text[-60:])
    assert budget == HEAD_TOKENS + (budget - HEAD_TOKENS)


def test_output_is_a_string(tokenizer):
    assert isinstance(truncate_head_tail("짧은 통화", tokenizer), str)
