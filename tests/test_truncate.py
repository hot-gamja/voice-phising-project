"""계획서 §24 실제 klue/roberta-base tokenizer에 대한 Truncate 테스트."""

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
    long_text = (sentence * 200).strip()  # 512 토큰을 훨씬 초과
    assert token_count(tokenizer, long_text) > text_token_budget(tokenizer)

    result = truncate_head_tail(long_text, tokenizer)
    budget = text_token_budget(tokenizer)

    # 특수 토큰이 추가되면 모델에 맞는다. 이어붙인 문자열을 다시 토큰화하면
    # 이어지는 지점에서 토큰 한두 개 차이가 날 수 있어, 아래에 약간의 여유를 둔다.
    with_special = len(tokenizer(result)["input_ids"])
    assert with_special <= MAX_LENGTH
    assert with_special >= MAX_LENGTH - 4

    # Head: 원문의 앞쪽 HEAD_TOKENS 토큰이 그대로 보존된다.
    head_end = tokenizer(long_text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"][HEAD_TOKENS - 1][1]
    assert result.startswith(long_text[:head_end])
    # Tail: 원문의 끝부분이 그대로 보존된다.
    assert result.endswith(long_text[-60:])
    assert budget == HEAD_TOKENS + (budget - HEAD_TOKENS)


def test_output_is_a_string(tokenizer):
    assert isinstance(truncate_head_tail("짧은 통화", tokenizer), str)
