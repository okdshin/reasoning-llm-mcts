import math
from unittest.mock import AsyncMock, Mock, patch

import pytest

from reasoning_llm_mcts.mcts import State
from reasoning_llm_mcts.reasoning_state import ReasoningState, calc_confidence_score


@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.tokenize.return_value.input_ids = [1, 2, 3]
    return tokenizer


@pytest.fixture
def mock_openai_client():
    return AsyncMock()


@pytest.fixture
def base_state(mock_openai_client, mock_tokenizer):
    return ReasoningState(
        openai_client=mock_openai_client,
        tokenizer=mock_tokenizer,
        max_total_tokens=100,
        max_delta_new_tokens=50,
        text_delta="Initial text",
        token_delta_num=3,
        confidence_score=0.8,
    )


def test_init_validation():
    with pytest.raises(ValueError, match="max_total_tokens must be positive"):
        ReasoningState(
            openai_client=Mock(),
            tokenizer=Mock(),
            max_total_tokens=0,
            max_delta_new_tokens=10,
            text_delta="",
            token_delta_num=0,
            confidence_score=0.0,
        )

    with pytest.raises(ValueError, match="max_delta_new_tokens must be positive"):
        ReasoningState(
            openai_client=Mock(),
            tokenizer=Mock(),
            max_total_tokens=100,
            max_delta_new_tokens=0,
            text_delta="",
            token_delta_num=0,
            confidence_score=0.0,
        )

    with pytest.raises(
        ValueError, match="max_delta_new_tokens cannot exceed max_total_tokens"
    ):
        ReasoningState(
            openai_client=Mock(),
            tokenizer=Mock(),
            max_total_tokens=10,
            max_delta_new_tokens=20,
            text_delta="",
            token_delta_num=0,
            confidence_score=0.0,
        )


@pytest.mark.asyncio
async def test_expand(base_state):
    # Create mock token objects with bytes attribute
    mock_token1 = Mock()
    mock_token1.bytes = b"Hello"
    mock_token2 = Mock()
    mock_token2.bytes = b" world"

    # Mock child state candidates
    mock_response1 = Mock()
    mock_response1.choices = [
        Mock(
            logprobs=Mock(
                token_logprobs=[-1.0, -2.0],
                top_logprobs=[{"a": -1.0, "b": -2.0}, {"c": -1.5, "d": -2.5}],
                tokens=[mock_token1, mock_token2],  # Add mock tokens
            ),
            text="Response 1",
        )
    ]

    mock_response2 = Mock()
    mock_response2.choices = [
        Mock(
            logprobs=Mock(
                token_logprobs=[-1.2, -2.2],
                top_logprobs=[{"e": -1.2, "f": -2.2}, {"g": -1.7, "h": -2.7}],
                tokens=[mock_token1, mock_token2],  # Add mock tokens
            ),
            text="Response 2",
        )
    ]

    base_state.child_state_candidates = [(0.9, mock_response1), (0.8, mock_response2)]

    child_states = await base_state.expand(1)
    assert len(child_states) == 1
    assert isinstance(child_states[0], ReasoningState)
    assert child_states[0].parent_state == base_state
    assert child_states[0].text_delta == "Hello world"  # Check the constructed text


@pytest.mark.asyncio
async def test_evaluate(base_state):
    mock_response = Mock()
    mock_response.choices = [
        Mock(
            logprobs=Mock(
                token_logprobs=[-1.0, -2.0],
                top_logprobs=[{"a": -1.0, "b": -2.0}, {"c": -1.5, "d": -2.5}],
            )
        )
    ]

    base_state.openai_client.completions.create.return_value = mock_response

    score = await base_state.evaluate()
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_calc_confidence_score():
    token_logprobs = [-1.0, -2.0]
    top_logprobs = [[-1.0, -2.0], [-1.5, -2.5]]

    score = calc_confidence_score(token_logprobs, top_logprobs)
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_total_token_num(base_state):
    assert base_state.total_token_num == 3

    child_state = ReasoningState(
        openai_client=base_state.openai_client,
        tokenizer=base_state.tokenizer,
        max_total_tokens=100,
        max_delta_new_tokens=50,
        text_delta="Child text",
        token_delta_num=2,
        confidence_score=0.7,
        parent_state=base_state,
    )

    assert child_state.total_token_num == 5


def test_total_prompt(base_state):
    assert base_state.total_prompt == "Initial text"

    child_state = ReasoningState(
        openai_client=base_state.openai_client,
        tokenizer=base_state.tokenizer,
        max_total_tokens=100,
        max_delta_new_tokens=50,
        text_delta="Child text",
        token_delta_num=2,
        confidence_score=0.7,
        parent_state=base_state,
    )

    assert child_state.total_prompt == "Initial textChild text"
