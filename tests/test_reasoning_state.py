import math
from unittest.mock import AsyncMock, Mock, patch

import pytest

from reasoning_llm_mcts.mcts import State
from reasoning_llm_mcts.reasoning_state import ReasoningState, calc_confidence_score


@pytest.fixture
def mock_openai():
    with patch("reasoning_llm_mcts.reasoning_state.AsyncOpenAI") as mock:
        mock_client = AsyncMock()
        mock.return_value = mock_client
        mock_client.completions.create = AsyncMock()
        yield mock


@pytest.fixture
def root_state():
    return ReasoningState(
        api_base_url="http://test.api",
        max_total_tokens=100,
        max_new_tokens_delta=50,
        root_prompt="Root prompt",
    )


@pytest.fixture
def child_state(root_state):
    return ReasoningState(
        api_base_url="http://test.api",
        max_total_tokens=100,
        max_new_tokens_delta=50,
        text_delta="Initial text",
        parent_state=root_state,
    )


def test_init_validation():
    # Test max_total_tokens validation
    with pytest.raises(ValueError, match="max_total_tokens must be positive"):
        ReasoningState(
            api_base_url="http://test.api",
            max_total_tokens=0,
            max_new_tokens_delta=10,
            root_prompt="test",
        )

    # Test max_new_tokens_delta validation
    with pytest.raises(ValueError, match="max_new_tokens_delta must be positive"):
        ReasoningState(
            api_base_url="http://test.api",
            max_total_tokens=100,
            max_new_tokens_delta=0,
            root_prompt="test",
        )

    # Test max_new_tokens_delta cannot exceed max_total_tokens
    with pytest.raises(
        ValueError, match="max_new_tokens_delta cannot exceed max_total_tokens"
    ):
        ReasoningState(
            api_base_url="http://test.api",
            max_total_tokens=10,
            max_new_tokens_delta=20,
            root_prompt="test",
        )

    # Test root state initialization
    root_state = ReasoningState(
        api_base_url="http://test.api",
        max_total_tokens=100,
        max_new_tokens_delta=50,
        root_prompt="Root prompt",
    )
    assert root_state.parent_state is None
    assert root_state.text_delta == ""
    assert root_state.num_tokens_delta == 0

    # Test root_prompt is required for root state
    with pytest.raises(AssertionError):
        ReasoningState(
            api_base_url="http://test.api",
            max_total_tokens=100,
            max_new_tokens_delta=50,
        )

    # Test non-root state must not have root_prompt
    with pytest.raises(AssertionError):
        ReasoningState(
            api_base_url="http://test.api",
            max_total_tokens=100,
            max_new_tokens_delta=50,
            root_prompt="Root prompt",
            parent_state=root_state,
        )


@pytest.mark.asyncio
async def test_expand(child_state):
    # Create mock token objects with bytes attribute
    token1_bytes = "Hello".encode("utf-8")
    token2_bytes = " world".encode("utf-8")
    mock_token1 = Mock(bytes=token1_bytes)
    mock_token2 = Mock(bytes=token2_bytes)

    # Mock child state candidates with tokens that have bytes
    mock_response1 = Mock()
    mock_response1.choices = [
        Mock(
            logprobs=Mock(
                token_logprobs=[-1.0, -2.0],
                top_logprobs=[{"a": -1.0, "b": -2.0}, {"c": -1.5, "d": -2.5}],
                tokens=[mock_token1, mock_token2],
                bytes=[token1_bytes, token2_bytes],
            )
        )
    ]

    mock_response2 = Mock()
    mock_response2.choices = [
        Mock(
            logprobs=Mock(
                token_logprobs=[-1.2, -2.2],
                top_logprobs=[{"e": -1.2, "f": -2.2}, {"g": -1.7, "h": -2.7}],
                tokens=[mock_token1, mock_token2],
                bytes=[token1_bytes, token2_bytes],
            )
        )
    ]

    child_state.child_state_candidates = [(0.9, mock_response1), (0.8, mock_response2)]

    child_states = await child_state.expand(1)
    assert len(child_states) == 1
    assert isinstance(child_states[0], ReasoningState)
    assert child_states[0].parent_state == child_state
    assert child_states[0].text_delta == "Hello world"

    # Test expansion with confidence score calculation
    child_states = await child_state.expand(2)
    assert len(child_states) == 2
    for state in child_states:
        assert isinstance(state, ReasoningState)
        assert state.parent_state == child_state
        assert state.text_delta == "Hello world"
        assert 0 <= state.confidence_score <= 1


@pytest.mark.asyncio
async def test_evaluate(child_state, mock_openai):
    # Mock OpenAI response
    mock_response = Mock()
    mock_response.choices = [
        Mock(
            logprobs=Mock(
                token_logprobs=[-1.0, -2.0],
                top_logprobs=[{"a": -1.0, "b": -2.0}, {"c": -1.5, "d": -2.5}],
            )
        )
    ]

    mock_client = mock_openai.return_value
    mock_client.completions.create.return_value = mock_response

    # Test evaluate with max_new_tokens > 0
    score = await child_state.evaluate()
    assert isinstance(score, float)
    assert 0 <= score <= 1

    # Verify OpenAI client was called with correct parameters
    mock_client.completions.create.assert_called_once()
    call_kwargs = mock_client.completions.create.call_args.kwargs
    assert call_kwargs["prompt"] == child_state.total_prompt
    assert call_kwargs["logprobs"] == child_state.top_logprobs_num
    assert (
        call_kwargs["max_tokens"]
        == child_state.max_total_tokens - child_state.total_new_token_num
    )

    # Test confidence score calculation and candidates update
    assert len(child_state.child_state_candidates) == 1
    assert child_state.child_state_candidates[0][0] == score

    # Test evaluate when max_new_tokens = 0
    root_state = ReasoningState(
        api_base_url="http://test.api",
        max_total_tokens=5,
        max_new_tokens_delta=5,
        root_prompt="Root",
    )
    max_tokens_state = ReasoningState(
        api_base_url="http://test.api",
        max_total_tokens=5,
        max_new_tokens_delta=5,
        text_delta="Long text that exceeds max tokens",
        num_tokens_delta=10,
        parent_state=root_state,
    )
    score = await max_tokens_state.evaluate()
    assert score == max_tokens_state.confidence_score


def test_calc_confidence_score():
    # Test basic case with probability normalization
    token_logprobs = [-1.0, -2.0]
    top_logprobs = [[-1.0, -2.0], [-1.5, -2.5]]
    score = calc_confidence_score(token_logprobs, top_logprobs)
    assert isinstance(score, float)
    assert 0 <= score <= 1

    # Test numerical stability with large negative values
    token_logprobs = [-1000.0, -2000.0]
    top_logprobs = [[-1000.0, -2000.0], [-1500.0, -2500.0]]
    score = calc_confidence_score(token_logprobs, top_logprobs)
    assert isinstance(score, float)
    assert 0 <= score <= 1

    # Test with equal probabilities
    token_logprobs = [-1.0, -1.0]
    top_logprobs = [[-1.0, -1.0], [-1.0, -1.0]]
    score = calc_confidence_score(token_logprobs, top_logprobs)
    assert math.isclose(
        score, 0.5, rel_tol=1e-9
    )  # Equal probabilities should result in 0.5

    # Test with very different scales
    token_logprobs = [-5.0, -10.0]
    top_logprobs = [[-5.0, -1000.0], [-10.0, -2000.0]]
    score = calc_confidence_score(token_logprobs, top_logprobs)
    assert score > 0.99  # Should be close to 1 due to large differences

    # Test with inconsistent lengths
    with pytest.raises(AssertionError):
        calc_confidence_score([-1.0], [[-1.0, -2.0], [-1.5, -2.5]])

    # Test with different number of top logprobs
    with pytest.raises(AssertionError):
        calc_confidence_score([-1.0, -2.0], [[-1.0, -2.0, -3.0], [-1.5, -2.5]])


@pytest.mark.asyncio
async def test_total_new_token_num():
    # Test root state
    root_state = ReasoningState(
        api_base_url="http://test.api",
        max_total_tokens=100,
        max_new_tokens_delta=50,
        root_prompt="Root prompt",
    )
    assert root_state.total_new_token_num == 0

    # Test child state token accumulation
    child_state = ReasoningState(
        api_base_url="http://test.api",
        max_total_tokens=100,
        max_new_tokens_delta=50,
        text_delta="Child text",
        num_tokens_delta=5,
        parent_state=root_state,
    )
    assert child_state.total_new_token_num == 5

    # Test multiple levels of nesting
    grandchild_state = ReasoningState(
        api_base_url="http://test.api",
        max_total_tokens=100,
        max_new_tokens_delta=50,
        text_delta="Grandchild text",
        num_tokens_delta=3,
        parent_state=child_state,
    )
    assert grandchild_state.total_new_token_num == 8

    # Test cached property behavior
    assert grandchild_state.total_new_token_num == 8  # Should use cached value


@pytest.mark.asyncio
async def test_total_prompt():
    # Test root state
    root_state = ReasoningState(
        api_base_url="http://test.api",
        max_total_tokens=100,
        max_new_tokens_delta=50,
        root_prompt="Root prompt",
    )
    assert root_state.total_prompt == "Root prompt"

    # Test child state prompt concatenation
    child_state = ReasoningState(
        api_base_url="http://test.api",
        max_total_tokens=100,
        max_new_tokens_delta=50,
        text_delta="Child text",
        parent_state=root_state,
    )
    assert child_state.total_prompt == "Root promptChild text"

    # Test multiple levels of prompt concatenation
    grandchild_state = ReasoningState(
        api_base_url="http://test.api",
        max_total_tokens=100,
        max_new_tokens_delta=50,
        text_delta="Grandchild text",
        parent_state=child_state,
    )
    assert grandchild_state.total_prompt == "Root promptChild textGrandchild text"

    # Test cached property behavior
    assert (
        grandchild_state.total_prompt == "Root promptChild textGrandchild text"
    )  # Should use cached value


def test_str_representation(child_state):
    assert str(child_state) == "Initial text"
