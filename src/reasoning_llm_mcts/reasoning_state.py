import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import openai

from reasoning_llm_mcts.mcts import State


@dataclass
class ReasoningState(State):
    openai_client: openai.AsyncOpenAI
    tokenizer: Any
    max_total_tokens: int
    max_delta_new_tokens: int
    text_delta: str
    token_delta_num: int
    confidence_score: float
    parent_state: State | None = None
    top_logprobs_num: int = 5
    child_state_candidates: list = field(default_factory=list)

    def __post_init__(self):
        if self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        if self.max_delta_new_tokens <= 0:
            raise ValueError("max_delta_new_tokens must be positive")
        if self.max_delta_new_tokens > self.max_total_tokens:
            raise ValueError("max_delta_new_tokens cannot exceed max_total_tokens")

    async def expand(self, expand_num: int) -> State:
        """Generate a new state by expanding the current state"""
        assert expand_num <= len(self.child_state_candidates)
        responses = list(
            zip(*(sorted(self.child_state_candidates, reverse=True)[:expand_num]))
        )[1]

        child_states = []
        for response in responses:
            logprobs = response.choices[0].logprobs

            # Use only first `self.max_delta_new_tokens` tokens
            token_logprobs = logprobs.token_logprobs[: self.max_delta_new_tokens]

            # top_logprobs may contain `self.top_logprobs_num + 1` elements.
            # So take the first `self.top_logprobs_num` explicitly.
            top_logprobs = [
                sorted(top_lps.values())[: self.top_logprobs_num]
                for top_lps in logprobs.top_logprobs[: self.max_delta_new_tokens]
            ]

            former_confidence_score = calc_confidence_score(
                token_logprobs=token_logprobs, top_logprobs=top_logprobs
            )
            token_delta_num = len(token_logprobs)
            child_confidence_score = (
                self.total_token_num * self.confidence_score
                + token_delta_num * former_confidence_score
            ) / (self.total_token_num + token_delta_num)

            # Construct text_delta from the first max_delta_new_tokens tokens
            tokens = logprobs.tokens[: self.max_delta_new_tokens]
            text_delta = "".join(bytes(t.bytes).decode("utf-8") for t in tokens)

            child_states.append(
                ReasoningState(
                    openai_client=self.openai_client,
                    tokenizer=self.tokenizer,
                    max_total_tokens=self.max_total_tokens,
                    max_delta_new_tokens=self.max_delta_new_tokens,
                    text_delta=text_delta,
                    token_delta_num=token_delta_num,
                    confidence_score=child_confidence_score,
                    parent_state=self,
                    top_logprobs_num=self.top_logprobs_num,
                )
            )
        return child_states

    async def evaluate(self) -> float:
        """Evaluate the current state and return a value"""
        max_new_tokens = max(self.max_total_tokens - self.total_token_num, 0)
        if max_new_tokens == 0:
            return self.confidence_score
        response = await self.openai_client.completions.create(
            prompt=self.total_prompt,
            logprobs=self.top_logprobs_num,
            max_tokens=max_new_tokens,
        )
        logprobs = response.choices[0].logprobs

        # top_logprobs may contain self.top_logprobs + 1 elements
        top_logprobs = [
            sorted(top_lps.values())[: self.top_logprobs_num]
            for top_lps in logprobs.top_logprobs
        ]

        rest_confidence_score = calc_confidence_score(
            token_logprobs=logprobs.token_logprobs, top_logprobs=top_logprobs
        )

        rest_token_num = len(logprobs.token_logprobs)

        confidence_score = (
            self.total_token_num * self.confidence_score
            + rest_token_num * rest_confidence_score
        ) / (self.total_token_num + rest_token_num)

        # Update candidates for expanding
        self.child_state_candidates.append((confidence_score, response))

        return confidence_score

    def __str__(self) -> str:
        return f"{self.text_delta}"

    @cached_property
    def total_token_num(self) -> int:
        if self.parent_state is None:
            return len(self.tokenizer.tokenize(self.text_delta).input_ids)
        return self.parent_state.total_token_num + self.token_delta_num

    @cached_property
    def total_prompt(self) -> str:
        if self.parent_state is None:
            return self.text_delta
        return self.parent_state.total_prompt + self.text_delta


def calc_confidence_score(
    token_logprobs: list[float], top_logprobs: list[list[float]]
) -> float:
    assert len(token_logprobs) == len(top_logprobs)
    ci_sum = 0.0
    for token_lp, top_lps in zip(token_logprobs, top_logprobs):
        assert len(top_lps) == len(top_logprobs[0])

        # This is actual computation
        # ci_sum += math.exp(token_lp) / sum([math.exp(top_lps) for top_lps in top_lps])

        max_lp = max(top_lps)
        ci_sum += (len(top_lps) * math.exp(token_lp - max_lp)) / sum(
            math.exp(lp - max_lp) for lp in top_lps
        )

    confidence_score = ci_sum / len(token_logprobs)
    return confidence_score
