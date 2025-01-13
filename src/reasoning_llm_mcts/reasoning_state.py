import math
from dataclasses import dataclass, field
from functools import cached_property

from openai import AsyncOpenAI

from reasoning_llm_mcts.mcts import State


@dataclass
class ReasoningState(State):
    api_base_url: str
    max_total_tokens: int
    max_new_tokens_delta: int

    root_prompt: str | None = None  # for the root state only

    text_delta: str = ""
    num_tokens_delta: int = 0

    confidence_score: float = 0.0
    parent_state: State | None = None
    top_logprobs_num: int = 5
    child_state_candidates: list = field(default_factory=list)

    def __post_init__(self):
        if self.parent_state is None:  # Root
            assert self.root_prompt is not None
            assert self.text_delta == ""
            assert self.num_tokens_delta == 0
        if self.parent_state is not None:
            assert self.root_prompt is None
        if self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        if self.max_new_tokens_delta <= 0:
            raise ValueError("max_new_tokens_delta must be positive")
        if self.max_new_tokens_delta > self.max_total_tokens:
            raise ValueError("max_new_tokens_delta cannot exceed max_total_tokens")

    async def expand(self, expand_num: int) -> State:
        """Generate a new state by expanding the current state"""
        print("expand", self.total_prompt)
        print(f"{self.child_state_candidates=}")
        assert expand_num <= len(self.child_state_candidates), "here"
        assert not self.is_terminal()
        top_candidates = sorted(
            self.child_state_candidates,
            reverse=True,
            key=lambda candidate: candidate[0],
        )[:expand_num]
        responses = [top_candidate[1] for top_candidate in top_candidates]

        child_states = []
        for response in responses:
            logprobs = response.choices[0].logprobs

            # Use only first `self.max_delta_new_tokens` tokens
            token_logprobs = logprobs.token_logprobs[: self.max_new_tokens_delta]

            # top_logprobs may contain `self.top_logprobs_num + 1` elements.
            # So take the first `self.top_logprobs_num` explicitly.
            top_logprobs = [
                sorted(top_lps.values())[: self.top_logprobs_num]
                for top_lps in logprobs.top_logprobs[: self.max_new_tokens_delta]
            ]

            former_confidence_score = calc_confidence_score(
                token_logprobs=token_logprobs,
                top_logprobs=top_logprobs,
                debug_logprobs=logprobs,
            )
            print(f"{former_confidence_score=}")
            token_delta_num = len(token_logprobs)
            child_confidence_score = (
                self.total_new_token_num * self.confidence_score
                + token_delta_num * former_confidence_score
            ) / (self.total_new_token_num + token_delta_num)
            print(f"{child_confidence_score=}")

            # Construct text_delta from the first max_delta_new_tokens tokens
            tokens = logprobs.tokens[: self.max_new_tokens_delta]
            # print(f"{tokens=}")
            # text_delta = "".join(bytes(t.bytes).decode("utf-8") for t in tokens)
            text_delta = "".join(tokens)
            print(f"{text_delta=}")

            child_states.append(
                ReasoningState(
                    api_base_url=self.api_base_url,
                    max_total_tokens=self.max_total_tokens,
                    max_new_tokens_delta=self.max_new_tokens_delta,
                    text_delta=text_delta,
                    num_tokens_delta=token_delta_num,
                    confidence_score=child_confidence_score,
                    parent_state=self,
                    top_logprobs_num=self.top_logprobs_num,
                )
            )
            # print(f"{child_states}")
        return child_states

    async def evaluate(self) -> float:
        """Evaluate the current state and return a value"""
        print(f"{self.total_prompt=}")
        if self.is_terminal():
            return self.confidence_score
        max_new_tokens = max(self.max_total_tokens - self.total_new_token_num, 0)
        while True:
            response = await AsyncOpenAI(base_url=self.api_base_url).completions.create(
                model="swallow-mx-4bit",  # TODO
                prompt=self.total_prompt,
                logprobs=self.top_logprobs_num,
                max_tokens=max_new_tokens,
                stop=["\n\n"],
            )
            if response.choices[0].text != "":
                break

        print(response.choices[0].text)
        logprobs = response.choices[0].logprobs

        print(response.choices[0].finish_reason)

        if logprobs is None:
            print(f"{response=}")
            print(f"{response.choices[0].text=}")
        # top_logprobs may contain self.top_logprobs + 1 elements
        top_logprobs = [
            sorted(top_lps.values())[: self.top_logprobs_num]
            for top_lps in logprobs.top_logprobs
        ]
        # print(f"{top_logprobs=}")

        rest_confidence_score = calc_confidence_score(
            token_logprobs=logprobs.token_logprobs,
            top_logprobs=top_logprobs,
            debug_logprobs=logprobs,
        )
        print(f"{rest_confidence_score=}")

        rest_token_num = len(logprobs.token_logprobs)
        print(f"{rest_token_num=}")

        combined_confidence_score = (
            self.total_new_token_num * self.confidence_score
            + rest_token_num * rest_confidence_score
        ) / (self.total_new_token_num + rest_token_num)
        print(f"{combined_confidence_score=}")

        # Update candidates for expanding
        self.child_state_candidates.append((combined_confidence_score, response))

        return 0.1 * combined_confidence_score

    def is_terminal(self) -> bool:
        if self.parent_state is None:
            return False  # root state is not terminal
        return (self.max_total_tokens <= self.total_new_token_num) or (
            self.num_tokens_delta < self.max_new_tokens_delta
        )

    def __str__(self) -> str:
        return f"{self.text_delta}"

    @cached_property
    def total_new_token_num(self) -> int:
        if self.parent_state is None:
            assert self.num_tokens_delta == 0, "here2"
            return self.num_tokens_delta
        return self.parent_state.total_new_token_num + self.num_tokens_delta

    @cached_property
    def total_prompt(self) -> str:
        if self.parent_state is None:
            return self.root_prompt
        return self.parent_state.total_prompt + self.text_delta


def calc_confidence_score(
    token_logprobs: list[float],
    top_logprobs: list[list[float]],
    debug_logprobs,
) -> float:
    assert len(token_logprobs) == len(top_logprobs), "here3"
    prob_sum = 0.0
    count = 0
    for i, (token_lp, top_lps) in enumerate(zip(token_logprobs, top_logprobs)):
        # print(f"{token_lp=}, {top_lps=}")
        # print(f"{debug_logprobs.top_logprobs[i]=}")
        # TODO
        # assert len(top_lps) == len(top_logprobs[0])

        # This is actual computation
        # ci_sum += math.exp(token_lp) / sum([math.exp(top_lps) for top_lps in top_lps])

        max_lp = max(top_lps + [token_lp])
        # print(f"{max_lp=}")
        prob = math.exp(token_lp - max_lp) / sum(
            math.exp(lp - max_lp) for lp in top_lps
        )
        if len(top_lps) != 5:  # TODO
            continue
        # print(f"{prob=}")
        prob_sum += prob
        # print(f"{prob_sum=}, {len(token_logprobs)=}")
        count += 1
        assert prob_sum <= count

    confidence_score = prob_sum / count
    print(f"{confidence_score=}")
    return confidence_score
