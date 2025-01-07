import argparse
import asyncio
import json
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from reasoning_llm_mcts.mcts import MCTS
from reasoning_llm_mcts.reasoning_state import ReasoningState

app = FastAPI()


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    logprobs: Optional[int] = None
    stop: Optional[list[str]] = None


class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    try:
        initial_state = ReasoningState(
            openai_client=app.state.openai_client,
            tokenizer=app.state.tokenizer,
            max_total_tokens=request.max_tokens,
            max_delta_new_tokens=min(50, request.max_tokens),  # Adjust this if needed
            text_delta=request.prompt,
            token_delta_num=len(app.state.tokenizer.tokenize(request.prompt).input_ids),
            confidence_score=1.0,  # Initial state has perfect confidence
            top_logprobs_num=request.logprobs if request.logprobs is not None else 5,
        )

        mcts = MCTS(
            expand_num=2,  # Adjust these parameters as needed
            visit_count_threshold=10,
            max_iteration=1000,
        )

        best_node = await mcts.search(initial_state)

        # Calculate total token usage
        input_tokens = len(app.state.tokenizer.tokenize(request.prompt).input_ids)
        completion_tokens = best_node.state.total_token_num - input_tokens

        return CompletionResponse(
            id="cmpl-" + "".join([str(x) for x in range(10)]),  # Generate a unique ID
            object="text_completion",
            created=int(asyncio.get_event_loop().time()),
            model=request.model,
            choices=[
                {
                    "text": best_node.state.total_prompt[len(request.prompt):],
                    "index": 0,
                    "logprobs": None,  # Add logprobs if needed
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": best_node.state.total_token_num,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the reasoning LLM MCTS server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--openai-api-key",
        type=str,
        required=True,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--openai-api-base",
        type=str,
        default="https://api.openai.com/v1",
        help="OpenAI API base URL",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="gpt2",
        help="Name of the tokenizer to use",
    )
    return parser


def init_app(
    openai_api_key: str,
    openai_api_base: str,
    tokenizer_name: str,
) -> None:
    import openai

    app.state.openai_client = openai.AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    app.state.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    init_app(
        openai_api_key=args.openai_api_key,
        openai_api_base=args.openai_api_base,
        tokenizer_name=args.tokenizer_name,
    )

    uvicorn.run(app, host=args.host, port=args.port)