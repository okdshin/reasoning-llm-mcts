import argparse
import asyncio
import uuid
import traceback
import sys
from pathlib import Path
from typing import Any, Optional

import jinja2
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from reasoning_llm_mcts.mcts import MCTS
from reasoning_llm_mcts.reasoning_state import ReasoningState


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the reasoning LLM MCTS server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--openai-api-base",
        type=str,
        default="https://api.openai.com/v1",
        help="OpenAI API base URL",
    )
    parser.add_argument(
        "--top-logprobs-num",
        type=int,
        default=5,
        help="Number of top logprobs to consider in the MCTS process",
    )
    parser.add_argument(
        "--max-new-tokens-delta",
        type=int,
        default=32,
        help="Maximum number of new tokens per MCTS step",
    )
    return parser


parser = create_parser()
args = parser.parse_args()
print(args)

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
        # Create initial state with root prompt
        initial_state = ReasoningState(
            api_base_url=args.openai_api_base,
            max_total_tokens=request.max_tokens,
            max_new_tokens_delta=args.max_new_tokens_delta,
            root_prompt=request.prompt,
            top_logprobs_num=args.top_logprobs_num,
        )

        mcts = MCTS(
            expand_num=2,  # Adjust these parameters as needed
            visit_count_threshold=10,
            max_iteration=1000,
        )

        best_node = await mcts.search(initial_state)
        print(f"{best_node.state.total_prompt=}")

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4()}",  # Generate a unique ID
            object="text_completion",
            created=int(asyncio.get_event_loop().time()),
            model=request.model,
            choices=[
                {
                    "text": best_node.state.total_prompt[len(request.prompt) :],
                    "index": 0,
                    "logprobs": None,  # Add logprobs if needed
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": 0,  # TODO
                "completion_tokens": best_node.state.total_new_token_num,
                "total_tokens": best_node.state.total_new_token_num,
            },
        )
    except Exception as e:
        # スタックトレースを標準出力に出力
        traceback.print_exc(file=sys.stdout)
        # エラーの詳細情報をログに残す
        print(f"Error details: {str(e)}", file=sys.stdout)
        # HTTPエラーを発生させる
        raise HTTPException(status_code=500, detail=str(e))


def load_template(template_path: Optional[str] = None) -> jinja2.Template:
    if template_path is None:
        # Use default template
        template_dir = Path(__file__).parent / "templates"
        template_path = template_dir / "default.j2"
    else:
        template_path = Path(template_path)
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

    template_loader = jinja2.FileSystemLoader(template_path.parent)
    template_env = jinja2.Environment(loader=template_loader)
    return template_env.get_template(template_path.name)


def run_server() -> None:
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run_server()
