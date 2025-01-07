import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

import jinja2
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: Optional[str] = None
    function_call: Optional[dict[str, str]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    template_path: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: dict[str, int]


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
                    "text": best_node.state.total_prompt[len(request.prompt) :],
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


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Load and render template
        template = load_template(request.template_path)
        prompt = template.render(
            messages=request.messages,
            add_assistant_prefix=True,  # Add 'Assistant: ' prefix for the response
        )

        # Create initial state
        initial_state = ReasoningState(
            openai_client=app.state.openai_client,
            tokenizer=app.state.tokenizer,
            max_total_tokens=request.max_tokens,
            max_delta_new_tokens=min(50, request.max_tokens),
            text_delta=prompt,
            token_delta_num=len(app.state.tokenizer.tokenize(prompt).input_ids),
            confidence_score=1.0,
            top_logprobs_num=5,  # Fixed for chat completions
        )

        mcts = MCTS(
            expand_num=2,
            visit_count_threshold=10,
            max_iteration=1000,
        )

        best_node = await mcts.search(initial_state)

        # Extract assistant's response (remove the prefix 'Assistant: ' if present)
        response_text = best_node.state.total_prompt[len(prompt) :]
        if response_text.startswith("Assistant: "):
            response_text = response_text[len("Assistant: ") :]

        # Calculate token usage
        input_tokens = len(app.state.tokenizer.tokenize(prompt).input_ids)
        completion_tokens = best_node.state.total_token_num - input_tokens

        return ChatCompletionResponse(
            id="chatcmpl-" + "".join([str(x) for x in range(10)]),
            created=int(asyncio.get_event_loop().time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_text.strip(),
                    ),
                    finish_reason="stop",
                )
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
    parser.add_argument(
        "--template-path",
        type=str,
        help="Path to the chat template file (optional)",
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
