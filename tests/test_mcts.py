import math
import random
from typing import Optional

import pytest

from reasoning_llm_mcts.mcts import MCTS, SearchNode, State


# テスト用のState実装
class NumberSearchState(State):
    def __init__(self, value: float, target: float = 0.5):
        self.value = value
        self.target = target

    async def expand(self, expand_num) -> "State":
        # 現在の値から少しずらした新しい値を生成
        child_states = []
        for _ in range(expand_num):
            delta = random.uniform(-0.1, 0.1)
            new_value = max(0, min(1, self.value + delta))
            child_states.append(NumberSearchState(new_value, self.target))
        return child_states

    async def evaluate(self) -> float:
        # targetに近いほど高い評価値を返す
        return 1.0 - abs(self.value - self.target)

    def __str__(self) -> str:
        return f"NumberSearchState(value={self.value:.3f}, target={self.target:.3f})"


@pytest.mark.asyncio
async def test_state_methods():
    state = NumberSearchState(0.3)
    # expandのテスト
    new_states = await state.expand(expand_num=2)
    assert len(new_states) == 2
    assert isinstance(new_states[0], NumberSearchState)
    assert 0 <= new_states[0].value <= 1

    # evaluateのテスト
    value = await state.evaluate()
    assert 0 <= value <= 1


@pytest.mark.asyncio
async def test_search_node_initialization():
    state = NumberSearchState(0.3)

    # 正常なケース
    node = SearchNode(state=state)
    assert node.parent is None
    assert len(node.children) == 0
    assert node.total_value == 0.0
    assert node.visit_count == 0

    # 不正なstate型のケース
    with pytest.raises(TypeError):
        SearchNode(state="not a state")


@pytest.mark.asyncio
async def test_search_node_expansion():
    state = NumberSearchState(0.3)
    node = SearchNode(state=state)

    await node.expand(expand_num=2)
    assert len(node.children) == 2
    for child in node.children:
        assert isinstance(child.state, NumberSearchState)
        assert child.parent == node


@pytest.mark.asyncio
async def test_node_evaluation():
    state = NumberSearchState(0.3)
    node = SearchNode(state=state)

    value = await node.evaluate()
    assert 0 <= value <= 1


@pytest.mark.asyncio
async def test_ucb1_calculation():
    state = NumberSearchState(0.3)
    node = SearchNode(state=state)

    # 未訪問ノードの場合
    assert node.ucb1(total_visit_count=1) == float("inf")

    # 訪問済みノードの場合
    node.update(0.5)
    ucb1_value = node.ucb1(total_visit_count=2)
    assert isinstance(ucb1_value, float)
    assert not math.isinf(ucb1_value)


@pytest.mark.asyncio
async def test_mcts_search():
    initial_state = NumberSearchState(0.1)
    mcts = MCTS(expand_num=2, visit_count_threshold=3, max_iteration=10)

    best_child = await mcts.search(initial_state)

    # 基本的な検証
    assert isinstance(best_child, SearchNode)


@pytest.mark.asyncio
async def test_mcts_convergence():
    target = 0.5
    initial_state = NumberSearchState(0.1, target)
    mcts = MCTS(expand_num=3, visit_count_threshold=5, max_iteration=5000)

    best_child = await mcts.search(initial_state)
    best_value = best_child.state.value

    # 目標値への収束を確認
    print(f"{best_value=}")
    assert abs(best_value - target) < 0.3


@pytest.mark.asyncio
async def test_mcts_backward():
    initial_state = NumberSearchState(0.3)
    mcts = MCTS()

    # ノードの連鎖を作成
    root = SearchNode(state=initial_state)
    child = SearchNode(state=initial_state, parent=root)
    grandchild = SearchNode(state=initial_state, parent=child)

    # バックワード更新をテスト
    test_value = 0.5
    mcts.backpropagate(start_node=grandchild, value=test_value)

    # 全ノードが更新されていることを確認
    for node in [grandchild, child, root]:
        assert node.visit_count == 1
        assert node.total_value == test_value


@pytest.mark.asyncio
async def test_error_handling():
    class ErrorState(State):
        async def expand(self, parent: Optional["SearchNode"]) -> "State":
            raise ValueError("Expansion error")

        async def evaluate(self) -> float:
            raise ValueError("Evaluation error")

        def __str__(self) -> str:
            return "ErrorState"

    error_state = ErrorState()
    mcts = MCTS(max_iteration=5)

    with pytest.raises(ValueError):
        await mcts.search(error_state)
