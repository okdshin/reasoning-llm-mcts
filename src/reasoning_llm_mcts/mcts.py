import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


class State(ABC):
    """Abstract base class for MCTS state"""

    @abstractmethod
    async def expand(self, expand_num: int) -> list["State"]:
        """Generate a new state by expanding the current state"""
        raise NotImplementedError

    @abstractmethod
    async def evaluate(self) -> float:
        """Evaluate the current state and return a value"""
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """String representation of the state"""
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self) -> bool:
        return False


@dataclass
class SearchNode:
    state: State
    parent: Optional["SearchNode"] = None
    children: list["SearchNode"] = field(default_factory=list)
    total_value: float = 0.0
    visit_count: int = 0

    def __post_init__(self):
        if not isinstance(self.state, State):
            raise TypeError("state must be an instance of State")

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    async def expand(self, expand_num: int) -> None:
        child_states = await self.state.expand(expand_num=expand_num)
        for child_state in child_states:
            self.children.append(SearchNode(parent=self, state=child_state))

    async def evaluate(self) -> float:
        return await self.state.evaluate()

    def update(self, value: float) -> None:
        self.total_value += value
        self.visit_count += 1

    def ucb1(self, total_visit_count: int) -> float:
        assert total_visit_count >= 0
        if self.visit_count == 0:
            return float("inf")
        value_mean = self.total_value / self.visit_count
        assert total_visit_count != 0
        bias = math.sqrt(2.0 * math.log(total_visit_count / self.visit_count))
        return value_mean + bias


@dataclass
class MCTS:
    expand_num: int = 2
    visit_count_threshold: int = 10
    max_iteration: int = 1000

    async def search(self, initial_state: State) -> SearchNode:
        root_node = SearchNode(state=initial_state)
        for _ in range(self.max_iteration):
            current_node = root_node
            while True:
                if not current_node.is_leaf():
                    current_node = max(
                        current_node.children,
                        key=lambda node: node.ucb1(
                            total_visit_count=root_node.visit_count
                        ),
                    )
                    continue
                assert current_node.is_leaf()
                if (not current_node.is_terminal()) and (
                        current_node.visit_count >= self.visit_count_threshold):
                    await current_node.expand(expand_num=self.expand_num)
                    current_node = root_node
                    continue
                assert current_node.is_terminal() or current_node.visit_count < self.visit_count_threshold
                """
                if current_node.visit_count >= self.visit_count_threshold:
                    await current_node.expand(expand_num=self.expand_num)
                    current_node = root_node
                    continue
                assert current_node.visit_count < self.visit_count_threshold
                """

                value = await current_node.evaluate()
                self.backpropagate(start_node=current_node, value=value)
                break
        return self.get_best_child(start_node=root_node)

    def backpropagate(self, start_node: SearchNode, value: float) -> None:
        current_node = start_node
        while current_node is not None:
            current_node.update(value=value)
            current_node = current_node.parent

    def get_best_child(self, start_node: SearchNode) -> SearchNode:
        current_node = start_node
        while not current_node.is_leaf():
            #print(f"{str(current_node.state)=} {current_node.children=}")
            current_node = max(current_node.children, key=lambda node: node.visit_count)
        return current_node
