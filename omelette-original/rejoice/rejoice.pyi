from typing import Optional, Callable, Any

class EGraph:
    def __init__(self, eval: Optional[Callable] = None) -> None: ...

    def add(self, expr: Any) -> int:
        """Add an expression to the egraph"""

    def union(self, *exprs) -> bool:
        """"""

    def rebuild(self) -> int:
        """Rebuilds the egraph."""

    def extract(self, *exprs) -> list:
        """Extracts the best solution from the EGraph"""

    def classes(self) -> dict[int, tuple[Any, list[Any]]]:
        """Dict of type (class_id, (data, [nodes]))"""

    def total_size(self) -> int:
        """Size of the hascons"""