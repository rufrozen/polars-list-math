import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[2] / "polars_list_math" / "typed_polars"


def test_typed_polars_has_no_function_local_imports() -> None:
    violations: list[str] = []
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            for node in ast.walk(function):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    violations.append(f"{path.name}:{node.lineno}")

    assert violations == []


def test_typed_polars_internal_import_graph_is_acyclic() -> None:
    module_names = {path.stem for path in PACKAGE.glob("*.py")}
    graph: dict[str, set[str]] = {name: set() for name in module_names}
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                dependency = node.module.split(".", maxsplit=1)[0]
                if dependency in module_names:
                    graph[path.stem].add(dependency)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str) -> None:
        if module in visiting:
            raise AssertionError(f"Cyclic typed_polars import through {module}")
        if module in visited:
            return
        visiting.add(module)
        for dependency in graph[module]:
            visit(dependency)
        visiting.remove(module)
        visited.add(module)

    for module in graph:
        visit(module)
