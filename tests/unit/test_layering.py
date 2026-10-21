import ast
from functools import cache
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
HTTP_LAYER = ("app.api", "fastapi", "starlette")
FORBIDDEN_IMPORTS = {
    "app.domain": ("app.pipeline", "app.graph", "app.llm", *HTTP_LAYER),
    "app.pipeline": HTTP_LAYER,
    "app.graph": HTTP_LAYER,
}


def _module_name(path: Path) -> str:
    parts = path.relative_to(APP_ROOT.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


MODULES = {_module_name(path): path for path in APP_ROOT.rglob("*.py")}


@cache
def _direct_imports(module: str) -> frozenset[str]:
    tree = ast.parse(MODULES[module].read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    return frozenset(imported)


def _transitive_imports(module: str) -> set[str]:
    seen: set[str] = set()
    pending = [module]
    while pending:
        current = pending.pop()
        for name in _direct_imports(current):
            if name not in seen:
                seen.add(name)
                if name in MODULES:
                    pending.append(name)
    return seen


def _within(name: str, package: str) -> bool:
    return name == package or name.startswith(f"{package}.")


def test_every_layer_has_modules() -> None:
    for package in FORBIDDEN_IMPORTS:
        assert any(_within(module, package) for module in MODULES)


@pytest.mark.parametrize(
    "module",
    sorted(m for m in MODULES if any(_within(m, package) for package in FORBIDDEN_IMPORTS)),
)
def test_layer_does_not_depend_on_outer_layers(module: str) -> None:
    package = next(p for p in FORBIDDEN_IMPORTS if _within(module, p))
    forbidden = FORBIDDEN_IMPORTS[package]
    violations = sorted(
        name
        for name in _transitive_imports(module)
        if any(_within(name, outer) for outer in forbidden)
    )
    assert violations == []
