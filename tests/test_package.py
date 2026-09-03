import pydantic
import typer
import yaml

from agent_bench import __version__


def test_package_has_version() -> None:
    assert __version__ == "0.1.0"


def test_runtime_dependencies_import() -> None:
    assert pydantic.__version__
    assert typer.__version__
    assert yaml.__version__
