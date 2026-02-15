"""Basic sanity checks for the environment setup."""
import sys


def test_python_version() -> None:
    assert sys.version_info >= (3, 11), "Python 3.11+ is required"


def test_imports() -> None:
    import importlib

    required = ["pydantic", "dotenv", "httpx", "tenacity"]
    for pkg in required:
        assert importlib.util.find_spec(pkg) is not None, f"{pkg} not installed"


def test_src_importable() -> None:
    import src  # noqa: F401
