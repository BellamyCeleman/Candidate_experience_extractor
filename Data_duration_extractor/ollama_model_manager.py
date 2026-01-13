"""
Utilities for working with local Ollama models.

Goal: ensure a required model (e.g. "phi4") exists on the device.
If it's missing, we can pull it automatically.
"""

from __future__ import annotations

from typing import Any


def _normalize_model_name(name: str) -> str:
    """
    Ollama model names often look like "phi4:latest".
    Users may pass "phi4" or "phi4:latest" — we treat them as compatible.
    """
    name = (name or "").strip()
    return name


def _matches_model(requested: str, available: str) -> bool:
    requested = _normalize_model_name(requested)
    available = _normalize_model_name(available)

    if not requested or not available:
        return False

    # Exact match covers "phi4:latest" vs "phi4:latest"
    if available == requested:
        return True

    # If user passed "phi4", accept any tag like "phi4:latest"
    if ":" not in requested and available.startswith(requested + ":"):
        return True

    return False


def is_ollama_model_available(model_name: str) -> bool:
    """
    Returns True if Ollama reports that the model exists locally.
    Raises RuntimeError if Ollama is unreachable.
    """
    try:
        import ollama  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Python package 'ollama' is not available. Install it first (pip install ollama)."
        ) from e

    try:
        data: Any = ollama.list()
    except Exception as e:
        raise RuntimeError(
            "Ollama is not reachable. Make sure the Ollama app/service is running."
        ) from e

    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return False

    for m in models:
        if isinstance(m, dict):
            name = m.get("name") or m.get("model")
        else:
            name = None
        if isinstance(name, str) and _matches_model(model_name, name):
            return True

    return False


def ensure_ollama_model(model_name: str, *, pull_if_missing: bool = True) -> bool:
    """
    Ensures the model is present locally.

    Returns:
        True if model exists (or was successfully pulled).
        False if it's missing and pull_if_missing=False.
    """
    model_name = _normalize_model_name(model_name)
    if not model_name:
        raise ValueError("model_name must be a non-empty string")

    if is_ollama_model_available(model_name):
        return True

    if not pull_if_missing:
        return False

    try:
        import ollama  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Python package 'ollama' is not available. Install it first (pip install ollama)."
        ) from e

    try:
        # Non-streaming pull; Ollama will download the model if missing.
        ollama.pull(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to pull Ollama model '{model_name}'. Ensure you have internet access and Ollama is running."
        ) from e

    return is_ollama_model_available(model_name)

