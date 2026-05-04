"""Model adapter registry for provider-specific LLM quirk handling."""
from groundguard.adapters.registry import get_adapter, ModelAdapter

__all__ = ["get_adapter", "ModelAdapter"]
