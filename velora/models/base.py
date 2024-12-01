from abc import ABC

from pydantic import BaseModel


class AgentModel(ABC, BaseModel):
    """A base class for Agent models."""

    pass
