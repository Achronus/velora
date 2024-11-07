from abc import ABC

from pydantic import BaseModel


class Storage(ABC, BaseModel):
    """A base class for agent storage containers."""

    pass
